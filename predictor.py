import gc
import math
import os
import re
from functools import lru_cache
from spellchecker import SpellChecker
import time
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_NAME = "facebook/opt-125m"
ADAPTER_DIR = "./models/opt-125m-lora-adapter"

PROFANITY_BLOCKLIST = {
    "damn", "hell", "shit", "fuck", "bitch", "bastard", "asshole", "crap", "idiot"
}

VARIANT_METADATA = {
    "lora_int8": {
        "label": "LoRA Fine-Tuned OPT-125M (INT8 Quantized)",
        "size_label": "~271.7 MB (INT8 in-memory)",
    },
    "lora_fp32": {
        "label": "LoRA Fine-Tuned OPT-125M (FP32)",
        "size_label": "500.8 MB (FP32)",
    },
    "base": {
        "label": "Base Pre-trained OPT-125M (Zero-Shot FP32)",
        "size_label": "500.8 MB (FP32)",
    },
}


@lru_cache(maxsize=1)
def english_spellchecker():
    return SpellChecker(language="en", distance=1)


def spelling_candidates(word, limit=5):
    """Suggest edits for unknown English words without changing user text."""
    if not word.isascii() or not word.isalpha() or not 3 <= len(word) <= 24:
        return []
    spell = english_spellchecker()
    lower = word.lower()
    if spell.known([lower]):
        return []
    candidates = spell.candidates(lower) or set()
    ranked = sorted(
        (w for w in candidates if w != lower and w not in PROFANITY_BLOCKLIST),
        key=lambda w: (-spell.word_usage_frequency(w), w),
    )[:limit]
    if word.isupper():
        return [w.upper() for w in ranked]
    if word.istitle():
        return [w.capitalize() for w in ranked]
    return ranked


class OPTKeyboardPredictor:
    def __init__(self, model, tokenizer, device: str = "cpu", variant: str = "lora_int8"):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.variant = variant
        self.variant_info = VARIANT_METADATA.get(variant, {})
        self.user_freq_cache = {}
        self._build_vocabulary_index()

    @classmethod
    def from_variant(cls, variant: str = "lora_int8", device: str = "cpu"):
        tok_source = ADAPTER_DIR if os.path.isdir(ADAPTER_DIR) else BASE_MODEL_NAME
        tokenizer = AutoTokenizer.from_pretrained(tok_source)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Explicitly force Float32 on CPU so INT8 dynamic quantization never hits a Half/Float mismatch
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float32,
        ).float().cpu()

        if variant == "base":
            model = base_model
        elif variant in ("lora_fp32", "lora_int8"):
            if not os.path.isdir(ADAPTER_DIR):
                raise FileNotFoundError(
                    f"LoRA adapter directory not found at '{ADAPTER_DIR}'. "
                    "Ensure 'models/opt-125m-lora-adapter/' is uploaded to your repository."
                )
            peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
            merged_model = peft_model.merge_and_unload().float().cpu()

            del peft_model
            del base_model
            gc.collect()

            if variant == "lora_fp32":
                model = merged_model
            else:
                device = "cpu"
                engines = torch.backends.quantized.supported_engines
                if "fbgemm" in engines:
                    torch.backends.quantized.engine = "fbgemm"
                elif "qnnpack" in engines:
                    torch.backends.quantized.engine = "qnnpack"

                try:
                    q_model = torch.ao.quantization.quantize_dynamic(
                        merged_model,
                        {torch.nn.Linear},
                        dtype=torch.qint8,
                    )
                    q_model.eval()
                    # Verify CPU quantized kernel works in this container
                    with torch.no_grad():
                        q_model(torch.tensor([[2]], dtype=torch.long))
                    model = q_model
                    del merged_model
                    gc.collect()
                except Exception:
                    # Graceful fallback to Float32 merged model if container lacks qint8 CPU ops
                    model = merged_model
        else:
            raise ValueError(f"Unknown variant: {variant}")

        return cls(model=model, tokenizer=tokenizer, device=device, variant=variant)

    def _build_vocabulary_index(self):
        vocab = self.tokenizer.get_vocab()
        self.whole_word_ids = []
        self.id_to_word = {}
        self.word_to_id = {}

        for token_str, token_id in vocab.items():
            if token_str.startswith("Ġ"):
                clean_word = token_str[1:]
                if (
                    clean_word.isalpha()
                    and len(clean_word) >= 1
                    and clean_word.lower() not in PROFANITY_BLOCKLIST
                ):
                    self.whole_word_ids.append(token_id)
                    self.id_to_word[token_id] = clean_word
                    lower_w = clean_word.lower()
                    if lower_w not in self.word_to_id:
                        self.word_to_id[lower_w] = token_id

        self.whole_word_tensor = torch.tensor(self.whole_word_ids, dtype=torch.long)

    def update_user_cache(self, text: str):
        words = [w.strip(".,!?;:'\"") for w in text.split()]
        for w in words:
            if w.isalpha() and len(w) > 1:
                key = w.lower()
                self.user_freq_cache[key] = self.user_freq_cache.get(key, 0) + 1

    def spelling_suggestion(self, text, context_before=""):
        """Check the last completed word, preserving its exact character span."""
        match = re.search(r"(?<![\w'’@/.-])([A-Za-z]+)([,.!?;:]*\s+)$", text)
        if match is None:
            return None

        word = match.group(1)

        # Skip words the user has already taught the keyboard via sent/accepted messages
        if word.lower() in self.user_freq_cache:
            return None

        candidates = spelling_candidates(word)
        if not candidates:
            return None

        prefix = context_before + text[:match.start(1)]
        has_prefix = bool(prefix.strip())
        base_left = (
            self.tokenizer.encode(prefix.rstrip(), add_special_tokens=False)
            if has_prefix
            else []
        )
        bos = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else self.tokenizer.eos_token_id
        )
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else (bos if bos is not None else 0)
        )

        # Build padded batch so all candidates are scored in a single forward pass
        valid_candidates = []
        batch_seqs = []
        batch_spans = []

        for candidate in candidates:
            suffix = self.tokenizer.encode(
                " " + candidate if has_prefix else candidate,
                add_special_tokens=False,
            )
            if not suffix:
                continue
            left = base_left[-max(1, 127 - len(suffix)):] if base_left else [bos]
            seq = left + suffix
            valid_candidates.append(candidate)
            batch_seqs.append(seq)
            batch_spans.append((len(left), suffix))

        if not batch_seqs:
            return None

        max_len = max(len(seq) for seq in batch_seqs)
        padded_ids = [seq + [pad_id] * (max_len - len(seq)) for seq in batch_seqs]
        attn_masks = [[1] * len(seq) + [0] * (max_len - len(seq)) for seq in batch_seqs]

        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(attn_masks, dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits.float()
            scores = torch.log_softmax(logits, dim=-1)

        scored = []
        for idx, (candidate, (left_len, suffix)) in enumerate(
            zip(valid_candidates, batch_spans)
        ):
            score = sum(
                scores[idx, left_len + i - 1, tid].item()
                for i, tid in enumerate(suffix)
            ) / len(suffix)
            scored.append((candidate, score))

        ranked = [cand for cand, _ in sorted(scored, key=lambda pair: -pair[1])]
        return {
            "word": word,
            "start": match.start(1),
            "end": match.end(1),
            "candidates": ranked[:3],
        }

    def clear_user_cache(self):
        self.user_freq_cache.clear()

    def _get_next_token_log_probs(self, context_text: str) -> torch.Tensor:
        if not context_text.strip():
            context_text = self.tokenizer.bos_token or "</s>"

        inputs = self.tokenizer(
            context_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            next_logits = outputs.logits[0, -1, :].float().cpu()
            log_probs = torch.log_softmax(next_logits, dim=-1)
        return log_probs

    def suggest(self, input_text: str, top_k: int = 3, cache_weight: float = 0.35):
        start_time = time.perf_counter()

        if not input_text:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return [("I", 0.32), ("Hi", 0.25), ("Hey", 0.18)], round(latency_ms, 2), "Starter"

        if input_text.endswith(" "):
            mode = "Next-Word"
            context_prefix = input_text.rstrip()
            partial_prefix = ""
        else:
            mode = "Prefix-Completion"
            parts = input_text.rsplit(" ", 1)
            if len(parts) == 2:
                context_prefix, partial_prefix = parts[0], parts[1]
            else:
                context_prefix, partial_prefix = "", parts[0]

        log_probs = self._get_next_token_log_probs(context_prefix)

        if self.user_freq_cache and cache_weight > 0:
            for cached_word, count in self.user_freq_cache.items():
                tok_id = self.word_to_id.get(cached_word)
                if tok_id is not None:
                    log_probs[tok_id] += cache_weight * math.log(1.0 + count)

        if mode == "Next-Word":
            candidate_ids = self.whole_word_tensor
        else:
            p_lower = partial_prefix.lower()
            matched_ids = [
                tid
                for tid, w in self.id_to_word.items()
                if w.lower().startswith(p_lower) and len(w) > len(partial_prefix)
            ]
            if not matched_ids:
                custom_matches = [
                    (w, 0.90)
                    for w in self.user_freq_cache
                    if w.startswith(p_lower) and len(w) > len(partial_prefix)
                ]
                latency_ms = (time.perf_counter() - start_time) * 1000
                return custom_matches[:top_k], round(latency_ms, 2), mode
            candidate_ids = torch.tensor(matched_ids, dtype=torch.long)

        candidate_scores = log_probs[candidate_ids]
        k = min(top_k, len(candidate_ids))
        top_scores, top_indices = torch.topk(candidate_scores, k=k)

        probs = torch.softmax(top_scores, dim=-1).tolist()
        suggestions = []
        for idx, prob in zip(top_indices.tolist(), probs):
            token_id = candidate_ids[idx].item()
            word = self.id_to_word[token_id]
            if partial_prefix and partial_prefix[0].isupper():
                word = word.capitalize()
            suggestions.append((word, round(prob, 4)))

        latency_ms = (time.perf_counter() - start_time) * 1000
        return suggestions, round(latency_ms, 2), mode
