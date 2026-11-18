import gc
import math
import os
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
        "size_label": "~245.0 MB (INT8 in-memory)",
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


class OPTKeyboardPredictor:
    def __init__(self, model, tokenizer, device: str = "cpu", variant: str = "lora_int8"):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.variant = variant
        self.variant_info = VARIANT_METADATA.get(variant, {})
        self.user_freq_cache = {}  # Tracks user-specific word frequencies
        self._build_vocabulary_index()

    @classmethod
    def from_variant(cls, variant: str = "lora_int8", device: str = "cpu"):
        """
        Reconstructs the selected model variant from base OPT-125M + the lightweight
        LoRA adapter (~1.2 MB) so no >100 MB weight files are needed in GitHub:
        - 'base': Zero-shot facebook/opt-125m (FP32)
        - 'lora_fp32': Base + merged LoRA adapter (FP32)
        - 'lora_int8': Base + merged LoRA adapter + 8-bit Dynamic Quantization (INT8, CPU)
        """
        tok_source = ADAPTER_DIR if os.path.isdir(ADAPTER_DIR) else BASE_MODEL_NAME
        tokenizer = AutoTokenizer.from_pretrained(tok_source)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if variant == "base":
            model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
        elif variant in ("lora_fp32", "lora_int8"):
            if not os.path.isdir(ADAPTER_DIR):
                raise FileNotFoundError(
                    f"LoRA adapter directory not found at '{ADAPTER_DIR}'. "
                    "Ensure 'models/opt-125m-lora-adapter/' is uploaded to your repository."
                )
            base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
            peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
            merged_model = peft_model.merge_and_unload().cpu()

            # Free intermediate references to keep RAM usage low on Streamlit Cloud
            del peft_model
            del base_model
            gc.collect()

            if variant == "lora_fp32":
                model = merged_model
            else:
                device = "cpu"  # PyTorch dynamic quantization executes on CPU
                model = torch.ao.quantization.quantize_dynamic(
                    merged_model,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                )
                del merged_model
                gc.collect()
        else:
            raise ValueError(f"Unknown variant: {variant}")

        return cls(model=model, tokenizer=tokenizer, device=device, variant=variant)

    def _build_vocabulary_index(self):
        """
        Pre-indexes valid 'Ġ'-prefixed whole-word BPE tokens for sub-50ms inference.
        """
        vocab = self.tokenizer.get_vocab()
        self.whole_word_ids = []
        self.id_to_word = {}
        self.word_to_id = {}

        for token_str, token_id in vocab.items():
            # OPT uses 'Ġ' to mark the start of a new space-separated word
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
        """Updates personal word frequency counts from confirmed user words."""
        words = [w.strip(".,!?;:'\"") for w in text.split()]
        for w in words:
            if w.isalpha() and len(w) > 1:
                key = w.lower()
                self.user_freq_cache[key] = self.user_freq_cache.get(key, 0) + 1

    def clear_user_cache(self):
        """Clears personalized word frequencies."""
        self.user_freq_cache.clear()

    def _get_next_token_log_probs(self, context_text: str) -> torch.Tensor:
        """Runs a single forward pass and returns log-probabilities for the next token."""
        if not context_text.strip():
            context_text = self.tokenizer.bos_token

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
        """
        Generates top-k word suggestions + latency diagnostics.
        - Trailing space -> Next-Word Prediction
        - Partial word   -> Mid-Word Prefix Completion
        """
        start_time = time.perf_counter()

        # Mode A: Empty input -> common conversation starters
        if not input_text:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return [("I", 0.32), ("Hi", 0.25), ("Hey", 0.18)], round(latency_ms, 2), "Starter"

        # Determine whether user is mid-word or ready for next word
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

        # Apply adaptive user personalization boost: log_p + lambda * log(1 + count)
        if self.user_freq_cache and cache_weight > 0:
            for cached_word, count in self.user_freq_cache.items():
                tok_id = self.word_to_id.get(cached_word)
                if tok_id is not None:
                    log_probs[tok_id] += cache_weight * math.log(1.0 + count)

        # Filter candidate whole-word tokens
        if mode == "Next-Word":
            candidate_ids = self.whole_word_tensor
        else:
            p_lower = partial_prefix.lower()
            matched_ids = [
                tid
                for tid, w in self.id_to_word.items()
                if w.lower().startswith(p_lower) and len(w) > len(partial_prefix)
            ]
            # Fallback if prefix is a custom cached word not in base BPE whole-words
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

        # Normalize top-k probabilities for UI confidence display
        probs = torch.softmax(top_scores, dim=-1).tolist()
        suggestions = []
        for idx, prob in zip(top_indices.tolist(), probs):
            token_id = candidate_ids[idx].item()
            word = self.id_to_word[token_id]
            # Preserve capitalization if user typed uppercase prefix
            if partial_prefix and partial_prefix[0].isupper():
                word = word.capitalize()
            suggestions.append((word, round(prob, 4)))

        latency_ms = (time.perf_counter() - start_time) * 1000
        return suggestions, round(latency_ms, 2), mode
