#!/usr/bin/env python3
"""Local prediction server for OPT-125M + LoRA mobile keyboard."""

from __future__ import annotations

import json
import math
import mimetypes
import os
import re
import threading
import time
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "web"
ADAPTER_DIR = ROOT_DIR / "models" / "opt-125m-lora-adapter"
SPELLING_PATH = ROOT_DIR / "browser" / "assets" / "spelling.json"
BASE_MODEL_ID = os.environ.get("OPT_BASE_MODEL", "facebook/opt-125m")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8765"))

MAX_CONTEXT = 128
MAX_REQUEST_BYTES = 512 * 1024
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
BLOCKLIST = {
    "damn", "hell", "shit", "fuck", "bitch", "bastard", "asshole", "crap", "idiot",
}
COMPLETED_WORD_RE = re.compile(r"(?<![A-Za-z0-9_'’@/.-])([A-Za-z]+)(?=[,.!?;:]*\s)")


def capitalize_word(word: str) -> str:
    return word[0].upper() + word[1:].lower() if word else word


def load_spelling_dictionary(vocab_words: list[tuple[str, int, str]]) -> dict[str, int]:
    if SPELLING_PATH.is_file():
        with SPELLING_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and data:
            return {str(k).lower(): int(v) for k, v in data.items()}

    # Fallback frequency table derived from BPE merge order (lower token ID = more common).
    dictionary: dict[str, int] = {}
    max_id = max((tok_id for _, tok_id, _ in vocab_words), default=50265)
    for _, tok_id, lower in vocab_words:
        if len(lower) >= 2 and lower.isascii() and lower.isalpha():
            rank = max(1, max_id - tok_id)
            if rank > dictionary.get(lower, 0):
                dictionary[lower] = rank
    return dictionary


def spelling_candidates(word: str, dictionary: dict[str, int]) -> list[str]:
    lower = word.lower()
    if len(lower) < 3 or len(lower) > 24 or lower in dictionary:
        return []

    edits: set[str] = set()
    for i in range(len(lower) + 1):
        left = lower[:i]
        right = lower[i:]
        if right:
            edits.add(left + right[1:])
        if len(right) > 1:
            edits.add(left + right[1] + right[0] + right[2:])
        for ch in ALPHABET:
            edits.add(left + ch + right)
            if right:
                edits.add(left + ch + right[1:])

    matches = [e for e in edits if e in dictionary and e not in BLOCKLIST]
    matches.sort(key=lambda e: (-dictionary[e], e))
    matches = matches[:5]

    if word.isupper():
        return [m.upper() for m in matches]
    if len(word) > 1 and word[0].isupper() and word[1:].islower():
        return [capitalize_word(m) for m in matches]
    return matches


def normalize_variant(variant: str | None) -> tuple[bool, bool, str]:
    raw = (variant or "lora-int8").strip().lower()
    use_lora = not ("base" in raw and "lora" not in raw and "fine" not in raw and "ft" not in raw)
    use_int8 = "int8" in raw or "quant" in raw
    key = f"{'lora' if use_lora else 'base'}-{'int8' if use_int8 else 'fp32'}"
    return use_lora, use_int8, key


def estimate_footprint(model: torch.nn.Module, use_lora: bool, use_int8: bool) -> str:
    total_bytes = 0
    for tensor in model.state_dict().values():
        if isinstance(tensor, torch.Tensor):
            total_bytes += tensor.numel() * tensor.element_size()
    mb = total_bytes / (1024 * 1024)
    if use_int8 and mb < 50:
        # Packed quantized linear weights are not always reported as raw tensors in state_dict().
        mb = 128.0
    label = f"{'LoRA' if use_lora else 'Base'} {'INT8' if use_int8 else 'FP32'}"
    return f"{mb:.0f} MB ({label})"


class KeyboardEngine:
    def __init__(self, use_lora: bool, use_int8: bool) -> None:
        self.use_lora = use_lora
        self.use_int8 = use_int8
        self.device = torch.device("cuda" if torch.cuda.is_available() and not use_int8 else "cpu")

        tokenizer_source = str(ADAPTER_DIR) if (
            use_lora and (ADAPTER_DIR / "tokenizer_config.json").is_file()
        ) else BASE_MODEL_ID
        print(f"[model] Loading tokenizer from {tokenizer_source}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        self.bos_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 2
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 1

        print(f"[model] Loading base model {BASE_MODEL_ID}...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID)

        if use_lora:
            self._validate_adapter_files()
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("Missing 'peft' package. Install it with: pip install peft") from exc

            print(f"[model] Applying LoRA adapter from {ADAPTER_DIR}...", flush=True)
            peft_model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
            model = peft_model.merge_and_unload()

        model.eval()

        actual_int8 = False
        if use_int8:
            try:
                quantize_fn = getattr(torch.ao.quantization, "quantize_dynamic", None)
                if quantize_fn is None:
                    quantize_fn = torch.quantization.quantize_dynamic
                model = quantize_fn(model, {torch.nn.Linear}, dtype=torch.qint8)
                actual_int8 = True
                print("[model] Applied dynamic INT8 quantization.", flush=True)
            except Exception as exc:
                print(f"[model] INT8 quantization unavailable ({exc}); falling back to FP32.", flush=True)

        self.use_int8 = actual_int8
        self.model = model.to(self.device)
        self.footprint = estimate_footprint(self.model, self.use_lora, self.use_int8)

        vocab_words: list[tuple[str, int, str]] = []
        for token, tok_id in self.tokenizer.get_vocab().items():
            if token.startswith("Ġ"):
                word = token[1:]
                lower = word.lower()
                if word.isalpha() and lower not in BLOCKLIST:
                    vocab_words.append((word, int(tok_id), lower))

        self.words = [w for w, _, _ in vocab_words]
        self.words_lower = [low for _, _, low in vocab_words]
        self.word_ids = torch.tensor([tid for _, tid, _ in vocab_words], dtype=torch.long)
        self.lower_to_indices: dict[str, list[int]] = {}
        for idx, lower in enumerate(self.words_lower):
            self.lower_to_indices.setdefault(lower, []).append(idx)

        self.punctuation: list[tuple[str, int]] = []
        for symbol in (".", ",", "?", "!"):
            tok_id = self.tokenizer.convert_tokens_to_ids(symbol)
            if isinstance(tok_id, int) and tok_id >= 0:
                self.punctuation.append((symbol, tok_id))

        self.dictionary = load_spelling_dictionary(vocab_words)
        self.logit_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.spelling_cache: OrderedDict[tuple[str, str], list[str]] = OrderedDict()
        self.typo_cache: OrderedDict[str, bool] = OrderedDict()
        print(f"[model] Ready: {self.footprint} on {self.device}", flush=True)

    @staticmethod
    def _validate_adapter_files() -> None:
        config_path = ADAPTER_DIR / "adapter_config.json"
        weight_paths = [
            p for p in (ADAPTER_DIR / "adapter_model.safetensors", ADAPTER_DIR / "adapter_model.bin")
            if p.is_file()
        ]
        if not config_path.is_file() or not weight_paths:
            raise FileNotFoundError(
                f"LoRA adapter files missing in '{ADAPTER_DIR}'. "
                "Expected adapter_config.json and adapter_model.safetensors (or adapter_model.bin)."
            )
        if all(p.stat().st_size < 1024 for p in weight_paths):
            raise RuntimeError(
                f"LoRA weights in '{ADAPTER_DIR}' appear to be a Git LFS pointer (< 1 KB). "
                "Run 'git lfs pull' or download the real weights file."
            )

    def is_detectable_typo(self, word: str) -> bool:
        lower = word.lower()
        if lower in self.dictionary:
            return False
        cached = self.typo_cache.get(lower)
        if cached is not None:
            return cached
        if len(self.typo_cache) >= 1024:
            self.typo_cache.popitem(last=False)
        result = len(spelling_candidates(lower, self.dictionary)) > 0
        self.typo_cache[lower] = result
        return result

    @torch.inference_mode()
    def next_logits(self, prefix: str) -> torch.Tensor:
        key = prefix if prefix.strip() else "</s>"
        if key in self.logit_cache:
            self.logit_cache.move_to_end(key)
            return self.logit_cache[key]

        if prefix.strip():
            ids = self.tokenizer.encode(prefix, add_special_tokens=True)[-MAX_CONTEXT:]
        else:
            ids = [self.bos_id]

        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits = self.model(input_ids=input_ids).logits[0, -1].float().cpu()

        if len(self.logit_cache) >= 16:
            self.logit_cache.popitem(last=False)
        self.logit_cache[key] = logits
        return logits

    @torch.inference_mode()
    def word_corrections(self, word: str, prefix: str) -> list[str]:
        cache_key = (word, prefix)
        if cache_key in self.spelling_cache:
            self.spelling_cache.move_to_end(cache_key)
            return self.spelling_cache[cache_key]

        candidates = spelling_candidates(word, self.dictionary)
        if len(candidates) < 2:
            return candidates

        left = prefix.rstrip()
        left_ids = self.tokenizer.encode(left, add_special_tokens=True) if left else [self.bos_id]
        alternatives = [
            self.tokenizer.encode((" " if left else "") + cand, add_special_tokens=False)
            for cand in candidates
        ]
        longest = max(len(ids) for ids in alternatives)
        context_ids = left_ids[-(MAX_CONTEXT - longest):] if MAX_CONTEXT > longest else [self.bos_id]
        ctx_len = len(context_ids)

        sequences = [context_ids + alt for alt in alternatives]
        max_len = max(len(seq) for seq in sequences)
        batch_ids = [seq + [self.pad_id] * (max_len - len(seq)) for seq in sequences]
        batch_mask = [[1] * len(seq) + [0] * (max_len - len(seq)) for seq in sequences]

        input_ids = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(batch_mask, dtype=torch.long, device=self.device)
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits.float().cpu()

        scored: list[tuple[float, str]] = []
        for i, alt_ids in enumerate(alternatives):
            total = 0.0
            for j, tok_id in enumerate(alt_ids):
                step_logits = logits[i, ctx_len - 1 + j]
                log_probs = torch.log_softmax(step_logits, dim=-1)
                total += float(log_probs[tok_id])
            scored.append((total / max(1, len(alt_ids)), candidates[i]))

        scored.sort(key=lambda item: item[0], reverse=True)
        ranked = [cand for _, cand in scored[:3]]
        if len(self.spelling_cache) >= 64:
            self.spelling_cache.popitem(last=False)
        self.spelling_cache[cache_key] = ranked
        return ranked

    @staticmethod
    def _top_suggestions(words: list[str], scores: torch.Tensor, limit: int = 3) -> list[dict[str, Any]]:
        if not words:
            return []
        k = min(len(words), limit * 4)
        top_vals, top_pos = torch.topk(scores, k=k)
        picked: list[tuple[str, float]] = []
        seen: set[str] = set()
        for val, pos in zip(top_vals.tolist(), top_pos.tolist()):
            word = words[pos]
            key = word.lower()
            if key in seen:
                continue
            seen.add(key)
            picked.append((word, val))
            if len(picked) == limit:
                break
        if not picked:
            return []
        max_score = picked[0][1]
        weights = [math.exp(score - max_score) for _, score in picked]
        total = sum(weights) or 1.0
        return [{"word": word, "probability": weight / total} for (word, _), weight in zip(picked, weights)]

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        draft = str(payload.get("draft", ""))
        raw_context = payload.get("context", [])
        context = [str(m).strip() for m in raw_context if str(m).strip()] if isinstance(raw_context, list) else []
        spelling = bool(payload.get("spelling", True))
        try:
            cache_weight = float(payload.get("cacheWeight", 0.5))
        except (TypeError, ValueError):
            cache_weight = 0.5

        raw_cache = payload.get("cache", {})
        cache: dict[str, int] = {}
        if isinstance(raw_cache, dict):
            for k, v in raw_cache.items():
                try:
                    count = int(v)
                except (TypeError, ValueError):
                    continue
                if count > 0:
                    cache[str(k).lower()] = count

        raw_confirmed = payload.get("confirmed", [])
        confirmed = {str(w).lower() for w in raw_confirmed} if isinstance(raw_confirmed, list) else set()

        def is_usable_cache_word(word: str) -> bool:
            return word in confirmed or not self.is_detectable_typo(word)

        previous = " ".join(context)
        text = f"{previous} {draft if draft.strip() else ''}" if previous else draft
        is_next = text.endswith(" ")
        split = text.rfind(" ")
        prefix = text.rstrip() if is_next else (text[:split] if split >= 0 else "")
        partial = "" if is_next else text[split + 1:]
        partial_lower = partial.lower()
        mode = "Starter" if not text else ("Next-Word" if is_next else "Prefix-Completion")

        if not text:
            suggestions = [
                {"word": "I", "probability": 0.32},
                {"word": "Hi", "probability": 0.25},
                {"word": "Hey", "probability": 0.18},
            ]
        else:
            if is_next:
                logits = self.next_logits(prefix)
                scores = logits[self.word_ids].clone()
                if cache and cache_weight:
                    for word, count in cache.items():
                        if is_usable_cache_word(word):
                            indices = self.lower_to_indices.get(word)
                            if indices:
                                boost = cache_weight * math.log1p(count)
                                for idx in indices:
                                    scores[idx] += boost
                suggestions = self._top_suggestions(self.words, scores, limit=3)
            else:
                match_indices = [
                    idx for idx, lower in enumerate(self.words_lower)
                    if lower.startswith(partial_lower) and len(self.words[idx]) > len(partial)
                ]
                if match_indices:
                    logits = self.next_logits(prefix)
                    sub_ids = self.word_ids[match_indices]
                    sub_words = [self.words[idx] for idx in match_indices]
                    scores = logits[sub_ids].clone()
                    if cache and cache_weight:
                        for pos, idx in enumerate(match_indices):
                            lower = self.words_lower[idx]
                            count = cache.get(lower, 0)
                            if count > 0 and is_usable_cache_word(lower):
                                scores[pos] += cache_weight * math.log1p(count)
                    suggestions = self._top_suggestions(sub_words, scores, limit=3)
                else:
                    fallback = [
                        (word, count) for word, count in cache.items()
                        if is_usable_cache_word(word)
                        and word.startswith(partial_lower)
                        and len(word) > len(partial_lower)
                    ]
                    fallback.sort(key=lambda item: (-item[1], item[0]))
                    suggestions = [{"word": word, "probability": 0.9} for word, _ in fallback[:3]]

            if partial and partial[0].isupper():
                for item in suggestions:
                    item["word"] = capitalize_word(item["word"])

        punctuation = None
        trimmed = text.rstrip()
        if draft.strip() and trimmed and trimmed[-1].isalnum() and (is_next or " " in trimmed):
            punct_logits = self.next_logits(trimmed)
            punct_words = [sym for sym, _ in self.punctuation]
            punct_scores = torch.tensor([float(punct_logits[tid]) for _, tid in self.punctuation])
            ranked_punct = self._top_suggestions(punct_words, punct_scores, limit=1)
            punctuation = ranked_punct[0] if ranked_punct else None

        corrections: list[dict[str, Any]] = []
        if spelling and draft:
            for match in COMPLETED_WORD_RE.finditer(draft):
                word = match.group(1)
                if word.lower() in confirmed:
                    continue
                left = draft[: match.start(1)]
                context_prefix = f"{previous} {left}" if previous else left
                candidates =
