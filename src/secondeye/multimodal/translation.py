"""Lazy pretrained English-to-Vietnamese translation for VQA output."""

from __future__ import annotations

import threading
import time
from collections import Counter
from typing import Any

from secondeye.accelerator import accelerator_guard


class PretrainedEnglishVietnameseTranslator:
    """Translate short model answers locally with a pretrained Marian model."""

    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-en-vi",
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.requested_device = device
        self.device: str | None = None
        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._lock = threading.Lock()

    def _load(self) -> tuple[Any, Any, Any]:
        with self._lock:
            if self._model is not None:
                return self._torch, self._tokenizer, self._model
            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    'Thiếu runtime dịch. Chạy: python -m pip install ".[multimodal]"'
                ) from exc
            device = self.requested_device
            if device == "auto":
                if torch.cuda.is_available():
                    device = "cuda:0"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            with accelerator_guard(device, torch):
                model = model.to(device)
            model.eval()
            self.device = device
            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model
            return torch, tokenizer, model

    def translate(self, text: str) -> dict[str, object]:
        source = " ".join(text.strip().split())
        if not source:
            raise ValueError("Nội dung cần dịch không được rỗng")
        started = time.perf_counter()
        torch, tokenizer, model = self._load()
        model_input = f"The answer is {source.rstrip('.')}."
        inputs = tokenizer(
            model_input,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        with accelerator_guard(self.device, torch):
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_length=64,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.1,
                    renormalize_logits=True,
                )
            translation = tokenizer.decode(
                generated[0], skip_special_tokens=True
            ).strip()
        for prefix in ("Câu trả lời là ", "Câu trả lời: ", "Đáp án là ", "Đáp án: "):
            if translation.casefold().startswith(prefix.casefold()):
                translation = translation[len(prefix) :].strip()
                break
        if not translation:
            raise RuntimeError("Model dịch không trả kết quả")
        words = translation.casefold().split()
        most_common = Counter(words).most_common(1)[0][1]
        if len(words) >= 8 and most_common > max(4, len(words) // 3):
            raise RuntimeError("Model dịch trả kết quả bị lặp bất thường")
        return {
            "schema_version": "1.0",
            "module": "translation_en_vi",
            "success": True,
            "model": self.model_name,
            "device": self.device,
            "source": source,
            "translation": translation,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
