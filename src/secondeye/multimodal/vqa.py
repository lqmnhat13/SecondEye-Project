"""Local pretrained visual question answering adapter."""

from __future__ import annotations

import time
from typing import Any

from secondeye.accelerator import accelerator_guard


class PretrainedVisualQuestionAnswering:
    def __init__(
        self,
        model_name: str = "Salesforce/blip-vqa-base",
        minimum_score: float = 0.20,
        device: str = "auto",
    ) -> None:
        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError("minimum_score phải nằm trong [0, 1]")
        try:
            import torch
            from transformers import BlipForQuestionAnswering, BlipProcessor
        except ImportError as exc:
            raise RuntimeError(
                'Thiếu VQA runtime. Chạy: python -m pip install ".[multimodal]"'
            ) from exc
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda:0"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.model_name = model_name
        self.minimum_score = minimum_score
        self.device = device
        self._torch = torch
        self.processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForQuestionAnswering.from_pretrained(model_name)
        with accelerator_guard(device, torch):
            self.model = model.to(device)
        self.model.eval()

    def ask_bgr(self, image: Any, question: str) -> dict[str, object]:
        try:
            import cv2
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Thiếu OpenCV/Pillow cho VQA") from exc
        question = question.strip()
        if not question:
            raise ValueError("question không được rỗng")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        started = time.perf_counter()
        inputs = self.processor(
            images=Image.fromarray(rgb), text=question, return_tensors="pt"
        )
        with accelerator_guard(self.device, self._torch):
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with self._torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            raw_answer = self.processor.decode(
                generated.sequences[0], skip_special_tokens=True
            ).strip()
            token_confidences = [
                float(self._torch.softmax(scores[0], dim=-1).max().item())
                for scores in generated.scores
            ]
        score = (
            sum(token_confidences) / len(token_confidences)
            if token_confidences
            else 0.0
        )
        abstained = not raw_answer or score < self.minimum_score
        answer = (
            "Tôi chưa đủ chắc chắn để trả lời từ hình ảnh hiện tại."
            if abstained
            else raw_answer
        )
        return {
            "schema_version": "1.0",
            "module": "vqa",
            "success": True,
            "model": self.model_name,
            "device": self.device,
            "question": question,
            "answer": answer,
            "raw_answer": raw_answer,
            "confidence": round(score, 4),
            "abstained": abstained,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "limitations": [
                "Confidence là token-generation score chưa calibration, không phải xác suất đúng.",
                "VQA pretrained có thể hallucinate; không dùng câu trả lời làm chỉ dẫn điều hướng.",
            ],
        }
