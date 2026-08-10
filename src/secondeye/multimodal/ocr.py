"""PaddleOCR 3.x adapter with a stable SecondEye result schema."""

from __future__ import annotations

import time
from typing import Any


class PaddleOcrReader:
    def __init__(self, language: str = "vi") -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                'Thiếu OCR runtime. Chạy: python -m pip install ".[ocr]"'
            ) from exc
        self.language = language
        self.engine = PaddleOCR(lang=language, use_doc_orientation_classify=False)

    @staticmethod
    def _result_payload(result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            return result.get("res", result)
        value = getattr(result, "json", None)
        if callable(value):
            value = value()
        if isinstance(value, dict):
            return value.get("res", value)
        return {}

    def read_bgr(self, image: Any) -> dict[str, object]:
        started = time.perf_counter()
        results = self.engine.predict(image)
        lines: list[dict[str, object]] = []
        for result in results:
            payload = self._result_payload(result)
            texts = payload.get("rec_texts", ())
            scores = payload.get("rec_scores", ())
            boxes = payload.get("rec_boxes", payload.get("dt_polys", ()))
            for index, text in enumerate(texts):
                score = float(scores[index]) if index < len(scores) else None
                box = boxes[index] if index < len(boxes) else None
                lines.append(
                    {
                        "text": str(text).strip(),
                        "confidence": None if score is None else round(score, 4),
                        "box": None if box is None else getattr(box, "tolist", lambda: box)(),
                    }
                )
        transcript = " ".join(line["text"] for line in lines if line["text"])
        return {
            "schema_version": "1.0",
            "module": "ocr",
            "success": True,
            "engine": "PaddleOCR",
            "language": self.language,
            "transcript": transcript,
            "lines": lines,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
