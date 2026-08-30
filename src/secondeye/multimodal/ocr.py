"""Native macOS and PaddleOCR adapters with one stable result schema."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


class AppleVisionOcrReader:
    """Accurate Vietnamese OCR through the native macOS Vision framework."""

    def __init__(
        self,
        executable: str | Path | None = None,
        *,
        minimum_line_confidence: float = 0.50,
        timeout_seconds: float = 12.0,
    ) -> None:
        if platform.system() != "Darwin":
            raise RuntimeError("Apple Vision OCR chỉ hỗ trợ macOS")
        if not 0.0 <= minimum_line_confidence <= 1.0:
            raise ValueError("minimum_line_confidence phải nằm trong [0, 1]")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds phải dương")
        configured = executable or os.environ.get("SECONDEYE_VISION_OCR")
        candidates = [
            Path(configured).expanduser() if configured else None,
            Path(sys.executable).parent / "secondeye-vision-ocr",
            Path(value) if (value := shutil.which("secondeye-vision-ocr")) else None,
        ]
        resolved = next(
            (
                candidate
                for candidate in candidates
                if candidate and candidate.is_file() and os.access(candidate, os.X_OK)
            ),
            None,
        )
        if resolved is None:
            raise RuntimeError(
                "Chưa có Apple Vision OCR helper. Chạy lại ./setup_mvp.sh."
            )
        self.executable = resolved
        self.language = "vi-VN"
        self.minimum_line_confidence = minimum_line_confidence
        self.timeout_seconds = timeout_seconds

    def read_bgr(self, image: Any) -> dict[str, object]:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("Thiếu OpenCV/NumPy cho Apple Vision OCR") from exc
        if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim != 3:
            raise ValueError("Apple Vision OCR cần ảnh OpenCV BGR không rỗng")
        height, width = map(int, image.shape[:2])
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="secondeye_vision_ocr_") as directory:
            image_path = Path(directory) / "frame.png"
            if not cv2.imwrite(str(image_path), image):
                raise RuntimeError("Không tạo được ảnh tạm cho Apple Vision OCR")
            try:
                completed = subprocess.run(
                    [str(self.executable), str(image_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("Apple Vision OCR hết thời gian xử lý") from exc
            except OSError as exc:
                raise RuntimeError("Không chạy được Apple Vision OCR helper") from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or "không rõ lỗi"
            raise RuntimeError(f"Apple Vision OCR lỗi: {detail}")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Apple Vision OCR trả JSON không hợp lệ") from exc
        raw_lines: list[dict[str, object]] = []
        for item in payload.get("lines", []):
            normalized_box = [float(value) for value in item.get("box", [])]
            if len(normalized_box) != 4:
                continue
            x1, min_y, x2, max_y = normalized_box
            raw_lines.append(
                {
                    "text": str(item.get("text", "")).strip(),
                    "confidence": round(float(item.get("confidence", 0.0)), 4),
                    "box": [
                        round(x1 * width, 2),
                        round((1.0 - max_y) * height, 2),
                        round(x2 * width, 2),
                        round((1.0 - min_y) * height, 2),
                    ],
                }
            )
        raw_lines.sort(key=lambda line: (line["box"][1], line["box"][0]))
        lines = [
            line
            for line in raw_lines
            if line["text"]
            and float(line["confidence"]) >= self.minimum_line_confidence
        ]
        discarded = [line for line in raw_lines if line not in lines]
        return {
            "schema_version": "1.0",
            "module": "ocr",
            "success": True,
            "engine": "Apple Vision",
            "language": self.language,
            "transcript": " ".join(line["text"] for line in lines),
            "raw_transcript": " ".join(line["text"] for line in raw_lines),
            "lines": lines,
            "discarded_lines": discarded,
            "minimum_line_confidence": self.minimum_line_confidence,
            "engine_latency_ms": round(float(payload.get("latency_ms", 0.0)), 2),
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }


class PaddleOcrReader:
    def __init__(
        self,
        language: str = "vi",
        *,
        minimum_line_confidence: float = 0.75,
    ) -> None:
        if not 0.0 <= minimum_line_confidence <= 1.0:
            raise ValueError("minimum_line_confidence phải nằm trong [0, 1]")
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                'Thiếu OCR runtime. Chạy: python -m pip install ".[ocr]"'
            ) from exc
        self.language = language
        self.minimum_line_confidence = minimum_line_confidence
        self.engine = PaddleOCR(
            lang=language,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

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
                        "box": None
                        if box is None
                        else getattr(box, "tolist", lambda: box)(),
                    }
                )
        accepted_lines = [
            line
            for line in lines
            if line["text"]
            and (
                line["confidence"] is None
                or float(line["confidence"]) >= self.minimum_line_confidence
            )
        ]
        discarded_lines = [line for line in lines if line not in accepted_lines]
        transcript = " ".join(line["text"] for line in accepted_lines)
        return {
            "schema_version": "1.0",
            "module": "ocr",
            "success": True,
            "engine": "PaddleOCR",
            "language": self.language,
            "transcript": transcript,
            "raw_transcript": " ".join(
                line["text"] for line in lines if line["text"]
            ),
            "lines": accepted_lines,
            "discarded_lines": discarded_lines,
            "minimum_line_confidence": self.minimum_line_confidence,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }


class AutomaticOcrReader:
    """Prefer Apple Vision on macOS and lazily fall back to PaddleOCR."""

    def __init__(self, language: str = "vi") -> None:
        self.language = language
        try:
            self._primary: AppleVisionOcrReader | None = AppleVisionOcrReader()
        except RuntimeError:
            self._primary = None
        self._fallback: PaddleOcrReader | None = None

    def read_bgr(self, image: Any) -> dict[str, object]:
        primary_error: str | None = None
        if self._primary is not None:
            try:
                return self._primary.read_bgr(image)
            except RuntimeError as exc:
                primary_error = f"{type(exc).__name__}: {exc}"
                self._primary = None
        self._fallback = self._fallback or PaddleOcrReader(language=self.language)
        result = self._fallback.read_bgr(image)
        return {
            **result,
            "fallback_from": "Apple Vision" if primary_error else None,
            "fallback_error": primary_error,
        }
