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
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


@dataclass(frozen=True, slots=True)
class OcrConsensusConfig:
    """Settings for selecting a stable transcript from a short frame burst."""

    max_candidates: int = 3
    minimum_consensus: float = 0.60

    def __post_init__(self) -> None:
        if self.max_candidates <= 0:
            raise ValueError("max_candidates phải dương")
        if not 0.0 <= self.minimum_consensus <= 1.0:
            raise ValueError("minimum_consensus phải nằm trong [0, 1]")


def _box_bounds(box: Any) -> tuple[float, float, float, float] | None:
    """Normalize an xyxy box or polygon to xyxy float bounds."""
    if box is None:
        return None
    value = getattr(box, "tolist", lambda: box)()
    if not isinstance(value, (list, tuple)) or not value:
        return None
    try:
        if len(value) == 4 and not isinstance(value[0], (list, tuple)):
            x1, y1, x2, y2 = (float(item) for item in value)
            return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
        points = [point for point in value if len(point) >= 2]
        if not points:
            return None
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        return min(xs), min(ys), max(xs), max(ys)
    except (TypeError, ValueError):
        return None


def _sort_lines_reading_order(
    lines: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Sort horizontal text into visual rows, then from left to right."""
    positioned: list[tuple[dict[str, object], tuple[float, float, float, float]]] = []
    unpositioned: list[dict[str, object]] = []
    for line in lines:
        bounds = _box_bounds(line.get("box"))
        if bounds is None:
            unpositioned.append(line)
        else:
            positioned.append((line, bounds))
    if not positioned:
        return list(lines)
    typical_height = median(max(1.0, box[3] - box[1]) for _, box in positioned)
    rows: list[dict[str, object]] = []
    for line, bounds in sorted(
        positioned, key=lambda item: ((item[1][1] + item[1][3]) / 2.0, item[1][0])
    ):
        center_y = (bounds[1] + bounds[3]) / 2.0
        matching = next(
            (
                row
                for row in reversed(rows)
                if abs(center_y - float(row["center_y"])) <= typical_height * 0.60
            ),
            None,
        )
        if matching is None:
            rows.append({"center_y": center_y, "items": [(line, bounds)]})
        else:
            items = matching["items"]
            assert isinstance(items, list)
            items.append((line, bounds))
            matching["center_y"] = sum(
                (item_box[1] + item_box[3]) / 2.0 for _, item_box in items
            ) / len(items)
    ordered: list[dict[str, object]] = []
    for row in rows:
        items = row["items"]
        assert isinstance(items, list)
        ordered.extend(line for line, _ in sorted(items, key=lambda item: item[1][0]))
    return ordered + unpositioned


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
        raw_lines = _sort_lines_reading_order(raw_lines)
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
            "languages": ["vi-VN", "en-US"],
            "transcript": " ".join(line["text"] for line in lines),
            "structured_transcript": "\n".join(line["text"] for line in lines),
            "raw_transcript": " ".join(line["text"] for line in raw_lines),
            "lines": lines,
            "discarded_lines": discarded,
            "minimum_line_confidence": self.minimum_line_confidence,
            "reading_order": "row_major_geometry_v1",
            "engine_latency_ms": round(float(payload.get("latency_ms", 0.0)), 2),
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "limitations": [
                "Confidence của engine chưa được calibration thành xác suất transcript đúng."
            ],
        }


class PaddleOcrReader:
    def __init__(
        self,
        language: str = "vi",
        *,
        minimum_line_confidence: float = 0.75,
        accept_missing_confidence: bool = False,
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
        self.accept_missing_confidence = accept_missing_confidence
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
                normalized_box = (
                    None if box is None else getattr(box, "tolist", lambda: box)()
                )
                lines.append(
                    {
                        "text": str(text).strip(),
                        "confidence": None if score is None else round(score, 4),
                        "box": normalized_box,
                    }
                )
        lines = _sort_lines_reading_order(lines)
        accepted_lines = [
            line
            for line in lines
            if line["text"]
            and (
                (
                    line["confidence"] is None
                    and getattr(self, "accept_missing_confidence", False)
                )
                or (
                    line["confidence"] is not None
                    and float(line["confidence"]) >= self.minimum_line_confidence
                )
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
            "structured_transcript": "\n".join(
                line["text"] for line in accepted_lines
            ),
            "raw_transcript": " ".join(
                line["text"] for line in lines if line["text"]
            ),
            "lines": accepted_lines,
            "discarded_lines": discarded_lines,
            "minimum_line_confidence": self.minimum_line_confidence,
            "accept_missing_confidence": getattr(
                self, "accept_missing_confidence", False
            ),
            "reading_order": "row_major_geometry_v1",
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "limitations": [
                "Confidence của engine chưa được calibration thành xác suất transcript đúng."
            ],
        }


class AutomaticOcrReader:
    """Prefer Apple Vision on macOS and lazily fall back to PaddleOCR."""

    def __init__(self, language: str = "vi") -> None:
        self.language = language
        self._primary_init_error: str | None = None
        try:
            self._primary: AppleVisionOcrReader | None = AppleVisionOcrReader()
        except RuntimeError as exc:
            self._primary = None
            self._primary_init_error = f"{type(exc).__name__}: {exc}"
        self._fallback: PaddleOcrReader | None = None

    def read_bgr(self, image: Any) -> dict[str, object]:
        primary_error: str | None = None
        fallback_reason = "primary_unavailable"
        if self._primary is not None:
            try:
                primary_result = self._primary.read_bgr(image)
            except Exception as exc:
                primary_error = f"{type(exc).__name__}: {exc}"
                fallback_reason = "primary_error"
                self._primary = None
            else:
                if str(primary_result.get("transcript", "")).strip():
                    return primary_result
                primary_error = "Apple Vision returned no accepted text"
                fallback_reason = "primary_empty"
        primary_error = primary_error or self._primary_init_error
        try:
            self._fallback = self._fallback or PaddleOcrReader(language=self.language)
            result = self._fallback.read_bgr(image)
        except Exception as exc:
            fallback_error = f"{type(exc).__name__}: {exc}"
            if primary_error:
                raise RuntimeError(
                    f"Cả hai OCR engine đều lỗi; primary={primary_error}; "
                    f"fallback={fallback_error}"
                ) from exc
            raise RuntimeError(f"PaddleOCR lỗi: {fallback_error}") from exc
        return {
            **result,
            "fallback_from": "Apple Vision" if primary_error else None,
            "fallback_error": primary_error,
            "fallback_reason": fallback_reason,
        }
