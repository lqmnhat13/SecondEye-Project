"""Thread-safe local inference wrapper for SecondEye object detection."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import DetectionPipelineConfig
from .risk import assess_detection_only
from .runtime import (
    ensure_class_schema,
    require_detection_runtime,
    select_device,
    synchronize_device,
)


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    bbox_xyxy: list[float]
    direction: str
    obstacle_candidate: bool
    candidate_reason: str
    depth_zone: None = None


class ObjectObstacleDetector:
    """Load one PT/ONNX model and return a stable multimodal-friendly schema.

    Inputs are OpenCV images (BGR uint8). Detection only produces obstacle
    candidates; a depth module must confirm distance/risk.
    """

    def __init__(self, weights: Path, config: DetectionPipelineConfig) -> None:
        cv2, torch, yolo_class = require_detection_runtime()
        del cv2
        weights = weights.expanduser().resolve()
        if not weights.is_file():
            raise FileNotFoundError(weights)
        requested_device = config.model.device
        self.device = "cpu" if weights.suffix.lower() == ".onnx" else select_device(
            requested_device, torch
        )
        self._torch = torch
        self.model = yolo_class(str(weights), task="detect")
        ensure_class_schema(self.model.names, config.class_names)
        self.config = config
        self.weights = weights
        self._lock = threading.Lock()

    @staticmethod
    def _validate_bgr(image: Any) -> None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Thiếu NumPy") from exc
        if not isinstance(image, np.ndarray):
            raise TypeError("image phải là numpy.ndarray từ OpenCV")
        if image.size == 0 or image.ndim not in (2, 3):
            raise ValueError("image phải là ảnh HxW hoặc HxWxC không rỗng")
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError("chiều cao/rộng ảnh phải dương")
        if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
            raise ValueError("ảnh chỉ được có 1, 3 hoặc 4 kênh")
        if image.dtype != np.uint8:
            raise ValueError("ảnh OpenCV phải có dtype uint8")

    def warmup(self) -> None:
        import numpy as np

        dummy = np.zeros(
            (self.config.model.image_size, self.config.model.image_size, 3), dtype=np.uint8
        )
        self._predict_result(dummy)

    def _predict_result(self, image: Any) -> tuple[Any, float]:
        self._validate_bgr(image)
        synchronize_device(self.device, self._torch)
        started = time.perf_counter()
        with self._lock:
            results = self.model.predict(
                source=image,
                conf=self.config.model.confidence_threshold,
                iou=self.config.model.iou_threshold,
                imgsz=self.config.model.image_size,
                device=self.device,
                verbose=False,
            )
        synchronize_device(self.device, self._torch)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if len(results) != 1:
            raise RuntimeError(f"Cần đúng một inference result, nhận {len(results)}")
        return results[0], latency_ms

    def predict_bgr(self, image: Any) -> dict[str, object]:
        result, latency_ms = self._predict_result(image)
        image_height, image_width = map(int, result.orig_shape)
        del image_height
        detections: list[Detection] = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                label = str(result.names[class_id])
                confidence = float(box.conf.item())
                xyxy = tuple(float(value) for value in box.xyxy[0].cpu().tolist())
                assessment = assess_detection_only(
                    label=label,
                    confidence=confidence,
                    bbox_xyxy=xyxy,
                    image_width=float(image_width),
                    candidate_classes=self.config.candidate_classes,
                    confidence_threshold=self.config.model.confidence_threshold,
                    central_zone_fraction=self.config.central_zone_fraction,
                )
                detections.append(
                    Detection(
                        class_id=class_id,
                        label=label,
                        confidence=round(confidence, 4),
                        bbox_xyxy=[round(value, 2) for value in xyxy],
                        direction=assessment.direction.value,
                        obstacle_candidate=assessment.is_candidate,
                        candidate_reason=assessment.reason,
                    )
                )
        detections.sort(key=lambda item: item.confidence, reverse=True)
        return {
            "schema_version": "2.0",
            "module": "object_obstacle_detection",
            "success": True,
            "model": self.weights.name,
            "device": self.device,
            "image_size": {"height": int(result.orig_shape[0]), "width": image_width},
            "detections": [asdict(item) for item in detections],
            "latency_ms": round(latency_ms, 2),
            "limitations": [
                "Detection 2D chỉ tạo ứng viên vật cản; cần depth để xác nhận khoảng cách."
            ],
        }

    def predict_and_render_bgr(self, image: Any) -> tuple[dict[str, object], Any]:
        """Predict once and return JSON plus an OpenCV BGR visualization."""
        result, latency_ms = self._predict_result(image)
        image_height, image_width = map(int, result.orig_shape)
        detections: list[Detection] = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                label = str(result.names[class_id])
                confidence = float(box.conf.item())
                xyxy = tuple(float(value) for value in box.xyxy[0].cpu().tolist())
                assessment = assess_detection_only(
                    label=label,
                    confidence=confidence,
                    bbox_xyxy=xyxy,
                    image_width=float(image_width),
                    candidate_classes=self.config.candidate_classes,
                    confidence_threshold=self.config.model.confidence_threshold,
                    central_zone_fraction=self.config.central_zone_fraction,
                )
                detections.append(
                    Detection(
                        class_id=class_id,
                        label=label,
                        confidence=round(confidence, 4),
                        bbox_xyxy=[round(value, 2) for value in xyxy],
                        direction=assessment.direction.value,
                        obstacle_candidate=assessment.is_candidate,
                        candidate_reason=assessment.reason,
                    )
                )
        detections.sort(key=lambda item: item.confidence, reverse=True)
        payload = {
            "schema_version": "2.0",
            "module": "object_obstacle_detection",
            "success": True,
            "model": self.weights.name,
            "device": self.device,
            "image_size": {"height": image_height, "width": image_width},
            "detections": [asdict(item) for item in detections],
            "latency_ms": round(latency_ms, 2),
            "limitations": [
                "Detection 2D chỉ tạo ứng viên vật cản; cần depth để xác nhận khoảng cách."
            ],
        }
        return payload, result.plot()
