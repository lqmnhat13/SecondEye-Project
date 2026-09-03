"""Optional Grounding DINO semantics for objects outside the COCO label set."""

from __future__ import annotations

import threading
import time
from typing import Any, Iterable

from secondeye.multimodal._model_loading import _from_pretrained_offline_first


DEFAULT_INDOOR_LABELS = (
    "door",
    "stairs",
    "column",
    "cabinet",
    "box",
    "trash can",
    "glass door",
    "curb",
)


class GroundingDinoDetector:
    """On-demand open-vocabulary detector for semantic descriptions only."""

    def __init__(
        self,
        *,
        model_name: str = "IDEA-Research/grounding-dino-tiny",
        revision: str | None = "a2bb814dd30d776dcf7e30523b00659f4f141c71",
        labels: Iterable[str] = DEFAULT_INDOOR_LABELS,
        threshold: float = 0.45,
        device: str = "cpu",
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("open-vocabulary threshold phải nằm trong (0, 1]")
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                'Thiếu open-vocabulary runtime. Chạy: python -m pip install ".[multimodal]"'
            ) from exc
        self.model_name = model_name
        self.requested_revision = revision
        self.labels = tuple(
            str(label).strip() for label in labels if str(label).strip()
        )
        if not self.labels:
            raise ValueError("open-vocabulary labels không được rỗng")
        self.threshold = threshold
        self.device = device
        load_kwargs = {} if revision is None else {"revision": revision}
        self.processor = _from_pretrained_offline_first(
            AutoProcessor, model_name, **load_kwargs
        )
        model = _from_pretrained_offline_first(
            AutoModelForZeroShotObjectDetection, model_name, **load_kwargs
        )
        self.model = model.to(device)
        self.model.eval()
        self.model_revision = getattr(
            getattr(self.model, "config", None), "_commit_hash", None
        ) or revision
        self._lock = threading.Lock()

    def predict_bgr(self, image: Any) -> dict[str, object]:
        try:
            import cv2
            import numpy as np
            import torch
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - optional dependencies
            raise RuntimeError("Thiếu dependency cho Grounding DINO") from exc
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("open-vocabulary input phải là ảnh OpenCV không rỗng")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        prompt = ". ".join(self.labels) + "."
        inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        started = time.perf_counter()
        with self._lock, torch.inference_mode():
            outputs = self.model(**inputs)
        processed = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.threshold,
            text_threshold=self.threshold,
            target_sizes=[image.shape[:2]],
        )[0]
        width = image.shape[1]
        detections: list[dict[str, object]] = []
        labels = processed.get("text_labels", processed.get("labels", []))
        for score, label, box in zip(
            processed["scores"], labels, processed["boxes"], strict=False
        ):
            confidence = float(score.detach().cpu().item())
            bbox = [float(value) for value in box.detach().cpu().tolist()]
            center_x = (bbox[0] + bbox[2]) / 2.0
            direction = (
                "left"
                if center_x < width * 0.30
                else "right" if center_x > width * 0.70 else "center"
            )
            detections.append(
                {
                    "label": str(label).strip().replace(" ", "_"),
                    "source_label": str(label).strip(),
                    "confidence": round(confidence, 4),
                    "bbox_xyxy": [round(value, 2) for value in bbox],
                    "direction": direction,
                    "obstacle_candidate": False,
                    "safety_evaluable": False,
                    "candidate_reason": "semantic_only_requires_metric_geometry",
                }
            )
        return {
            "schema_version": "1.0",
            "module": "open_vocabulary_detection",
            "success": True,
            "model": self.model_name,
            "model_revision": self.model_revision,
            "device": self.device,
            "detections": detections,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "limitations": [
                "Open-vocabulary chỉ bổ sung nhãn cho mô tả; không phát cảnh báo an toàn.",
                "Nhãn phải được xác nhận bằng geometry metric nếu dùng cho vật cản.",
            ],
        }
