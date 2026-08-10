"""Relative monocular depth adapter for Depth Anything V2 Small."""

from __future__ import annotations

import time
from typing import Any


def relative_depth_band(value: float) -> str:
    """Map normalized inverse depth to a deliberately non-metric band."""
    if not 0.0 <= value <= 1.0:
        raise ValueError("relative depth phải nằm trong [0, 1]")
    if value >= 2.0 / 3.0:
        return "near"
    if value >= 1.0 / 3.0:
        return "medium"
    return "far"


class DepthAnythingEstimator:
    """Lazy local Depth Anything V2 Small inference.

    Output is relative inverse depth only. It must never be described as metres.
    """

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str = "auto",
    ) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise RuntimeError(
                'Thiếu depth runtime. Chạy: python -m pip install ".[multimodal]"'
            ) from exc
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model_name = model_name
        self._torch = torch
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)
        self.model.eval()

    def predict_bgr(self, image: Any) -> dict[str, object]:
        try:
            import cv2
            import numpy as np
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - installed by extras
            raise RuntimeError("Thiếu OpenCV/NumPy/Pillow cho depth") from exc
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("depth input phải là ảnh OpenCV không rỗng")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        started = time.perf_counter()
        with self._torch.inference_mode():
            output = self.model(**inputs).predicted_depth
            resized = self._torch.nn.functional.interpolate(
                output.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth = resized.detach().float().cpu().numpy()
        low, high = np.percentile(depth, (2.0, 98.0))
        if high <= low:
            normalized = np.zeros_like(depth, dtype=np.float32)
        else:
            normalized = np.clip((depth - low) / (high - low), 0.0, 1.0).astype(
                np.float32
            )
        return {
            "schema_version": "1.0",
            "module": "relative_depth",
            "success": True,
            "model": self.model_name,
            "device": self.device,
            "relative_inverse_depth": normalized,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "semantics": "larger_is_closer_relative_only",
            "limitations": [
                "near/medium/far là band tương đối theo từng frame, không phải khoảng cách mét."
            ],
        }


def attach_depth_zones(
    detections: list[dict[str, Any]], relative_inverse_depth: Any
) -> list[dict[str, Any]]:
    """Attach a robust median depth band to every detection bbox."""
    import numpy as np

    if not isinstance(relative_inverse_depth, np.ndarray) or relative_inverse_depth.ndim != 2:
        raise ValueError("relative_inverse_depth phải là ma trận HxW")
    height, width = relative_inverse_depth.shape
    enriched: list[dict[str, Any]] = []
    for original in detections:
        item = dict(original)
        x1, y1, x2, y2 = (int(round(float(v))) for v in item["bbox_xyxy"])
        x1, x2 = max(0, min(x1, width - 1)), max(1, min(x2, width))
        y1, y2 = max(0, min(y1, height - 1)), max(1, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            item["depth_zone"] = "unknown"
            item["relative_depth"] = None
        else:
            value = float(np.median(relative_inverse_depth[y1:y2, x1:x2]))
            item["relative_depth"] = round(value, 4)
            item["depth_zone"] = relative_depth_band(value)
        enriched.append(item)
    return enriched
