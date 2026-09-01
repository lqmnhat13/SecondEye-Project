"""Relative monocular depth adapter for Depth Anything V2 Small."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from secondeye.accelerator import accelerator_guard
from secondeye.multimodal._model_loading import _from_pretrained_offline_first


@dataclass(frozen=True, slots=True)
class DepthFusionConfig:
    """Safety-oriented settings for turning relative depth into coarse bands.

    Thresholds remain deliberately non-metric. They can be tuned on a locked
    validation set, but must not be presented as distances in metres.
    """

    medium_threshold: float = 1.0 / 3.0
    near_threshold: float = 2.0 / 3.0
    horizontal_inset: float = 0.20
    top_inset: float = 0.15
    bottom_inset: float = 0.10
    min_valid_pixels: int = 16
    max_iqr: float = 0.35
    medium_bbox_area_fraction: float = 0.025
    near_bbox_area_fraction: float = 0.12
    medium_bbox_height_fraction: float = 0.25
    near_bbox_height_fraction: float = 0.60

    def __post_init__(self) -> None:
        if not 0.0 < self.medium_threshold < self.near_threshold < 1.0:
            raise ValueError(
                "depth thresholds phải thỏa 0 < medium < near < 1"
            )
        for name, value in (
            ("horizontal_inset", self.horizontal_inset),
            ("top_inset", self.top_inset),
            ("bottom_inset", self.bottom_inset),
        ):
            if not 0.0 <= value < 0.5:
                raise ValueError(f"{name} phải nằm trong [0, 0.5)")
        if self.top_inset + self.bottom_inset >= 1.0:
            raise ValueError("tổng top_inset và bottom_inset phải nhỏ hơn 1")
        if self.min_valid_pixels <= 0:
            raise ValueError("min_valid_pixels phải dương")
        if not 0.0 < self.max_iqr <= 1.0:
            raise ValueError("max_iqr phải nằm trong (0, 1]")
        if not (
            0.0
            < self.medium_bbox_area_fraction
            < self.near_bbox_area_fraction
            < 1.0
        ):
            raise ValueError("ngưỡng bbox area phải thỏa 0 < medium < near < 1")
        if not (
            0.0
            < self.medium_bbox_height_fraction
            < self.near_bbox_height_fraction
            <= 1.0
        ):
            raise ValueError("ngưỡng bbox height phải thỏa 0 < medium < near <= 1")


DEFAULT_DEPTH_FUSION_CONFIG = DepthFusionConfig()


def _bbox_proximity_band(
    *,
    area_fraction: float,
    height_fraction: float,
    config: DepthFusionConfig,
) -> str:
    """Estimate coarse proximity from how much of the frame a bbox occupies."""
    if (
        area_fraction >= config.near_bbox_area_fraction
        or height_fraction >= config.near_bbox_height_fraction
    ):
        return "near"
    if (
        area_fraction >= config.medium_bbox_area_fraction
        or height_fraction >= config.medium_bbox_height_fraction
    ):
        return "medium"
    return "far"


def relative_depth_band(
    value: float, config: DepthFusionConfig = DEFAULT_DEPTH_FUSION_CONFIG
) -> str:
    """Map normalized inverse depth to a deliberately non-metric band."""
    if not 0.0 <= value <= 1.0:
        raise ValueError("relative depth phải nằm trong [0, 1]")
    if value >= config.near_threshold:
        return "near"
    if value >= config.medium_threshold:
        return "medium"
    return "far"


def attach_bbox_proximity_zones(
    detections: list[dict[str, Any]],
    *,
    frame_width: int,
    frame_height: int,
    config: DepthFusionConfig = DEFAULT_DEPTH_FUSION_CONFIG,
) -> list[dict[str, Any]]:
    """Attach a fast coarse proximity band using only bbox geometry."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("kích thước frame phải dương")
    enriched: list[dict[str, Any]] = []
    for original in detections:
        item = dict(original)
        x1, y1, x2, y2 = (float(value) for value in item["bbox_xyxy"])
        x1, x2 = max(0.0, min(x1, frame_width)), max(
            0.0, min(x2, frame_width)
        )
        y1, y2 = max(0.0, min(y1, frame_height)), max(
            0.0, min(y2, frame_height)
        )
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        area_fraction = (box_width * box_height) / (frame_width * frame_height)
        height_fraction = box_height / frame_height
        bbox_band = _bbox_proximity_band(
            area_fraction=area_fraction,
            height_fraction=height_fraction,
            config=config,
        )
        item["bbox_area_fraction"] = round(area_fraction, 4)
        item["bbox_height_fraction"] = round(height_fraction, 4)
        item["bbox_proximity_zone"] = bbox_band
        item["proximity_zone"] = bbox_band
        item["proximity_reason"] = "bbox_geometry_fast_path"
        enriched.append(item)
    return enriched


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
        self.processor = _from_pretrained_offline_first(
            AutoImageProcessor, model_name
        )
        model = _from_pretrained_offline_first(
            AutoModelForDepthEstimation, model_name
        )
        with accelerator_guard(device, torch):
            self.model = model.to(device)
        self.model.eval()

    def warmup(self) -> None:
        """Pay the one-time model/backend startup cost before live frames."""
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - installed by extras
            raise RuntimeError("Thiếu NumPy cho depth warmup") from exc
        self.predict_bgr(np.zeros((384, 384, 3), dtype=np.uint8))

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
        with accelerator_guard(self.device, self._torch):
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
        usable = bool(np.isfinite(low) and np.isfinite(high) and high > low)
        if not usable:
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
            "usable": usable,
            "normalization_percentiles": [2.0, 98.0],
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "semantics": "larger_is_closer_relative_only",
            "limitations": [
                "near/medium/far là band tương đối theo từng frame, không phải khoảng cách mét.",
                "depth_confidence là độ nhất quán không gian heuristic, không phải xác suất đã calibration.",
            ],
        }


def attach_depth_zones(
    detections: list[dict[str, Any]],
    relative_inverse_depth: Any,
    *,
    config: DepthFusionConfig = DEFAULT_DEPTH_FUSION_CONFIG,
) -> list[dict[str, Any]]:
    """Attach a confidence-gated depth band sampled from each bbox core.

    Insetting the box reduces background leakage. An object whose core contains
    strongly mixed depths is marked unknown instead of forcing a safety claim.
    """
    import numpy as np

    if (
        not isinstance(relative_inverse_depth, np.ndarray)
        or relative_inverse_depth.ndim != 2
    ):
        raise ValueError("relative_inverse_depth phải là ma trận HxW")
    height, width = relative_inverse_depth.shape
    enriched: list[dict[str, Any]] = []
    for original in detections:
        item = dict(original)
        x1, y1, x2, y2 = (int(round(float(v))) for v in item["bbox_xyxy"])
        x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
        y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
        box_width = x2 - x1
        box_height = y2 - y1
        area_fraction = (box_width * box_height) / max(1, width * height)
        height_fraction = box_height / max(1, height)
        bbox_band = _bbox_proximity_band(
            area_fraction=area_fraction,
            height_fraction=height_fraction,
            config=config,
        )
        item["bbox_area_fraction"] = round(area_fraction, 4)
        item["bbox_height_fraction"] = round(height_fraction, 4)
        item["bbox_proximity_zone"] = bbox_band
        core_x1 = x1 + int(round(box_width * config.horizontal_inset))
        core_x2 = x2 - int(round(box_width * config.horizontal_inset))
        core_y1 = y1 + int(round(box_height * config.top_inset))
        core_y2 = y2 - int(round(box_height * config.bottom_inset))
        if core_x2 <= core_x1 or core_y2 <= core_y1:
            core_x1, core_y1, core_x2, core_y2 = x1, y1, x2, y2
        item["depth_sample_xyxy"] = [core_x1, core_y1, core_x2, core_y2]
        if core_x2 <= core_x1 or core_y2 <= core_y1:
            item["depth_zone"] = "unknown"
            item["relative_depth"] = None
            item["depth_confidence"] = 0.0
            item["depth_reason"] = "invalid_bbox"
        else:
            sample = relative_inverse_depth[core_y1:core_y2, core_x1:core_x2]
            valid = sample[np.isfinite(sample)]
            required_pixels = min(config.min_valid_pixels, sample.size)
            if valid.size < required_pixels:
                item["depth_zone"] = "unknown"
                item["relative_depth"] = None
                item["depth_confidence"] = 0.0
                item["depth_reason"] = "insufficient_valid_depth"
                enriched.append(item)
                continue
            value = float(np.median(valid))
            q25, q75 = np.percentile(valid, (25.0, 75.0))
            iqr = float(q75 - q25)
            confidence = max(0.0, 1.0 - iqr)
            item["relative_depth"] = round(value, 4)
            item["depth_iqr"] = round(iqr, 4)
            item["depth_confidence"] = round(confidence, 4)
            relative_band = relative_depth_band(value, config)
            item["relative_depth_band"] = relative_band
            if iqr >= config.max_iqr:
                if bool(item.get("obstacle_candidate")) and bbox_band in {
                    "near",
                    "far",
                }:
                    # Relative depth often mixes foreground/background inside a
                    # detection box. For risk candidates, a clearly very large
                    # or very small bbox is a useful fallback signal.
                    item["depth_zone"] = bbox_band
                    item["depth_reason"] = (
                        "bbox_geometry_fallback_ambiguous_relative_depth"
                    )
                else:
                    item["depth_zone"] = "unknown"
                    item["depth_reason"] = "ambiguous_bbox_depth"
            else:
                if bool(item.get("obstacle_candidate")) and bbox_band in {
                    "near",
                    "far",
                }:
                    item["depth_zone"] = bbox_band
                    item["depth_reason"] = "bbox_geometry_with_relative_depth"
                else:
                    item["depth_zone"] = relative_band
                    item["depth_reason"] = "relative_bbox_core"
            item["proximity_zone"] = item["depth_zone"]
            item["proximity_reason"] = item["depth_reason"]
        enriched.append(item)
    return enriched
