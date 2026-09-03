"""Contract for synchronized hardware depth providers (for example ARKit LiDAR)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class AlignedMetricDepthFrame:
    """Metric depth already registered to its RGB frame."""

    metric_depth_m: Any
    captured_at: float
    fx: float
    fy: float
    cx: float
    cy: float
    source: str = "hardware_depth"
    confidence: Any | None = None

    def as_result(self) -> dict[str, object]:
        import numpy as np

        depth = self.metric_depth_m
        if not isinstance(depth, np.ndarray) or depth.ndim != 2:
            raise ValueError("metric_depth_m từ sensor phải là ma trận HxW")
        valid = np.isfinite(depth) & (depth > 0.0)
        return {
            "schema_version": "1.0",
            "module": "metric_depth",
            "success": True,
            "model": self.source,
            "device": "sensor",
            "depth_type": "metric",
            "metric_depth_m": depth.astype(np.float32, copy=False),
            "usable": bool(valid.any() and float(valid.mean()) >= 0.05),
            "valid_fraction": round(float(valid.mean()), 4),
            "captured_at": self.captured_at,
            "intrinsics": {
                "fx": self.fx,
                "fy": self.fy,
                "cx": self.cx,
                "cy": self.cy,
            },
            "confidence_available": self.confidence is not None,
            "semantics": "larger_is_farther_metres",
        }


class SynchronizedDepthProvider(Protocol):
    """Adapter boundary for a camera that emits aligned RGB and metric depth."""

    def read(self) -> tuple[Any, AlignedMetricDepthFrame]: ...
