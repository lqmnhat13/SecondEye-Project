"""Detection training, inference, and obstacle-candidate assessment."""

from .config import DetectionPipelineConfig, load_detection_config
from .model import Detection, ObjectObstacleDetector
from .risk import (
    Direction,
    ObstacleCandidate,
    assess_detection_only,
    direction_from_bbox,
)

__all__ = [
    "Detection",
    "DetectionPipelineConfig",
    "Direction",
    "ObjectObstacleDetector",
    "ObstacleCandidate",
    "assess_detection_only",
    "direction_from_bbox",
    "load_detection_config",
]
