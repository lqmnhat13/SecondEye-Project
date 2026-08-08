"""Pure detection-only risk helpers.

This module intentionally does not claim that an object is near. Near/medium/far
requires the depth module and is evaluated separately for RQ1.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Collection


class Direction(str, Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


@dataclass(frozen=True)
class ObstacleCandidate:
    is_candidate: bool
    direction: Direction
    reason: str


def direction_from_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    image_width: float,
    central_zone_fraction: float = 0.40,
) -> Direction:
    """Map the bounding-box center to left/center/right image regions."""
    if image_width <= 0:
        raise ValueError("image_width must be positive")
    if not 0.0 < central_zone_fraction < 1.0:
        raise ValueError("central_zone_fraction must be between 0 and 1")

    x1, _, x2, _ = bbox_xyxy
    if x1 > x2:
        raise ValueError("bbox x1 must not exceed x2")

    center_x = (x1 + x2) / 2.0
    margin_fraction = (1.0 - central_zone_fraction) / 2.0
    left_boundary = image_width * margin_fraction
    right_boundary = image_width * (1.0 - margin_fraction)

    if center_x < left_boundary:
        return Direction.LEFT
    if center_x > right_boundary:
        return Direction.RIGHT
    return Direction.CENTER


def assess_detection_only(
    *,
    label: str,
    confidence: float,
    bbox_xyxy: tuple[float, float, float, float],
    image_width: float,
    candidate_classes: Collection[str],
    confidence_threshold: float,
    central_zone_fraction: float = 0.40,
) -> ObstacleCandidate:
    """Flag an object as a candidate, never as a confirmed near obstacle."""
    direction = direction_from_bbox(bbox_xyxy, image_width, central_zone_fraction)
    if confidence < confidence_threshold:
        return ObstacleCandidate(False, direction, "confidence_below_threshold")
    if label not in candidate_classes:
        return ObstacleCandidate(False, direction, "class_not_in_candidate_set")
    if direction is not Direction.CENTER:
        return ObstacleCandidate(False, direction, "outside_central_travel_zone")
    return ObstacleCandidate(True, direction, "central_detection_requires_depth")

