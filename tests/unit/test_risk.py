import pytest

from secondeye.detection.risk import (
    Direction,
    assess_detection_only,
    direction_from_bbox,
)


@pytest.mark.parametrize(
    ("bbox", "expected"),
    [
        ((0, 0, 100, 100), Direction.LEFT),
        ((300, 0, 500, 100), Direction.CENTER),
        ((700, 0, 900, 100), Direction.RIGHT),
    ],
)
def test_direction_from_bbox_uses_box_center(bbox, expected):
    assert (
        direction_from_bbox(bbox, image_width=1000, central_zone_fraction=0.4)
        is expected
    )


def test_direction_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        direction_from_bbox((0, 0, 1, 1), image_width=0)
    with pytest.raises(ValueError):
        direction_from_bbox((0, 0, 1, 1), image_width=1, central_zone_fraction=1.0)
    with pytest.raises(ValueError):
        direction_from_bbox((2, 0, 1, 1), image_width=10)


def test_detection_only_marks_central_candidate_without_claiming_depth():
    result = assess_detection_only(
        label="chair",
        confidence=0.9,
        bbox_xyxy=(400, 0, 600, 500),
        image_width=1000,
        candidate_classes={"chair"},
        confidence_threshold=0.25,
    )
    assert result.is_candidate is True
    assert result.reason == "central_detection_requires_depth"


@pytest.mark.parametrize(
    ("label", "confidence", "bbox", "reason"),
    [
        ("chair", 0.2, (400, 0, 600, 500), "confidence_below_threshold"),
        ("book", 0.9, (400, 0, 600, 500), "class_not_in_candidate_set"),
        ("chair", 0.9, (0, 0, 100, 500), "outside_central_travel_zone"),
    ],
)
def test_detection_only_rejects_non_candidates(label, confidence, bbox, reason):
    result = assess_detection_only(
        label=label,
        confidence=confidence,
        bbox_xyxy=bbox,
        image_width=1000,
        candidate_classes={"chair"},
        confidence_threshold=0.25,
    )
    assert result.is_candidate is False
    assert result.reason == reason
