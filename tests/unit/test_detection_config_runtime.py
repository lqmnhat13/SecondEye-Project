from types import SimpleNamespace

import pytest

from secondeye.detection.config import load_detection_config
from secondeye.detection.pipeline import (
    _adapt_coco_result_to_second_eye,
    _pretrained_inference_floor,
    build_parser,
)
from secondeye.detection.runtime import ensure_class_schema


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _Box:
    def __init__(self, class_id: int, confidence: float):
        self.cls = _Scalar(class_id)
        self.conf = _Scalar(confidence)


class _Boxes:
    def __init__(self, boxes):
        self.values = list(boxes)

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, indices):
        if isinstance(indices, list):
            return _Boxes(self.values[index] for index in indices)
        return self.values[indices]


def test_default_yolo26_config_has_expected_pretrained_indoor_schema():
    config = load_detection_config()

    assert config.model.base_weights == "yolo26m.pt"
    assert config.model.device == "auto"
    assert len(config.class_names) == 15
    assert config.class_names == (
        "person",
        "chair",
        "table",
        "sofa",
        "bed",
        "backpack",
        "handbag",
        "suitcase",
        "bottle",
        "potted_plant",
        "tv",
        "laptop",
        "toilet",
        "sink",
        "refrigerator",
    )
    assert config.candidate_classes <= set(config.class_names)
    assert config.pretrained_coco.global_confidence_threshold == pytest.approx(0.35)
    assert dict(config.pretrained_coco.class_thresholds) == {
        "person": 0.29,
        "chair": 0.69,
        "table": 0.28,
        "sofa": 0.31,
        "bed": 0.48,
        "backpack": 0.17,
        "handbag": 0.35,
        "suitcase": 0.35,
        "bottle": 0.35,
        "potted_plant": 0.35,
        "tv": 0.35,
        "laptop": 0.35,
        "toilet": 0.35,
        "sink": 0.35,
        "refrigerator": 0.35,
    }
    assert dict(config.pretrained_coco.class_mapping)["dining table"] == "table"
    assert config.pretrained_coco.unsupported_second_eye_classes == ()


def test_camera_demo_uses_configured_yolo26_without_model_argument():
    args = build_parser().parse_args(["camera-demo", "--camera", "1"])

    assert args.command == "camera-demo"
    assert args.camera == 1
    assert not hasattr(args, "model")


def test_pretrained_coco_filter_maps_only_supported_classes():
    config = load_detection_config()
    result = SimpleNamespace(
        names={0: "person", 1: "chair", 2: "dining table", 3: "bus"},
        boxes=_Boxes(
            (
                _Box(0, 0.29),
                _Box(0, 0.28),
                _Box(1, 0.68),
                _Box(1, 0.70),
                _Box(2, 0.30),
                _Box(3, 0.99),
            )
        ),
    )

    filtered = _adapt_coco_result_to_second_eye(result, config)

    assert _pretrained_inference_floor(config) == pytest.approx(0.17)
    assert [(box.cls.item(), box.conf.item()) for box in filtered.boxes] == [
        (0, 0.29),
        (1, 0.70),
        (2, 0.30),
    ]
    assert filtered.names[2] == "table"


def test_schema_guard_rejects_pretrained_coco_mapping():
    with pytest.raises(ValueError, match="không khớp"):
        ensure_class_schema(
            {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle"},
            ("person", "bicycle", "motorcycle", "car"),
        )
