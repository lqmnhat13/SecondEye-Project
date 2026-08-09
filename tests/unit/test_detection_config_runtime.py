import pytest

from secondeye.detection.config import load_detection_config
from secondeye.detection.pipeline import build_parser
from secondeye.detection.runtime import ensure_class_schema


def test_default_yolo26_config_has_expected_second_eye_schema():
    config = load_detection_config()

    assert config.model.base_weights == "yolo26m.pt"
    assert config.model.device == "auto"
    assert len(config.class_names) == 15
    assert config.class_names[6:11] == (
        "doorway_open",
        "door_closed",
        "glass_door",
        "stairs_up",
        "stairs_down",
    )
    assert config.candidate_classes <= set(config.class_names)


def test_camera_demo_uses_configured_yolo26_without_model_argument():
    args = build_parser().parse_args(["camera-demo", "--camera", "1"])

    assert args.command == "camera-demo"
    assert args.camera == 1
    assert not hasattr(args, "model")


def test_schema_guard_rejects_pretrained_coco_mapping():
    with pytest.raises(ValueError, match="không khớp"):
        ensure_class_schema(
            {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle"},
            ("person", "bicycle", "motorcycle", "car"),
        )
