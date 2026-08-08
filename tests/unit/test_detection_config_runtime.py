import pytest

from secondeye.detection.config import load_detection_config
from secondeye.detection.runtime import ensure_class_schema


def test_default_yolo11_config_has_expected_second_eye_schema():
    config = load_detection_config()

    assert config.model.base_weights == "yolo11n.pt"
    assert config.model.device == "auto"
    assert len(config.class_names) == 12
    assert "pothole" in config.class_names
    assert config.candidate_classes <= set(config.class_names)


def test_schema_guard_rejects_pretrained_coco_mapping():
    with pytest.raises(ValueError, match="không khớp"):
        ensure_class_schema(
            {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle"},
            ("person", "bicycle", "motorcycle", "car"),
        )
