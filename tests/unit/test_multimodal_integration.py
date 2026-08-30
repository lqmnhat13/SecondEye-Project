from types import SimpleNamespace
import time

import numpy as np
import pytest

from secondeye.multimodal.depth import attach_depth_zones, relative_depth_band
from secondeye.multimodal.quality import assess_image_quality
from secondeye.multimodal.speech import (
    localize_vqa_answer,
    normalize_vietnamese_speech,
)
from secondeye.system.camera import AsyncVisionRuntime, LatestFrameBuffer
from secondeye.system.cli import build_parser
from secondeye.system.orchestrator import SystemOrchestrator, SystemState
from secondeye.system.pipeline import SecondEyeSystem


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, "far"), (0.4, "medium"), (0.9, "near")],
)
def test_relative_depth_band_is_explicitly_non_metric(value, expected):
    assert relative_depth_band(value) == expected


def test_attach_depth_zones_uses_bbox_median():
    depth = np.zeros((10, 10), dtype=np.float32)
    depth[2:8, 2:8] = 0.9
    detections = [{"label": "chair", "bbox_xyxy": [2, 2, 8, 8]}]

    enriched = attach_depth_zones(detections, depth)

    assert enriched[0]["depth_zone"] == "near"
    assert enriched[0]["relative_depth"] == pytest.approx(0.9)
    assert "depth_zone" not in detections[0]


def test_orchestrator_requires_depth_confirmation_and_applies_cooldown():
    clock = SimpleNamespace(value=10.0)
    orchestrator = SystemOrchestrator(cooldown_seconds=4.0, clock=lambda: clock.value)
    detection = {
        "label": "chair",
        "direction": "center",
        "obstacle_candidate": True,
        "depth_zone": None,
    }
    assert orchestrator.obstacle_alerts([detection]) == []

    detection["depth_zone"] = "near"
    assert orchestrator.obstacle_alerts([detection]) == []
    assert len(orchestrator.obstacle_alerts([detection])) == 1
    assert orchestrator.state is SystemState.OBSTACLE
    assert orchestrator.obstacle_alerts([detection]) == []

    clock.value += 4.1
    assert len(orchestrator.obstacle_alerts([detection])) == 1


def test_two_objects_in_one_frame_do_not_fake_temporal_confirmation():
    orchestrator = SystemOrchestrator(confirmation_frames=2)
    detection = {
        "label": "chair",
        "direction": "center",
        "obstacle_candidate": True,
        "depth_zone": "near",
    }

    assert orchestrator.obstacle_alerts([detection, dict(detection)]) == []
    assert len(orchestrator.obstacle_alerts([detection])) == 1


class _FakeDetector:
    def predict_bgr(self, image):
        return {
            "detections": [
                {
                    "label": "chair",
                    "confidence": 0.9,
                    "bbox_xyxy": [1, 1, 5, 5],
                    "direction": "center",
                    "obstacle_candidate": True,
                }
            ],
            "latency_ms": 1.0,
        }


class _FakeDepth:
    def predict_bgr(self, image):
        return {
            "relative_inverse_depth": np.ones((6, 6), dtype=np.float32),
            "latency_ms": 2.0,
        }


class _FakeTts:
    def __init__(self):
        self.messages = []

    def speak(self, text, *, interrupt=False):
        self.messages.append((text, interrupt))

    def stop(self):
        self.messages.append(("STOP", True))


class _FakeVqa:
    def ask_bgr(self, image, question):
        return {"answer": "four", "question": question, "abstained": False}


class _FakeEnglishVqa:
    def ask_bgr(self, image, question):
        return {"answer": "black shirt", "question": question, "abstained": False}


class _FakeTranslator:
    def translate(self, text):
        assert text == "black shirt"
        return {
            "model": "fake-en-vi",
            "source": text,
            "translation": "áo sơ mi màu đen",
        }


def test_second_eye_system_integrates_detection_depth_risk_and_tts():
    tts = _FakeTts()
    system = SecondEyeSystem(detector=_FakeDetector(), depth=_FakeDepth(), tts=tts)

    system.process_frame(np.zeros((6, 6, 3), dtype=np.uint8))
    result = system.process_frame(np.zeros((6, 6, 3), dtype=np.uint8))

    assert result["mode"] == "pretrained_integration"
    assert result["state"] == "OBSTACLE"
    assert result["detection"]["detections"][0]["depth_zone"] == "near"
    assert result["alerts"][0]["text"] == "Cảnh báo, có ghế ở gần phía trước."
    assert tts.messages == [("Cảnh báo, có ghế ở gần phía trước.", True)]


def test_second_eye_system_reads_localized_vqa_answer():
    tts = _FakeTts()
    system = SecondEyeSystem(detector=_FakeDetector(), vqa=_FakeVqa(), tts=tts)

    pattern = (np.indices((300, 300)).sum(axis=0) % 2 * 255).astype(np.uint8)
    image = np.repeat(pattern[:, :, None], 3, axis=2)
    result = system.ask(image, "How many people?")

    assert result["spoken_answer_vi"] == "bốn"
    assert result["localization_abstained"] is False
    assert tts.messages == [("bốn", False)]


def test_second_eye_system_translates_unknown_english_vqa_answer():
    tts = _FakeTts()
    system = SecondEyeSystem(
        detector=_FakeDetector(),
        vqa=_FakeEnglishVqa(),
        translator=_FakeTranslator(),
        tts=tts,
    )
    pattern = (np.indices((300, 300)).sum(axis=0) % 2 * 255).astype(np.uint8)
    image = np.repeat(pattern[:, :, None], 3, axis=2)

    result = system.ask(image, "What is the person wearing?")

    assert result["spoken_answer_vi"] == "áo sơ mi màu đen"
    assert result["translation_used"] is True
    assert result["localization_abstained"] is False
    assert result["abstained"] is False
    assert tts.messages == [("áo sơ mi màu đen", False)]


def test_latest_frame_buffer_drops_stale_frames():
    frames = LatestFrameBuffer()
    frames.publish(np.zeros((2, 2, 3), dtype=np.uint8), captured_at=1.0)
    frames.publish(np.ones((2, 2, 3), dtype=np.uint8), captured_at=2.0)

    packet = frames.wait_for_new(-1, timeout=0.01)

    assert packet is not None
    assert packet.frame_id == 1
    assert packet.captured_at == 2.0
    assert np.all(packet.frame == 1)


def test_async_runtime_processes_latest_available_frame_only():
    frames = LatestFrameBuffer()
    for value in range(5):
        frames.publish(np.full((6, 6, 3), value, dtype=np.uint8))
    runtime = AsyncVisionRuntime(
        SecondEyeSystem(detector=_FakeDetector(), depth=_FakeDepth()),
        frames,
        detection_fps=100.0,
        depth_fps=100.0,
    ).start()
    try:
        deadline = time.monotonic() + 1.0
        result = None
        while result is None and time.monotonic() < deadline:
            result = runtime.latest()
            time.sleep(0.005)
        assert result is not None
        assert result["frame_id"] == 4
        assert result["depth_age_ms"] is not None
    finally:
        runtime.stop()


@pytest.mark.parametrize(
    ("raw", "spoken", "abstained"),
    [
        ("four", "bốn", False),
        ("left", "bên trái", False),
        ("blue", "xanh dương", False),
        ("bus", "xe buýt", False),
        (
            "red bus",
            "red bus",
            True,
        ),
    ],
)
def test_vqa_answers_are_localized_or_safely_abstained(raw, spoken, abstained):
    assert localize_vqa_answer(raw) == (spoken, abstained)


def test_vietnamese_speech_normalizes_project_terms():
    assert normalize_vietnamese_speech("  SecondEye   dùng YOLO và OCR ") == (
        "Se-cần Ai dùng Yô-lô và Ô Xi A"
    )


def test_camera_cli_defaults_to_async_rates_and_vietnamese_voice():
    args = build_parser().parse_args(["camera", "--camera", "1", "--depth"])

    assert args.voice == "Linh"
    assert args.speech_rate == 165
    assert args.width == 1280
    assert args.height == 720
    assert args.detection_fps == 12.0
    assert args.depth_fps == 3.0


def test_demo_cli_enables_all_pretrained_mvp_features():
    args = build_parser().parse_args(["demo", "--camera", "1"])

    assert args.depth is True
    assert args.ocr is True
    assert args.lazy_semantic is True
    assert args.priority_audio is True
    assert args.semantic_device == "cpu"
    assert args.max_depth_age == 1.5


def test_quality_gate_rejects_dark_or_blurry_frames():
    quality = assess_image_quality(np.zeros((300, 300, 3), dtype=np.uint8))

    assert quality.acceptable is False
    assert quality.reason == "too_dark"


def test_vqa_blocks_navigation_advice_before_calling_model():
    system = SecondEyeSystem(detector=_FakeDetector(), vqa=_FakeVqa())

    result = system.ask(
        np.zeros((300, 300, 3), dtype=np.uint8),
        "Có an toàn để đi qua đường này không?",
    )

    assert result["abstained"] is True
    assert result["reason"] == "navigation_request_blocked"


def test_scene_description_is_grounded_in_detection_labels():
    system = SecondEyeSystem(detector=_FakeDetector())

    result = system.describe_scene(
        detection_result={
            "detections": [
                {"label": "chair", "direction": "center"},
                {"label": "person", "direction": "left"},
            ]
        }
    )

    assert result["source"] == "pretrained_detection"
    assert "ghế phía trước" in result["description"]
    assert "người bên trái" in result["description"]
