from types import SimpleNamespace

import numpy as np
import pytest

from secondeye.multimodal.depth import attach_depth_zones, relative_depth_band
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
    orchestrator = SystemOrchestrator(
        cooldown_seconds=4.0, clock=lambda: clock.value
    )
    detection = {
        "label": "chair",
        "direction": "center",
        "obstacle_candidate": True,
        "depth_zone": None,
    }
    assert orchestrator.obstacle_alerts([detection]) == []

    detection["depth_zone"] = "near"
    assert len(orchestrator.obstacle_alerts([detection])) == 1
    assert orchestrator.state is SystemState.OBSTACLE
    assert orchestrator.obstacle_alerts([detection]) == []

    clock.value += 4.1
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


def test_second_eye_system_integrates_detection_depth_risk_and_tts():
    tts = _FakeTts()
    system = SecondEyeSystem(
        detector=_FakeDetector(), depth=_FakeDepth(), tts=tts
    )

    result = system.process_frame(np.zeros((6, 6, 3), dtype=np.uint8))

    assert result["mode"] == "pretrained_integration"
    assert result["state"] == "OBSTACLE"
    assert result["detection"]["detections"][0]["depth_zone"] == "near"
    assert result["alerts"][0]["text"] == "Cảnh báo, có ghế ở gần phía trước."
    assert tts.messages == [("Cảnh báo, có ghế ở gần phía trước.", True)]
