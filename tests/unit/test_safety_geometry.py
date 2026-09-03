import numpy as np
import pytest

from secondeye.detection.config import load_detection_config
from secondeye.detection.geometry import (
    CameraIntrinsics,
    GeometryObstacleConfig,
    detect_geometry_obstacles,
    fuse_geometry_with_detections,
)
from secondeye.evaluation.safety import evaluate_safety_rows
from secondeye.multimodal.depth import (
    attach_metric_depth_zones,
    metric_depth_band,
)
from secondeye.multimodal.depth_provider import AlignedMetricDepthFrame
from secondeye.system.orchestrator import (
    SystemOrchestrator,
    SystemState,
    compose_obstacle_announcement,
)
from secondeye.system.tracking import DetectionTracker
from secondeye.system.pipeline import SecondEyeSystem


def _floor_with_obstacle():
    height, width = 120, 160
    intrinsics = CameraIntrinsics.from_horizontal_fov(width, height, 60.0)
    yy, xx = np.indices((height, width))
    depth = np.full((height, width), np.nan, dtype=np.float32)
    floor = yy > intrinsics.cy + 2
    depth[floor] = (1.2 * intrinsics.fy / (yy - intrinsics.cy))[floor]
    depth[(yy >= 55) & (yy < 98) & (xx >= 64) & (xx < 96)] = 1.2
    return depth, intrinsics


def test_geometry_detects_an_unlabelled_obstacle_above_the_floor():
    depth, intrinsics = _floor_with_obstacle()

    obstacles, diagnostics = detect_geometry_obstacles(
        depth,
        intrinsics=intrinsics,
        config=GeometryObstacleConfig(min_component_pixels=30),
    )

    assert diagnostics["usable"] is True
    assert diagnostics["floor_inlier_ratio"] > 0.7
    assert len(obstacles) == 1
    assert obstacles[0]["label"] == "unknown_obstacle"
    assert obstacles[0]["geometry_confirmed"] is True
    assert obstacles[0]["distance_m"] == pytest.approx(1.2, abs=0.01)
    assert obstacles[0]["proximity_zone"] == "near"


def test_geometry_abstains_when_a_floor_plane_cannot_be_estimated():
    depth = np.full((120, 160), np.nan, dtype=np.float32)
    depth[90:95, 70:75] = 1.0

    obstacles, diagnostics = detect_geometry_obstacles(depth)

    assert obstacles == []
    assert diagnostics["usable"] is False
    assert diagnostics["reason"] == "floor_plane_unavailable"


def test_geometry_region_receives_semantic_label_without_losing_metric_evidence():
    geometry = [
        {
            "label": "unknown_obstacle",
            "bbox_xyxy": [40.0, 40.0, 60.0, 80.0],
            "distance_m": 1.1,
            "depth_zone": "near",
            "proximity_zone": "near",
            "direction": "center",
            "obstacle_candidate": True,
            "geometry_confirmed": True,
            "safety_evaluable": True,
        }
    ]
    semantic = [{"label": "chair", "bbox_xyxy": [35.0, 35.0, 65.0, 85.0]}]

    fused = fuse_geometry_with_detections(semantic, geometry)

    assert len(fused) == 1
    assert fused[0]["label"] == "chair"
    assert fused[0]["geometry_confirmed"] is True
    assert fused[0]["distance_m"] == 1.1


def test_metric_bbox_depth_is_in_metres_but_not_safety_evidence_by_itself():
    depth = np.full((20, 20), 1.25, dtype=np.float32)

    result = attach_metric_depth_zones(
        [{"label": "chair", "bbox_xyxy": [0, 0, 20, 20]}], depth
    )[0]

    assert result["distance_m"] == pytest.approx(1.25)
    assert result["proximity_zone"] == "near"
    assert result["safety_evaluable"] is False
    assert metric_depth_band(0.5) == "emergency"


def test_tracker_keeps_identity_across_label_changes_and_computes_ttc():
    tracker = DetectionTracker(iou_threshold=0.2, speed_smoothing=0.0)
    first = tracker.update(
        [
            {
                "label": "unknown_obstacle",
                "bbox_xyxy": [10, 10, 30, 40],
                "distance_m": 2.0,
            }
        ],
        timestamp=1.0,
    )[0]
    second = tracker.update(
        [
            {
                "label": "chair",
                "bbox_xyxy": [11, 10, 31, 40],
                "distance_m": 1.5,
            }
        ],
        timestamp=2.0,
    )[0]

    assert second["track_id"] == first["track_id"]
    assert second["track_hits"] == 2
    assert second["approach_speed_mps"] == pytest.approx(0.5)
    assert second["time_to_collision_s"] == pytest.approx(3.0)


def test_detection_only_updates_do_not_shorten_metric_speed_interval():
    tracker = DetectionTracker(iou_threshold=0.2, speed_smoothing=0.0)
    tracker.update(
        [{"label": "chair", "bbox_xyxy": [10, 10, 30, 40], "distance_m": 2.0}],
        timestamp=1.0,
    )
    tracker.update(
        [{"label": "chair", "bbox_xyxy": [10, 10, 30, 40]}],
        timestamp=1.8,
    )
    result = tracker.update(
        [{"label": "chair", "bbox_xyxy": [10, 10, 30, 40], "distance_m": 1.5}],
        timestamp=2.0,
    )[0]

    assert result["approach_speed_mps"] == pytest.approx(0.5)


def test_orchestrator_keys_by_track_and_combines_simultaneous_alerts():
    orchestrator = SystemOrchestrator(confirmation_frames=1)
    detections = [
        {
            "track_id": 4,
            "label": "person",
            "direction": "center",
            "distance_m": 1.2,
            "obstacle_candidate": True,
            "proximity_zone": "near",
        },
        {
            "track_id": 5,
            "label": "chair",
            "direction": "left",
            "distance_m": 1.5,
            "obstacle_candidate": True,
            "proximity_zone": "near",
        },
    ]

    alerts = orchestrator.obstacle_alerts(detections)

    assert {alert.key for alert in alerts} == {"track:4", "track:5"}
    assert compose_obstacle_announcement(alerts) == (
        "Cẩn thận: người phía trước, cách 1.2 mét; " "ghế bên trái, cách 1.5 mét."
    )
    assert orchestrator.obstacle_alerts(detections) == []
    assert orchestrator.state is SystemState.OBSTACLE


def test_emergency_metric_observation_bypasses_confirmation_delay():
    orchestrator = SystemOrchestrator(confirmation_frames=3)
    alerts = orchestrator.obstacle_alerts(
        [
            {
                "track_id": 1,
                "label": "unknown_obstacle",
                "direction": "center",
                "obstacle_candidate": True,
                "proximity_zone": "emergency",
            }
        ]
    )

    assert len(alerts) == 1


def test_obstacle_state_expires_if_metric_evidence_stops_arriving():
    clock = type("Clock", (), {"value": 10.0})()
    orchestrator = SystemOrchestrator(
        confirmation_frames=1,
        max_evidence_gap_seconds=0.5,
        clock=lambda: clock.value,
    )
    orchestrator.obstacle_alerts(
        [
            {
                "track_id": 1,
                "label": "chair",
                "obstacle_candidate": True,
                "proximity_zone": "near",
            }
        ]
    )
    clock.value += 0.6

    orchestrator.evidence_unavailable()

    assert orchestrator.state is SystemState.IDLE


def test_aligned_sensor_depth_preserves_intrinsics_and_metric_semantics():
    frame = AlignedMetricDepthFrame(
        metric_depth_m=np.ones((4, 5), dtype=np.float32),
        captured_at=10.0,
        fx=100.0,
        fy=101.0,
        cx=2.0,
        cy=1.5,
        source="arkit_scene_depth",
    )

    result = frame.as_result()

    assert result["depth_type"] == "metric"
    assert result["model"] == "arkit_scene_depth"
    assert result["intrinsics"] == {"fx": 100.0, "fy": 101.0, "cx": 2.0, "cy": 1.5}


def test_event_evaluation_reports_recall_latency_false_and_stale_alerts():
    rows = [
        {"timestamp_s": 0.0, "hazard_present": False, "alert": False},
        {
            "timestamp_s": 1.0,
            "hazard_present": True,
            "hazard_id": "a",
            "critical": True,
            "alert": False,
        },
        {
            "timestamp_s": 1.4,
            "hazard_present": True,
            "hazard_id": "a",
            "critical": True,
            "alert": True,
            "source_age_ms": 200,
        },
        {"timestamp_s": 30.0, "hazard_present": True, "hazard_id": "b", "alert": False},
        {
            "timestamp_s": 60.0,
            "hazard_present": False,
            "alert": True,
            "source_age_ms": 900,
        },
    ]

    metrics = evaluate_safety_rows(rows)

    assert metrics.hazard_events == 2
    assert metrics.hazard_event_recall == pytest.approx(0.5)
    assert metrics.critical_event_recall == pytest.approx(1.0)
    assert metrics.alert_latency_p50_ms == pytest.approx(400.0)
    assert metrics.false_alerts_per_minute == pytest.approx(1.0)
    assert metrics.stale_alerts == 1


def test_default_config_selects_metric_depth_and_temporal_confirmation():
    config = load_detection_config()

    assert "Metric-Indoor" in config.depth.model_name
    assert len(config.depth.model_revision or "") == 40
    assert config.depth.allow_relative_alerts is False
    assert config.safety.confirmation_frames == 2
    assert config.safety.max_result_age_seconds == pytest.approx(0.75)


class _Detector:
    def predict_bgr(self, image):
        return {
            "detections": [
                {
                    "label": "chair",
                    "confidence": 0.9,
                    "bbox_xyxy": [62, 52, 98, 100],
                    "direction": "center",
                    "obstacle_candidate": True,
                }
            ],
            "image_size": {"height": 120, "width": 160},
        }


def test_pipeline_suppresses_safety_evaluation_after_source_age_deadline():
    depth, _ = _floor_with_obstacle()
    system = SecondEyeSystem(
        detector=_Detector(),
        orchestrator=SystemOrchestrator(confirmation_frames=1),
    )
    detection = _Detector().predict_bgr(None)

    result = system.fuse_detection_and_depth(
        detection,
        {"depth_type": "metric", "metric_depth_m": depth, "usable": True},
        safety_age_check=lambda: False,
    )

    assert result["geometry"]["usable"] is True
    assert result["risk_evidence_current"] is False
    assert result["alerts"] == []
    assert result["state"] == "IDLE"


def test_pipeline_never_uses_relative_depth_as_alert_evidence():
    system = SecondEyeSystem(
        detector=_Detector(),
        orchestrator=SystemOrchestrator(confirmation_frames=1),
    )
    relative = np.ones((120, 160), dtype=np.float32)

    result = system.fuse_detection_and_depth(
        _Detector().predict_bgr(None),
        {
            "depth_type": "relative",
            "relative_inverse_depth": relative,
            "usable": True,
        },
    )

    assert result["detection"]["detections"][0]["depth_zone"] == "near"
    assert result["detection"]["detections"][0]["safety_evaluable"] is False
    assert result["alerts"] == []
    assert result["depth_used_for_alert"] is False


def test_open_vocabulary_expands_scene_semantics_but_not_safety():
    class SemanticDetector:
        def predict_bgr(self, image):
            return {
                "model": "grounding-dino-test",
                "latency_ms": 5.0,
                "detections": [
                    {
                        "label": "door",
                        "confidence": 0.8,
                        "bbox_xyxy": [100, 10, 150, 100],
                        "direction": "right",
                        "obstacle_candidate": False,
                        "safety_evaluable": False,
                    }
                ],
            }

    system = SecondEyeSystem(detector=_Detector(), semantic_detector=SemanticDetector())

    result = system.describe_scene(
        np.zeros((120, 160, 3), dtype=np.uint8),
        detection_result=_Detector().predict_bgr(None),
    )

    assert result["source"] == "pretrained_detection_plus_open_vocabulary"
    assert "cửa bên phải" in result["description"]
