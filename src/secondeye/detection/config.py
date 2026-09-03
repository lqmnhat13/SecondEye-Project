"""Typed configuration for the local YOLO detection pipeline."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def discover_project_root() -> Path:
    """Find the checkout even when this package is installed as a wheel."""
    configured = os.environ.get("SECONDEYE_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    search_starts = (Path.cwd().resolve(), Path(__file__).resolve())
    visited: set[Path] = set()
    for start in search_starts:
        for candidate in (start, *start.parents):
            if candidate in visited:
                continue
            visited.add(candidate)
            if (candidate / "pyproject.toml").is_file() and (
                candidate / "configs" / "pretrained_indoor.toml"
            ).is_file():
                return candidate
    return Path.cwd().resolve()


PROJECT_ROOT = discover_project_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pretrained_indoor.toml"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    base_weights: str
    image_size: int
    confidence_threshold: float
    iou_threshold: float
    device: str


@dataclass(frozen=True, slots=True)
class PretrainedCocoConfig:
    global_confidence_threshold: float
    optimization_objective: str
    benchmark_images: int
    benchmark_boxes: int
    unsupported_second_eye_classes: tuple[str, ...]
    class_mapping: tuple[tuple[str, str], ...]
    class_thresholds: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    patience: int
    workers: int
    seed: int
    deterministic: bool


@dataclass(frozen=True, slots=True)
class ExportConfig:
    format: str
    opset: int
    dynamic: bool
    simplify: bool


@dataclass(frozen=True, slots=True)
class PathConfig:
    dataset_root: Path
    runs_root: Path
    artifact_root: Path


@dataclass(frozen=True, slots=True)
class DepthRuntimeConfig:
    model_name: str
    model_revision: str | None
    emergency_distance_m: float
    warning_distance_m: float
    medium_distance_m: float
    metric_percentile: float
    allow_relative_alerts: bool


@dataclass(frozen=True, slots=True)
class GeometryRuntimeConfig:
    horizontal_fov_degrees: float
    min_depth_m: float
    max_depth_m: float
    floor_region_top_fraction: float
    corridor_top_fraction: float
    corridor_top_width_fraction: float
    corridor_bottom_width_fraction: float
    min_obstacle_height_m: float
    max_obstacle_height_m: float
    floor_ransac_threshold_m: float
    floor_min_inlier_ratio: float
    floor_min_points: int
    min_component_pixels: int


@dataclass(frozen=True, slots=True)
class SafetyRuntimeConfig:
    confirmation_frames: int
    rearm_absent_frames: int
    cooldown_seconds: float
    emergency_ttc_seconds: float
    max_depth_age_seconds: float
    max_result_age_seconds: float
    max_evidence_gap_seconds: float


@dataclass(frozen=True, slots=True)
class DetectionPipelineConfig:
    source_path: Path
    model: ModelConfig
    pretrained_coco: PretrainedCocoConfig
    training: TrainingConfig
    export: ExportConfig
    paths: PathConfig
    class_names: tuple[str, ...]
    candidate_classes: frozenset[str]
    central_zone_fraction: float
    depth: DepthRuntimeConfig
    geometry: GeometryRuntimeConfig
    safety: SafetyRuntimeConfig


def _require_table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Thiếu bảng cấu hình [{name}]")
    return value


def _project_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _probability(value: Any, name: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} phải nằm trong [0, 1]")
    return result


def load_detection_config(path: Path = DEFAULT_CONFIG_PATH) -> DetectionPipelineConfig:
    """Load and validate a local detection pipeline TOML file."""
    resolved_path = path.expanduser().resolve()
    with resolved_path.open("rb") as stream:
        raw = tomllib.load(stream)

    model_raw = _require_table(raw, "model")
    pretrained_raw = _require_table(raw, "pretrained_coco")
    training_raw = _require_table(raw, "training")
    export_raw = _require_table(raw, "export")
    paths_raw = _require_table(raw, "paths")
    schema_raw = _require_table(raw, "class_schema")
    risk_raw = _require_table(raw, "risk")
    depth_raw = raw.get("depth", {})
    geometry_raw = raw.get("geometry", {})
    safety_raw = raw.get("safety", {})
    if not all(
        isinstance(value, dict) for value in (depth_raw, geometry_raw, safety_raw)
    ):
        raise ValueError("depth/geometry/safety phải là TOML table")
    config_project_root = (
        resolved_path.parent.parent
        if resolved_path.parent.name == "configs"
        else resolved_path.parent
    )

    class_names = tuple(str(name).strip() for name in schema_raw.get("names", ()))
    if not class_names or any(not name for name in class_names):
        raise ValueError("class_schema.names không được rỗng")
    if len(class_names) != len(set(class_names)):
        raise ValueError("class_schema.names không được có tên lớp trùng")

    candidate_classes = frozenset(
        str(name).strip() for name in risk_raw.get("candidate_classes", ())
    )
    unknown_candidates = candidate_classes - set(class_names)
    if unknown_candidates:
        raise ValueError(
            "risk.candidate_classes chứa lớp ngoài schema: "
            + ", ".join(sorted(unknown_candidates))
        )

    model = ModelConfig(
        base_weights=str(model_raw["base_weights"]),
        image_size=int(model_raw["image_size"]),
        confidence_threshold=_probability(
            model_raw["confidence_threshold"], "model.confidence_threshold"
        ),
        iou_threshold=_probability(model_raw["iou_threshold"], "model.iou_threshold"),
        device=str(model_raw.get("device", "auto")),
    )
    if model.image_size <= 0:
        raise ValueError("model.image_size phải dương")

    mapping_raw = pretrained_raw.get("class_mapping", {})
    thresholds_raw = pretrained_raw.get("class_thresholds", {})
    if not isinstance(mapping_raw, dict) or not isinstance(thresholds_raw, dict):
        raise ValueError("pretrained_coco mapping/thresholds phải là TOML table")
    class_mapping = tuple(
        (str(source).strip(), str(target).strip())
        for source, target in mapping_raw.items()
    )
    if any(not source or target not in class_names for source, target in class_mapping):
        raise ValueError("pretrained_coco.class_mapping chứa lớp rỗng/ngoài schema")
    mapped_classes = tuple(target for _, target in class_mapping)
    if len(mapped_classes) != len(set(mapped_classes)):
        raise ValueError(
            "pretrained_coco.class_mapping không được ánh xạ trùng lớp đích"
        )
    unsupported_classes = tuple(
        str(name).strip()
        for name in pretrained_raw.get("unsupported_second_eye_classes", ())
    )
    if len(unsupported_classes) != len(set(unsupported_classes)):
        raise ValueError("pretrained_coco.unsupported_second_eye_classes bị trùng")
    expected_unsupported = set(class_names) - set(mapped_classes)
    if set(unsupported_classes) != expected_unsupported:
        raise ValueError(
            "pretrained_coco.unsupported_second_eye_classes phải đúng bằng các lớp "
            "SecondEye chưa có ánh xạ COCO"
        )
    class_thresholds = tuple(
        (str(name).strip(), _probability(value, f"pretrained_coco.{name}"))
        for name, value in thresholds_raw.items()
    )
    if any(name not in class_names for name, _ in class_thresholds):
        raise ValueError("pretrained_coco.class_thresholds chứa lớp ngoài schema")
    if {name for name, _ in class_thresholds} != set(mapped_classes):
        raise ValueError(
            "pretrained_coco.class_thresholds phải bao phủ đúng các lớp đã ánh xạ"
        )
    pretrained_coco = PretrainedCocoConfig(
        global_confidence_threshold=_probability(
            pretrained_raw["global_confidence_threshold"],
            "pretrained_coco.global_confidence_threshold",
        ),
        optimization_objective=str(pretrained_raw["optimization_objective"]),
        benchmark_images=int(pretrained_raw["benchmark_images"]),
        benchmark_boxes=int(pretrained_raw["benchmark_boxes"]),
        unsupported_second_eye_classes=unsupported_classes,
        class_mapping=class_mapping,
        class_thresholds=class_thresholds,
    )
    if pretrained_coco.benchmark_images <= 0 or pretrained_coco.benchmark_boxes <= 0:
        raise ValueError("pretrained_coco benchmark counts phải dương")

    training = TrainingConfig(
        epochs=int(training_raw["epochs"]),
        batch_size=int(training_raw["batch_size"]),
        patience=int(training_raw["patience"]),
        workers=int(training_raw["workers"]),
        seed=int(training_raw["seed"]),
        deterministic=bool(training_raw.get("deterministic", True)),
    )
    if training.epochs <= 0 or training.batch_size <= 0:
        raise ValueError("training.epochs và training.batch_size phải dương")
    if training.patience < 0 or training.workers < 0:
        raise ValueError("training.patience và training.workers không được âm")

    export = ExportConfig(
        format=str(export_raw.get("format", "onnx")),
        opset=int(export_raw.get("opset", 17)),
        dynamic=bool(export_raw.get("dynamic", True)),
        simplify=bool(export_raw.get("simplify", True)),
    )
    if export.format != "onnx":
        raise ValueError("Pipeline hiện chỉ kiểm chứng export.format='onnx'")
    if export.opset <= 0:
        raise ValueError("export.opset phải dương")

    central_zone_fraction = float(risk_raw.get("central_zone_fraction", 0.4))
    if not 0.0 < central_zone_fraction < 1.0:
        raise ValueError("risk.central_zone_fraction phải nằm trong (0, 1)")

    depth = DepthRuntimeConfig(
        model_name=str(
            depth_raw.get(
                "model_name",
                "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
            )
        ),
        model_revision=(
            str(depth_raw["model_revision"])
            if depth_raw.get("model_revision")
            else None
        ),
        emergency_distance_m=float(depth_raw.get("emergency_distance_m", 0.8)),
        warning_distance_m=float(depth_raw.get("warning_distance_m", 1.8)),
        medium_distance_m=float(depth_raw.get("medium_distance_m", 3.0)),
        metric_percentile=float(depth_raw.get("metric_percentile", 25.0)),
        allow_relative_alerts=bool(depth_raw.get("allow_relative_alerts", False)),
    )
    if depth.allow_relative_alerts:
        raise ValueError(
            "depth.allow_relative_alerts=true bị từ chối trong safety runtime"
        )
    if not (
        0.0
        < depth.emergency_distance_m
        < depth.warning_distance_m
        < depth.medium_distance_m
    ):
        raise ValueError("ngưỡng depth metric không hợp lệ")
    geometry = GeometryRuntimeConfig(
        horizontal_fov_degrees=float(geometry_raw.get("horizontal_fov_degrees", 60.0)),
        min_depth_m=float(geometry_raw.get("min_depth_m", 0.15)),
        max_depth_m=float(geometry_raw.get("max_depth_m", 5.0)),
        floor_region_top_fraction=float(
            geometry_raw.get("floor_region_top_fraction", 0.52)
        ),
        corridor_top_fraction=float(geometry_raw.get("corridor_top_fraction", 0.35)),
        corridor_top_width_fraction=float(
            geometry_raw.get("corridor_top_width_fraction", 0.24)
        ),
        corridor_bottom_width_fraction=float(
            geometry_raw.get("corridor_bottom_width_fraction", 0.78)
        ),
        min_obstacle_height_m=float(geometry_raw.get("min_obstacle_height_m", 0.10)),
        max_obstacle_height_m=float(geometry_raw.get("max_obstacle_height_m", 2.50)),
        floor_ransac_threshold_m=float(
            geometry_raw.get("floor_ransac_threshold_m", 0.06)
        ),
        floor_min_inlier_ratio=float(geometry_raw.get("floor_min_inlier_ratio", 0.30)),
        floor_min_points=int(geometry_raw.get("floor_min_points", 180)),
        min_component_pixels=int(geometry_raw.get("min_component_pixels", 80)),
    )
    safety = SafetyRuntimeConfig(
        confirmation_frames=int(safety_raw.get("confirmation_frames", 2)),
        rearm_absent_frames=int(safety_raw.get("rearm_absent_frames", 3)),
        cooldown_seconds=float(safety_raw.get("cooldown_seconds", 4.0)),
        emergency_ttc_seconds=float(safety_raw.get("emergency_ttc_seconds", 1.5)),
        max_depth_age_seconds=float(safety_raw.get("max_depth_age_seconds", 0.50)),
        max_result_age_seconds=float(safety_raw.get("max_result_age_seconds", 0.75)),
        max_evidence_gap_seconds=float(safety_raw.get("max_evidence_gap_seconds", 1.0)),
    )
    if (
        safety.confirmation_frames <= 0
        or safety.rearm_absent_frames <= 0
        or min(
            safety.cooldown_seconds,
            safety.emergency_ttc_seconds,
            safety.max_depth_age_seconds,
            safety.max_result_age_seconds,
            safety.max_evidence_gap_seconds,
        )
        <= 0.0
    ):
        raise ValueError("cấu hình safety phải dương")

    return DetectionPipelineConfig(
        source_path=resolved_path,
        model=model,
        pretrained_coco=pretrained_coco,
        training=training,
        export=export,
        paths=PathConfig(
            dataset_root=_project_path(
                str(paths_raw["dataset_root"]), config_project_root
            ),
            runs_root=_project_path(str(paths_raw["runs_root"]), config_project_root),
            artifact_root=_project_path(
                str(paths_raw["artifact_root"]), config_project_root
            ),
        ),
        class_names=class_names,
        candidate_classes=candidate_classes,
        central_zone_fraction=central_zone_fraction,
        depth=depth,
        geometry=geometry,
        safety=safety,
    )
