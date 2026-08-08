"""Typed configuration for the local YOLO detection pipeline."""

from __future__ import annotations

import tomllib
import os
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
            if (
                (candidate / "pyproject.toml").is_file()
                and (candidate / "configs" / "yolo11_obstacles.toml").is_file()
            ):
                return candidate
    return Path.cwd().resolve()


PROJECT_ROOT = discover_project_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "yolo11_obstacles.toml"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    base_weights: str
    image_size: int
    confidence_threshold: float
    iou_threshold: float
    device: str


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
class DetectionPipelineConfig:
    source_path: Path
    model: ModelConfig
    training: TrainingConfig
    export: ExportConfig
    paths: PathConfig
    class_names: tuple[str, ...]
    candidate_classes: frozenset[str]
    central_zone_fraction: float


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
    training_raw = _require_table(raw, "training")
    export_raw = _require_table(raw, "export")
    paths_raw = _require_table(raw, "paths")
    schema_raw = _require_table(raw, "class_schema")
    risk_raw = _require_table(raw, "risk")
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

    return DetectionPipelineConfig(
        source_path=resolved_path,
        model=model,
        training=training,
        export=export,
        paths=PathConfig(
            dataset_root=_project_path(str(paths_raw["dataset_root"]), config_project_root),
            runs_root=_project_path(str(paths_raw["runs_root"]), config_project_root),
            artifact_root=_project_path(str(paths_raw["artifact_root"]), config_project_root),
        ),
        class_names=class_names,
        candidate_classes=candidate_classes,
        central_zone_fraction=central_zone_fraction,
    )
