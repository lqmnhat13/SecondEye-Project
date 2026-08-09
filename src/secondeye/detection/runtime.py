"""Runtime helpers shared by local detection commands."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import PROJECT_ROOT


ULTRALYTICS_CONFIG_DIR = PROJECT_ROOT / ".cache" / "ultralytics"


def require_detection_runtime() -> tuple[Any, Any, Any]:
    """Import the heavy optional dependencies only for commands that need them."""
    ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))
    try:
        import cv2
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            'Thiếu dependency detection. Chạy: python -m pip install ".[detection]"'
        ) from exc
    return cv2, torch, YOLO


def select_device(requested: str, torch_module: Any) -> str:
    """Select CUDA, then Apple MPS, then CPU when configured as auto."""
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda:0"
    mps = getattr(torch_module.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int, torch_module: Any) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - NumPy is installed with detection extra
        pass
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def synchronize_device(device: str, torch_module: Any) -> None:
    if device.startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
    elif device == "mps":
        mps = getattr(torch_module, "mps", None)
        if mps is not None and hasattr(mps, "synchronize"):
            mps.synchronize()


def normalized_model_names(names: Mapping[int, str] | Sequence[str]) -> tuple[str, ...]:
    if isinstance(names, Mapping):
        return tuple(str(names[index]) for index in sorted(names))
    return tuple(map(str, names))


def ensure_class_schema(
    names: Mapping[int, str] | Sequence[str], expected: Sequence[str]
) -> None:
    actual = normalized_model_names(names)
    expected_tuple = tuple(expected)
    if actual != expected_tuple:
        raise ValueError(
            "Schema lớp của model không khớp cấu hình. "
            f"Model có {len(actual)} lớp {actual}; cấu hình cần "
            f"{len(expected_tuple)} lớp {expected_tuple}. "
            "Hãy dùng best.pt đã train với dataset SecondEye, không dùng trực tiếp "
            "yolo26m.pt pretrained COCO."
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def git_commit() -> str | None:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return value or None
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_manifest(device: str) -> dict[str, object]:
    packages = {}
    for package in ("ultralytics", "torch", "opencv-python", "onnx", "onnxruntime"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": device,
        "packages": packages,
        "executable": Path(sys.executable).name,
    }
