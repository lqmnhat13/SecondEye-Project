from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import pytest

from secondeye.detection.dataset import (
    DatasetValidationError,
    dataset_fingerprint,
    safe_extract_dataset,
    validate_dataset,
)


CLASS_NAMES = ("person", "pothole")


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((16, 16, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _write_label(path: Path, content: str = "0 0.5 0.5 0.5 0.5\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _valid_dataset(root: Path) -> Path:
    _write_image(root / "images/train/train.png", 10)
    _write_label(root / "labels/train/train.txt")
    _write_image(root / "images/val/val.png", 20)
    _write_label(root / "labels/val/val.txt", "1 0.5 0.5 0.25 0.25\n")
    return root


def test_validate_dataset_reports_valid_split_stats(tmp_path):
    root = _valid_dataset(tmp_path / "dataset")

    stats = validate_dataset(root, CLASS_NAMES)

    assert stats["train"].images == 1
    assert stats["train"].boxes == 1
    assert stats["val"].class_counts == {1: 1}
    assert stats["test"].images == 0
    assert dataset_fingerprint(root)["file_count"] == 4


def test_dataset_fingerprint_changes_when_label_changes(tmp_path):
    root = _valid_dataset(tmp_path / "dataset")
    before = dataset_fingerprint(root)["sha256"]
    _write_label(root / "labels/val/val.txt", "1 0.5 0.5 0.20 0.20\n")

    assert dataset_fingerprint(root)["sha256"] != before


def test_validate_dataset_rejects_bbox_outside_normalized_image(tmp_path):
    root = _valid_dataset(tmp_path / "dataset")
    _write_label(root / "labels/train/train.txt", "0 0.9 0.5 0.5 0.5\n")

    with pytest.raises(DatasetValidationError, match="vượt ra ngoài biên"):
        validate_dataset(root, CLASS_NAMES)


def test_validate_dataset_rejects_identical_images_across_splits(tmp_path):
    root = _valid_dataset(tmp_path / "dataset")
    train_bytes = (root / "images/train/train.png").read_bytes()
    (root / "images/val/val.png").write_bytes(train_bytes)

    with pytest.raises(DatasetValidationError, match="nhiều split"):
        validate_dataset(root, CLASS_NAMES)


def test_safe_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with ZipFile(archive, "w") as stream:
        stream.writestr("../escaped.txt", "blocked")

    with pytest.raises(DatasetValidationError, match="path traversal"):
        safe_extract_dataset(archive, tmp_path / "extract")

    assert not (tmp_path / "escaped.txt").exists()
