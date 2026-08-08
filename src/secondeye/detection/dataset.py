"""Safe extraction and strict validation for YOLO detection datasets."""

from __future__ import annotations

import hashlib
import os
import stat
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
REQUIRED_SPLITS = ("train", "val")
OPTIONAL_SPLITS = ("test",)


class DatasetValidationError(ValueError):
    """Raised when a dataset cannot be trusted for training or evaluation."""


@dataclass(frozen=True, slots=True)
class SplitStats:
    images: int
    labeled_images: int
    background_images: int
    boxes: int
    class_counts: dict[int, int]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(dataset_root: Path) -> dict[str, object]:
    """Hash paths and bytes of all dataset images/labels for provenance."""
    dataset_root = dataset_root.expanduser().resolve()
    files = sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file()
        and (path.suffix.lower() in IMAGE_SUFFIXES or path.suffix.lower() == ".txt")
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(dataset_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return {"sha256": digest.hexdigest(), "file_count": len(files)}


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def safe_extract_dataset(archive_path: Path, destination: Path) -> Path:
    """Extract a ZIP without traversal/symlinks and return its dataset root.

    The destination must be absent or empty so this operation never destroys an
    existing local dataset.
    """
    archive_path = archive_path.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Thư mục đích phải rỗng: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise DatasetValidationError(
                    f"ZIP chứa symbolic link không được phép: {member.filename}"
                )
            target = (destination / member.filename).resolve()
            if os.path.commonpath((str(destination), str(target))) != str(destination):
                raise DatasetValidationError(
                    f"ZIP path traversal bị từ chối: {member.filename}"
                )
        archive.extractall(destination)

    return locate_dataset_root(destination)


def locate_dataset_root(search_root: Path) -> Path:
    """Find one directory containing images/train and labels/train."""
    search_root = search_root.expanduser().resolve()
    candidates = [
        path.resolve()
        for path in (search_root, *search_root.rglob("*"))
        if path.is_dir()
        and (path / "images" / "train").is_dir()
        and (path / "labels" / "train").is_dir()
    ]
    if len(candidates) != 1:
        rendered = ", ".join(map(str, candidates)) or "không có"
        raise DatasetValidationError(
            "Cần đúng một dataset root chứa images/train và labels/train; "
            f"đã tìm thấy: {rendered}"
        )
    return candidates[0]


def _image_keys(image_root: Path, images: list[Path]) -> dict[Path, Path]:
    keys: dict[Path, Path] = {}
    for image_path in images:
        key = image_path.relative_to(image_root).with_suffix("")
        if key in keys:
            raise DatasetValidationError(
                f"Hai ảnh dùng chung một label: {keys[key]} và {image_path}"
            )
        keys[key] = image_path
    return keys


def _validate_box(fields: list[str], class_count: int, location: str) -> int:
    if len(fields) != 5:
        raise DatasetValidationError(f"{location}: cần 5 trường, nhận {len(fields)}")
    try:
        class_id_float, x_center, y_center, width, height = map(float, fields)
    except ValueError as exc:
        raise DatasetValidationError(f"{location}: có giá trị không phải số") from exc
    if not class_id_float.is_integer():
        raise DatasetValidationError(f"{location}: class_id phải là số nguyên")
    class_id = int(class_id_float)
    if not 0 <= class_id < class_count:
        raise DatasetValidationError(
            f"{location}: class_id {class_id} ngoài [0, {class_count - 1}]"
        )
    values = (x_center, y_center, width, height)
    if not all(value == value and abs(value) != float("inf") for value in values):
        raise DatasetValidationError(f"{location}: bbox chứa NaN hoặc infinity")
    if not 0.0 <= x_center <= 1.0 or not 0.0 <= y_center <= 1.0:
        raise DatasetValidationError(f"{location}: tâm bbox phải nằm trong [0, 1]")
    if not 0.0 < width <= 1.0 or not 0.0 < height <= 1.0:
        raise DatasetValidationError(f"{location}: width/height phải nằm trong (0, 1]")
    tolerance = 1e-6
    if (
        x_center - width / 2 < -tolerance
        or x_center + width / 2 > 1 + tolerance
        or y_center - height / 2 < -tolerance
        or y_center + height / 2 > 1 + tolerance
    ):
        raise DatasetValidationError(f"{location}: bbox vượt ra ngoài biên ảnh chuẩn hóa")
    return class_id


def validate_split(dataset_root: Path, split: str, class_names: Sequence[str]) -> SplitStats:
    """Validate one split, including image decoding and every label row."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional installation
        raise RuntimeError('Thiếu OpenCV. Chạy: python -m pip install ".[detection]"') from exc

    image_root = dataset_root / "images" / split
    label_root = dataset_root / "labels" / split
    images = image_files(image_root)
    if split in REQUIRED_SPLITS and not images:
        raise DatasetValidationError(f"Split bắt buộc '{split}' không có ảnh")
    if not images:
        return SplitStats(0, 0, 0, 0, {})
    if not label_root.is_dir():
        raise DatasetValidationError(f"Thiếu thư mục nhãn: {label_root}")

    image_map = _image_keys(image_root, images)
    label_files = sorted(label_root.rglob("*.txt"))
    label_map = {path.relative_to(label_root).with_suffix(""): path for path in label_files}
    orphan_labels = sorted(set(label_map) - set(image_map))
    if orphan_labels:
        preview = ", ".join(str(key) for key in orphan_labels[:5])
        raise DatasetValidationError(f"Split {split} có label không có ảnh: {preview}")

    counts: Counter[int] = Counter()
    labeled_images = 0
    box_count = 0
    errors: list[str] = []
    for key, image_path in image_map.items():
        decoded = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if decoded is None or decoded.size == 0:
            errors.append(f"{image_path}: ảnh hỏng hoặc không giải mã được")
        label_path = label_map.get(key)
        if label_path is None:
            continue
        lines = [
            line.strip()
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if lines:
            labeled_images += 1
        if len(lines) != len(set(lines)):
            errors.append(f"{label_path}: có annotation trùng hoàn toàn")
        for line_number, line in enumerate(lines, start=1):
            try:
                class_id = _validate_box(
                    line.split(), len(class_names), f"{label_path}:{line_number}"
                )
            except DatasetValidationError as exc:
                errors.append(str(exc))
                continue
            counts[class_id] += 1
            box_count += 1

    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:30])
        suffix = f"\n... và {len(errors) - 30} lỗi khác" if len(errors) > 30 else ""
        raise DatasetValidationError(f"Split {split} không hợp lệ:\n{preview}{suffix}")
    if split == "train" and box_count == 0:
        raise DatasetValidationError("Split train không có bounding box nào")
    return SplitStats(
        images=len(images),
        labeled_images=labeled_images,
        background_images=len(images) - labeled_images,
        boxes=box_count,
        class_counts=dict(sorted(counts.items())),
    )


def validate_no_split_leakage(dataset_root: Path, splits: Sequence[str]) -> None:
    """Reject byte-identical images that occur in more than one split."""
    hashes: defaultdict[str, list[tuple[str, Path]]] = defaultdict(list)
    for split in splits:
        for image_path in image_files(dataset_root / "images" / split):
            hashes[sha256_file(image_path)].append((split, image_path))
    leaks = [items for items in hashes.values() if len({split for split, _ in items}) > 1]
    if leaks:
        rendered = []
        for items in leaks[:10]:
            rendered.append(", ".join(f"{split}:{path.name}" for split, path in items))
        raise DatasetValidationError(
            "Phát hiện ảnh giống hệt ở nhiều split:\n- " + "\n- ".join(rendered)
        )


def validate_dataset(dataset_root: Path, class_names: Sequence[str]) -> dict[str, SplitStats]:
    """Validate all available splits and return reproducible statistics."""
    dataset_root = dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    stats: dict[str, SplitStats] = {}
    for split in (*REQUIRED_SPLITS, *OPTIONAL_SPLITS):
        stats[split] = validate_split(dataset_root, split, class_names)
    available = [name for name, value in stats.items() if value.images]
    validate_no_split_leakage(dataset_root, available)
    return stats


def stats_as_dict(stats: dict[str, SplitStats]) -> dict[str, dict[str, object]]:
    return {name: asdict(value) for name, value in stats.items()}


def write_dataset_yaml(
    dataset_root: Path,
    class_names: Sequence[str],
    output_path: Path,
    *,
    include_test: bool,
) -> Path:
    """Write an absolute-path Ultralytics dataset YAML for this validated dataset."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('Thiếu PyYAML. Chạy: python -m pip install ".[detection]"') from exc

    payload: dict[str, object] = {
        "path": str(dataset_root.expanduser().resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    if include_test:
        payload["test"] = "images/test"
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return output_path
