#!/usr/bin/env python3
"""Merge reviewed public candidates into a fresh SecondEye YOLO v1.1 dataset.

This builder never edits the locked pilot or an existing output directory. It
accepts only explicit review decisions, preserves per-image attribution, maps
generic Door/Stairs boxes only when a SecondEye class was assigned, and keeps
all generated dataset artifacts under ``data/local``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2


SCHEMA = (
    "person",
    "chair",
    "table_desk",
    "sofa",
    "bed",
    "cabinet",
    "doorway_open",
    "door_closed",
    "glass_door",
    "stairs_up",
    "stairs_down",
    "backpack_bag",
    "box",
    "trash_bin",
    "column",
)
SOURCE_TO_TARGET = {
    "/m/01g317": "person",
    "/m/01mzpv": "chair",
    "/m/04bcr3": "table_desk",
    "/m/02crq1": "sofa",
    "/m/03ssj5": "bed",
    "/m/01s105": "cabinet",
    "/m/01940j": "backpack_bag",
    "/m/025dyy": "box",
    "/m/0bjyj5": "trash_bin",
}
GROUP_TO_TARGET = {
    "person": "person",
    "backpack_bag": "backpack_bag",
    "box": "box",
    "trash_bin": "trash_bin",
}
SAFE_CONTEXT_CLASSES = {"chair", "table_desk", "sofa", "bed", "cabinet"}
MANIFEST_FIELDS = [
    "sample_id",
    "asset_relpath",
    "task",
    "group_id",
    "split",
    "capture_session_id",
    "scene_id",
    "video_id",
    "frame_index",
    "source_type",
    "source_origin",
    "license",
    "consent_status",
    "contains_personal_data",
    "capture_device",
    "width_px",
    "height_px",
    "sha256",
    "annotation_status",
    "notes",
]


@dataclass
class Candidate:
    key: str
    source: str
    source_id: str
    image_path: Path
    width: int
    height: int
    lines: list[str]
    classes: set[str]
    origin: str
    license: str
    notes: str
    author: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("data/local/indoor_dataset_v1_1"))
    parser.add_argument(
        "--openimages-review",
        type=Path,
        default=Path("data/local/public_review/openimages_v11"),
    )
    parser.add_argument(
        "--openimages-image-decisions",
        type=Path,
        default=Path("data/local/public_review/openimages_v11/image_review.csv"),
    )
    parser.add_argument(
        "--openimages-box-decisions",
        type=Path,
        default=Path("data/local/public_review/openimages_v11/bbox_review/bbox_review.csv"),
    )
    parser.add_argument(
        "--openimages-test-stairs-review",
        type=Path,
        default=Path("data/local/public_review/openimages_test_stairs_v11"),
    )
    parser.add_argument(
        "--openimages-test-stairs-decisions",
        type=Path,
        default=Path(
            "data/local/public_review/openimages_test_stairs_v11/bbox_review/bbox_review.csv"
        ),
    )
    parser.add_argument(
        "--ade-review",
        type=Path,
        default=Path("data/local/public_review/ade20k_columns_v11"),
    )
    parser.add_argument(
        "--ade-decisions",
        type=Path,
        default=Path("data/local/public_review/ade20k_columns_v11/column_review.csv"),
    )
    parser.add_argument(
        "--ade-stairs-review",
        type=Path,
        default=Path("data/local/public_review/ade20k_stairs_v11"),
    )
    parser.add_argument(
        "--ade-stairs-decisions",
        type=Path,
        default=Path("data/local/public_review/ade20k_stairs_v11/stairs_review.csv"),
    )
    parser.add_argument(
        "--openimages-boxes",
        type=Path,
        default=Path("data/local/public_cache/openimages_v7/validation-annotations-bbox.csv"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/local/indoor_dataset_v1_1_public_candidate")
    )
    parser.add_argument("--config", type=Path, default=Path("configs/yolo26_obstacles.toml"))
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def absolute(project_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def yolo_line(class_name: str, xmin: float, ymin: float, xmax: float, ymax: float) -> str:
    if not (0 <= xmin < xmax <= 1 and 0 <= ymin < ymax <= 1):
        raise ValueError(f"BBox ngoài biên: {(xmin, ymin, xmax, ymax)}")
    class_id = SCHEMA.index(class_name)
    return (
        f"{class_id} {(xmin + xmax) / 2:.8f} {(ymin + ymax) / 2:.8f} "
        f"{xmax - xmin:.8f} {ymax - ymin:.8f}"
    )


def validate_schema(config_path: Path) -> None:
    with config_path.open("rb") as stream:
        configured = tuple(tomllib.load(stream)["class_schema"]["names"])
    if configured != SCHEMA:
        raise ValueError(f"Schema config không khớp v1: {configured!r}")


def load_openimages_candidates(
    review_root: Path,
    image_decisions_path: Path,
    box_decisions_path: Path,
    source_boxes_path: Path,
) -> list[Candidate]:
    inventory_rows = read_csv(review_root / "candidate_inventory.csv")
    inventory = {(row["group"], row["image_id"]): row for row in inventory_rows}
    image_decisions = read_csv(image_decisions_path)
    accepted_groups: dict[str, set[str]] = defaultdict(set)
    review_notes: dict[str, list[str]] = defaultdict(list)
    for row in image_decisions:
        if row["decision"].strip().lower() != "accepted":
            continue
        key = (row["group"], row["image_id"])
        if key not in inventory:
            raise ValueError(f"Image decision không có trong inventory: {key}")
        accepted_groups[row["image_id"]].add(row["group"])
        if row.get("review_notes", "").strip():
            review_notes[row["image_id"]].append(row["review_notes"].strip())

    accepted_manual_boxes: dict[str, list[tuple[str, float, float, float, float]]] = defaultdict(list)
    for row in read_csv(box_decisions_path):
        if row["decision"].strip().lower() != "accepted":
            continue
        target = row["secondeye_class"].strip()
        allowed = (
            {"doorway_open", "door_closed", "glass_door"}
            if row["group"] == "door_review"
            else {"stairs_up", "stairs_down"}
        )
        if target not in allowed:
            raise ValueError(f"Class review không hợp lệ: {row['box_key']} -> {target}")
        image_id = row["image_id"]
        if (row["group"], image_id) not in inventory:
            raise ValueError(f"BBox decision không có image inventory: {row['box_key']}")
        accepted_groups[image_id].add(row["group"])
        accepted_manual_boxes[image_id].append(
            (
                target,
                float(row["xmin"]),
                float(row["ymin"]),
                float(row["xmax"]),
                float(row["ymax"]),
            )
        )
        if row.get("review_notes", "").strip():
            review_notes[image_id].append(row["review_notes"].strip())

    source_boxes: dict[str, list[dict[str, str]]] = defaultdict(list)
    selected_ids = set(accepted_groups)
    with source_boxes_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["ImageID"] not in selected_ids or row["LabelName"] not in SOURCE_TO_TARGET:
                continue
            if row["IsDepiction"] == "1" or row["IsGroupOf"] == "1":
                continue
            source_boxes[row["ImageID"]].append(row)

    candidates: list[Candidate] = []
    for image_id in sorted(selected_ids):
        groups = accepted_groups[image_id]
        inv = next(inventory[(group, image_id)] for group in sorted(groups))
        lines: set[str] = set()
        classes: set[str] = set()
        direct_targets = {GROUP_TO_TARGET[group] for group in groups if group in GROUP_TO_TARGET}
        for row in source_boxes.get(image_id, []):
            target = SOURCE_TO_TARGET[row["LabelName"]]
            if target not in SAFE_CONTEXT_CLASSES and target not in direct_targets:
                continue
            lines.add(
                yolo_line(
                    target,
                    float(row["XMin"]),
                    float(row["YMin"]),
                    float(row["XMax"]),
                    float(row["YMax"]),
                )
            )
            classes.add(target)
        for target, xmin, ymin, xmax, ymax in accepted_manual_boxes.get(image_id, []):
            lines.add(yolo_line(target, xmin, ymin, xmax, ymax))
            classes.add(target)
        if not lines:
            raise ValueError(f"Ảnh accepted nhưng không có bbox usable: {image_id}")
        image_path = review_root / "images" / f"{image_id}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Ảnh hỏng: {image_path}")
        height, width = image.shape[:2]
        origin = inv.get("landing_url", "").strip() or (
            f"https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg"
        )
        candidates.append(
            Candidate(
                key=f"oi_{image_id}",
                source="openimages",
                source_id=image_id,
                image_path=image_path,
                width=width,
                height=height,
                lines=sorted(lines, key=lambda value: (int(value.split()[0]), value)),
                classes=classes,
                origin=origin,
                license=inv["license"],
                author=inv.get("author", "").strip(),
                notes=(
                    "Open Images V7 CC BY 2.0 pixel; Google verified bbox CC BY 4.0; "
                    f"reviewed groups={','.join(sorted(groups))}; manual relabel recorded"
                    + (f"; {' | '.join(sorted(set(review_notes[image_id])))}" if review_notes[image_id] else "")
                ),
            )
        )
    return candidates


def load_openimages_test_stairs(
    review_root: Path, decisions_path: Path
) -> list[Candidate]:
    inventory = {
        row["image_id"]: row for row in read_csv(review_root / "candidate_inventory.csv")
    }
    accepted: dict[str, list[tuple[str, float, float, float, float]]] = defaultdict(list)
    notes: dict[str, list[str]] = defaultdict(list)
    for row in read_csv(decisions_path):
        if row["decision"].strip().lower() != "accepted":
            continue
        target = row["secondeye_class"].strip()
        if target not in {"stairs_up", "stairs_down"}:
            raise ValueError(f"Open Images test stairs class lỗi: {row['box_key']} -> {target}")
        image_id = row["image_id"]
        if image_id not in inventory:
            raise ValueError(f"Open Images test stairs image thiếu inventory: {image_id}")
        accepted[image_id].append(
            (
                target,
                float(row["xmin"]),
                float(row["ymin"]),
                float(row["xmax"]),
                float(row["ymax"]),
            )
        )
        if row.get("review_notes", "").strip():
            notes[image_id].append(row["review_notes"].strip())
    candidates: list[Candidate] = []
    for image_id, boxes in sorted(accepted.items()):
        inv = inventory[image_id]
        image_path = review_root / "images" / f"{image_id}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Ảnh Open Images test hỏng: {image_path}")
        height, width = image.shape[:2]
        lines = {yolo_line(target, xmin, ymin, xmax, ymax) for target, xmin, ymin, xmax, ymax in boxes}
        classes = {target for target, *_ in boxes}
        candidates.append(
            Candidate(
                key=f"oi_test_{image_id}",
                source="openimages_test",
                source_id=image_id,
                image_path=image_path,
                width=width,
                height=height,
                lines=sorted(lines, key=lambda value: (int(value.split()[0]), value)),
                classes=classes,
                origin=inv.get("landing_url", "").strip()
                or f"https://open-images-dataset.s3.amazonaws.com/test/{image_id}.jpg",
                license=inv["license"],
                author=inv.get("author", "").strip(),
                notes=(
                    "Open Images V7 test CC BY 2.0 pixel; V5 generic Stairs bbox CC BY 4.0; "
                    "manual camera-viewpoint relabel completed"
                    + (f"; {' | '.join(sorted(set(notes[image_id])))}" if notes[image_id] else "")
                ),
            )
        )
    return candidates


def load_ade_candidates(review_root: Path, decisions_path: Path) -> list[Candidate]:
    inventory = {row["source_id"]: row for row in read_csv(review_root / "candidate_inventory.csv")}
    candidates: list[Candidate] = []
    for decision in read_csv(decisions_path):
        if decision["decision"].strip().lower() != "accepted":
            continue
        source_id = decision["source_id"]
        row = inventory.get(source_id)
        if row is None:
            raise ValueError(f"ADE decision không có trong inventory: {source_id}")
        boxes = json.loads(row["column_boxes_yolo"])
        requested = decision.get("accepted_box_indices", "").strip()
        indices = (
            {int(value) for value in requested.replace(",", ";").split(";") if value.strip()}
            if requested
            else set(range(1, len(boxes) + 1))
        )
        if not indices or min(indices) < 1 or max(indices) > len(boxes):
            raise ValueError(f"ADE accepted_box_indices lỗi: {source_id} -> {sorted(indices)}")
        lines = []
        for index, (center_x, center_y, width, height) in enumerate(boxes, start=1):
            if index not in indices:
                continue
            lines.append(f"14 {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}")
        image_path = review_root / "images" / f"{source_id}.jpg"
        candidates.append(
            Candidate(
                key=f"ade_{source_id.lower()}",
                source="ade20k",
                source_id=source_id,
                image_path=image_path,
                width=int(row["width"]),
                height=int(row["height"]),
                lines=lines,
                classes={"column"},
                origin="https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
                license="ADE20K_noncommercial_research_education; annotations_BSD-3-Clause",
                notes=(
                    "ADE20K validation class 43 column/pillar converted from semantic component; "
                    "visual review completed; faces/license plates blurred by official release"
                    + (f"; {decision['review_notes'].strip()}" if decision.get("review_notes", "").strip() else "")
                ),
            )
        )
    return candidates


def load_ade_stairs_candidates(review_root: Path, decisions_path: Path) -> list[Candidate]:
    inventory = {row["source_id"]: row for row in read_csv(review_root / "candidate_inventory.csv")}
    candidates: list[Candidate] = []
    for decision in read_csv(decisions_path):
        if decision["decision"].strip().lower() != "accepted":
            continue
        source_id = decision["source_id"]
        row = inventory.get(source_id)
        if row is None:
            raise ValueError(f"ADE stairs decision không có trong inventory: {source_id}")
        target = decision["secondeye_class"].strip()
        if target not in {"stairs_up", "stairs_down"}:
            raise ValueError(f"ADE stairs class lỗi: {source_id} -> {target}")
        boxes = json.loads(row["stairs_boxes_yolo"])
        requested = decision.get("accepted_box_indices", "").strip()
        indices = (
            {int(value) for value in requested.replace(",", ";").split(";") if value.strip()}
            if requested
            else set(range(1, len(boxes) + 1))
        )
        if not indices or min(indices) < 1 or max(indices) > len(boxes):
            raise ValueError(f"ADE stairs accepted_box_indices lỗi: {source_id}")
        class_id = SCHEMA.index(target)
        lines = [
            f"{class_id} {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}"
            for index, (center_x, center_y, width, height) in enumerate(boxes, start=1)
            if index in indices
        ]
        candidates.append(
            Candidate(
                key=f"ade_stairs_{source_id.lower()}",
                source="ade20k",
                source_id=source_id,
                image_path=review_root / "images" / f"{source_id}.jpg",
                width=int(row["width"]),
                height=int(row["height"]),
                lines=lines,
                classes={target},
                origin="https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
                license="ADE20K_noncommercial_research_education; annotations_BSD-3-Clause",
                notes=(
                    "ADE20K validation stairs semantic component manually mapped by camera viewpoint; "
                    "visual review completed; faces/license plates blurred by official release"
                    + (f"; {decision['review_notes'].strip()}" if decision.get("review_notes", "").strip() else "")
                ),
            )
        )
    return candidates


def merge_same_source(candidates: list[Candidate]) -> list[Candidate]:
    merged: dict[tuple[str, str], Candidate] = {}
    for item in candidates:
        identity = (item.source, item.source_id)
        previous = merged.get(identity)
        if previous is None:
            merged[identity] = item
            continue
        if sha256_file(previous.image_path) != sha256_file(item.image_path):
            raise ValueError(f"Cùng source_id nhưng khác pixel: {identity}")
        previous.lines = sorted(
            set(previous.lines) | set(item.lines),
            key=lambda value: (int(value.split()[0]), value),
        )
        previous.classes |= item.classes
        previous.notes += "; " + item.notes
    return list(merged.values())


def perceptual_hash(path: Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Không đọc được ảnh để tính pHash: {path}")
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype("float32")
    low = cv2.dct(resized)[:8, :8]
    median = float(sorted(low.flatten()[1:])[31])
    value = 0
    for coefficient in low.flatten():
        value = (value << 1) | int(coefficient > median)
    return value


def assign_splits(
    candidates: list[Candidate], seed: int
) -> tuple[dict[str, str], dict[str, str]]:
    parents = list(range(len(candidates)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    hashes = [perceptual_hash(item.image_path) for item in candidates]
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if (hashes[left] ^ hashes[right]).bit_count() <= 4:
                union(left, right)
    clusters: dict[int, list[Candidate]] = defaultdict(list)
    for index, item in enumerate(candidates):
        clusters[find(index)].append(item)

    rng = random.Random(seed)
    cluster_items = list(clusters.values())
    rng.shuffle(cluster_items)
    cluster_split: dict[str, str] = {}
    group_ids: dict[str, str] = {}
    for index, cluster in enumerate(cluster_items):
        split = "val" if index % 4 == 0 else "train"
        cluster_token = hashlib.sha256(
            "|".join(sorted(item.key for item in cluster)).encode("utf-8")
        ).hexdigest()[:16]
        for item in cluster:
            cluster_split[item.key] = split
            group_ids[item.key] = cluster_token
    for class_name in SCHEMA:
        relevant = [item for item in candidates if class_name in item.classes]
        if len(relevant) < 2:
            continue
        if not any(cluster_split[item.key] == "val" for item in relevant):
            token = group_ids[relevant[0].key]
            for item in candidates:
                if group_ids[item.key] == token:
                    cluster_split[item.key] = "val"
        if not any(cluster_split[item.key] == "train" for item in relevant):
            token = group_ids[relevant[-1].key]
            for item in candidates:
                if group_ids[item.key] == token:
                    cluster_split[item.key] = "train"
    return cluster_split, group_ids


def copy_base(base: Path, output: Path) -> list[dict[str, str]]:
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
        for source in sorted((base / "images" / split).glob("*")):
            if source.is_file() and source.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                shutil.copy2(source, output / "images" / split / source.name)
        for source in sorted((base / "labels" / split).glob("*.txt")):
            shutil.copy2(source, output / "labels" / split / source.name)
    return read_csv(base / "sample_manifest.csv")


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    base = absolute(project_root, args.base)
    oi_review = absolute(project_root, args.openimages_review)
    ade_review = absolute(project_root, args.ade_review)
    output = absolute(project_root, args.output)
    if output.exists():
        raise FileExistsError(f"Output phải chưa tồn tại: {output}")
    validate_schema(absolute(project_root, args.config))
    candidates = load_openimages_candidates(
        oi_review,
        absolute(project_root, args.openimages_image_decisions),
        absolute(project_root, args.openimages_box_decisions),
        absolute(project_root, args.openimages_boxes),
    )
    oi_test_decisions = absolute(project_root, args.openimages_test_stairs_decisions)
    if oi_test_decisions.is_file():
        candidates.extend(
            load_openimages_test_stairs(
                absolute(project_root, args.openimages_test_stairs_review),
                oi_test_decisions,
            )
        )
    candidates.extend(
        load_ade_candidates(ade_review, absolute(project_root, args.ade_decisions))
    )
    candidates.extend(
        load_ade_stairs_candidates(
            absolute(project_root, args.ade_stairs_review),
            absolute(project_root, args.ade_stairs_decisions),
        )
    )
    candidates = merge_same_source(candidates)
    if not candidates:
        raise ValueError("Không có public candidate đã accepted")

    output.mkdir(parents=True)
    manifest_rows = copy_base(base, output)
    existing_ids = {row["sample_id"] for row in manifest_rows}
    existing_hashes = {row["sha256"] for row in manifest_rows}
    assignments, perceptual_groups = assign_splits(candidates, args.seed)
    inventory_rows: list[dict[str, str]] = []
    for item in candidates:
        split = assignments[item.key]
        suffix = item.image_path.suffix.lower() or ".jpg"
        filename = f"{item.key}{suffix}"
        destination = output / "images" / split / filename
        shutil.copy2(item.image_path, destination)
        digest = sha256_file(destination)
        if digest in existing_hashes:
            raise ValueError(f"Ảnh trùng byte với dataset base: {item.source_id}")
        existing_hashes.add(digest)
        label_name = f"{item.key}.txt"
        (output / "labels" / split / label_name).write_text(
            "\n".join(item.lines) + "\n", encoding="utf-8"
        )
        sample_id = f"obs_{item.key}"
        if sample_id in existing_ids:
            raise ValueError(f"Trùng sample_id: {sample_id}")
        existing_ids.add(sample_id)
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "asset_relpath": f"images/{split}/{filename}",
                "task": "obstacle",
                "group_id": f"grp_pubscene_{perceptual_groups[item.key]}",
                "split": "test" if split == "val" else "development",
                "capture_session_id": {
                    "openimages": "ses_openimages_v7_validation",
                    "openimages_test": "ses_openimages_v7_test",
                    "ade20k": "ses_ade20k_validation",
                }[item.source],
                "scene_id": f"scn_pubscene_{perceptual_groups[item.key]}",
                "video_id": "",
                "frame_index": "",
                "source_type": "public_dataset",
                "source_origin": item.origin,
                "license": item.license,
                "consent_status": "not_applicable",
                "contains_personal_data": "false",
                "capture_device": "unknown_source_device",
                "width_px": str(item.width),
                "height_px": str(item.height),
                "sha256": digest,
                "annotation_status": "accepted",
                "notes": item.notes,
            }
        )
        inventory_rows.append(
            {
                "sample_id": sample_id,
                "source": item.source,
                "source_id": item.source_id,
                "split": split,
                "classes": ";".join(sorted(item.classes, key=SCHEMA.index)),
                "box_count": str(len(item.lines)),
                "author": item.author,
                "source_origin": item.origin,
                "license": item.license,
                "sha256": digest,
            }
        )

    with (output / "sample_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (output / "public_source_inventory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory_rows[0]))
        writer.writeheader()
        writer.writerows(inventory_rows)

    box_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    for split in ("train", "val"):
        split_counts[split] = len(list((output / "images" / split).glob("*")))
        for label_path in (output / "labels" / split).glob("*.txt"):
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    box_counts[SCHEMA[int(line.split()[0])]] += 1
    yaml_lines = [
        f"path: {output}",
        "train: images/train",
        "val: images/val",
        "names:",
        *(f"  {index}: {name}" for index, name in enumerate(SCHEMA)),
    ]
    (output / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    manifest_hash = sha256_file(output / "sample_manifest.csv")
    quality = {
        "schema_version": "1.1.0-public-candidate",
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "base_dataset": str(base),
        "base_images": len(manifest_rows) - len(candidates),
        "new_public_images": len(candidates),
        "accepted_images": len(manifest_rows),
        "split_counts": dict(split_counts),
        "box_counts": {name: box_counts.get(name, 0) for name in SCHEMA},
        "uncovered_classes": [name for name in SCHEMA if box_counts.get(name, 0) == 0],
        "classes_below_20_boxes": [name for name in SCHEMA if box_counts.get(name, 0) < 20],
        "manifest_sha256": manifest_hash,
        "sources": sorted(
            {
                {
                    "openimages": "Open Images V7 validation",
                    "openimages_test": "Open Images V7 test",
                    "ade20k": "ADE20K validation",
                }[item.source]
                for item in candidates
            }
        ),
        "repository_policy": "pixels_labels_and_detailed_manifest_remain_under_ignored_data/local",
    }
    (output / "quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "dataset_version.json").write_text(
        json.dumps(
            {
                "schema_version": "1.1.0-public-candidate",
                "created_at_utc": quality["created_at_utc"],
                "manifest_sha256": manifest_hash,
                "status": "requires_validator_before_promotion",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Dataset candidate: {output}")
    print(json.dumps(quality, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
