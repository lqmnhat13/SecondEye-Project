#!/usr/bin/env python3
"""Build a private, reproducible 80-image indoor YOLO pilot from Open Images.

The script deliberately keeps pixels, labels, per-image provenance and manifests
under ``data/local`` so Git cannot publish them accidentally. It imports only
source classes with an unambiguous mapping to the locked SecondEye v1 schema.
Door state, glass doors and stair direction are not inferred from generic Open
Images labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import ssl
import tomllib
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import certifi


CLASS_URL = (
    "https://storage.googleapis.com/openimages/v7/"
    "oidv7-class-descriptions-boxable.csv"
)
BOX_URL = (
    "https://storage.googleapis.com/openimages/v5/"
    "validation-annotations-bbox.csv"
)
METADATA_URL = (
    "https://storage.googleapis.com/openimages/2018_04/validation/"
    "validation-images-with-rotation.csv"
)
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg"

# These mappings are semantic equivalents. Generic Door and Stairs are
# intentionally absent because the v1 schema requires state/direction.
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
INDOOR_ANCHORS = {
    "/m/02crq1": "sofa",
    "/m/03ssj5": "bed",
    "/m/01s105": "cabinet",
}
PERSON_MID = "/m/01g317"
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


@dataclass(frozen=True)
class AcceptedImage:
    image_id: str
    anchor: str
    cached_path: Path
    width: int
    height: int
    brightness: float
    blur_variance: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/local/indoor_pilot_v1"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/yolo26_obstacles.toml"),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(".cache/openimages_indoor_pilot"),
    )
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--exclude-file",
        type=Path,
        default=Path("configs/indoor_pilot_exclusions_v1.csv"),
    )
    parser.add_argument("--min-short-side", type=int, default=480)
    parser.add_argument("--min-blur-variance", type=float, default=35.0)
    parser.add_argument(
        "--review-complete",
        action="store_true",
        help="mark the deterministic output as visually reviewed after checking it",
    )
    return parser.parse_args()


def download(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "SecondEyePilot/1.0"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=90, context=context) as response:
        with temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
    temporary.replace(destination)


def load_schema(config_path: Path) -> tuple[str, ...]:
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)
    names = tuple(raw["class_schema"]["names"])
    expected = (
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
    if names != expected:
        raise ValueError(f"Schema config không khớp indoor v1: {names!r}")
    return names


def read_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["ImageID"]: row for row in csv.DictReader(stream)}


def read_exclusions(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            row["image_id"].strip(): row["reason"].strip()
            for row in csv.DictReader(stream)
            if row.get("image_id", "").strip()
        }


def read_boxes(
    path: Path,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[str]], set[str]]:
    boxes: dict[str, list[dict[str, str]]] = defaultdict(list)
    candidates: dict[str, list[str]] = defaultdict(list)
    person_images: set[str] = set()
    seen_candidate: set[tuple[str, str]] = set()
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            image_id = row["ImageID"]
            mid = row["LabelName"]
            if mid == PERSON_MID and row["IsDepiction"] != "1":
                person_images.add(image_id)
            if mid in SOURCE_TO_TARGET and row["IsDepiction"] != "1" and row["IsGroupOf"] != "1":
                boxes[image_id].append(row)
            if mid not in INDOOR_ANCHORS:
                continue
            if row["IsDepiction"] == "1" or row["IsGroupOf"] == "1":
                continue
            if row["IsTruncated"] == "1":
                continue
            area = (float(row["XMax"]) - float(row["XMin"])) * (
                float(row["YMax"]) - float(row["YMin"])
            )
            if area < 0.03:
                continue
            key = (mid, image_id)
            if key not in seen_candidate:
                candidates[mid].append(image_id)
                seen_candidate.add(key)
    for values in candidates.values():
        values.sort()
    return boxes, candidates, person_images


def image_quality(path: Path) -> tuple[int, int, float, float] | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return width, height, brightness, blur_variance


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_images(
    *,
    count: int,
    seed: int,
    candidates: dict[str, list[str]],
    person_images: set[str],
    metadata: dict[str, dict[str, str]],
    cache: Path,
    excluded: dict[str, str],
    min_short_side: int,
    min_blur_variance: float,
) -> tuple[list[AcceptedImage], list[dict[str, object]]]:
    rng = random.Random(seed)
    queues: dict[str, list[str]] = {}
    for mid in INDOOR_ANCHORS:
        queue = [
            image_id
            for image_id in candidates[mid]
            if image_id not in person_images and image_id not in excluded
        ]
        rng.shuffle(queue)
        queues[mid] = queue
    positions = {mid: 0 for mid in queues}
    accepted: list[AcceptedImage] = []
    rejected: list[dict[str, object]] = [
        {"image_id": image_id, "reason": f"manual_review:{reason}"}
        for image_id, reason in sorted(excluded.items())
    ]
    attempted: set[str] = set()
    anchor_mids = tuple(INDOOR_ANCHORS)
    cursor = 0
    image_cache = cache / "images"
    image_cache.mkdir(parents=True, exist_ok=True)

    while len(accepted) < count:
        mid = anchor_mids[cursor % len(anchor_mids)]
        cursor += 1
        queue = queues[mid]
        while positions[mid] < len(queue) and queue[positions[mid]] in attempted:
            positions[mid] += 1
        if positions[mid] >= len(queue):
            if all(positions[key] >= len(queues[key]) for key in queues):
                break
            continue
        image_id = queue[positions[mid]]
        positions[mid] += 1
        attempted.add(image_id)
        if image_id not in metadata:
            rejected.append({"image_id": image_id, "reason": "missing_metadata"})
            continue
        cached_path = image_cache / f"{image_id}.jpg"
        try:
            download(IMAGE_URL.format(image_id=image_id), cached_path)
        except Exception as exc:  # network errors are recorded, not hidden
            rejected.append({"image_id": image_id, "reason": f"download_error:{exc}"})
            continue
        quality = image_quality(cached_path)
        if quality is None:
            rejected.append({"image_id": image_id, "reason": "decode_error"})
            continue
        width, height, brightness, blur_variance = quality
        reason = None
        if min(width, height) < min_short_side:
            reason = "short_side_too_small"
        elif not 20.0 <= brightness <= 235.0:
            reason = "brightness_out_of_range"
        elif blur_variance < min_blur_variance:
            reason = "too_blurry"
        if reason:
            rejected.append(
                {
                    "image_id": image_id,
                    "reason": reason,
                    "width": width,
                    "height": height,
                    "brightness": round(brightness, 3),
                    "blur_variance": round(blur_variance, 3),
                }
            )
            continue
        accepted.append(
            AcceptedImage(
                image_id=image_id,
                anchor=INDOOR_ANCHORS[mid],
                cached_path=cached_path,
                width=width,
                height=height,
                brightness=brightness,
                blur_variance=blur_variance,
            )
        )

    if len(accepted) != count:
        raise RuntimeError(f"Chỉ chọn được {len(accepted)}/{count} ảnh đạt chất lượng")
    return accepted, rejected


def assign_splits(images: list[AcceptedImage], seed: int) -> dict[str, str]:
    rng = random.Random(seed + 1)
    by_anchor: dict[str, list[AcceptedImage]] = defaultdict(list)
    for item in images:
        by_anchor[item.anchor].append(item)
    assignments: dict[str, str] = {}
    desired_val = len(images) // 4
    val_ids: set[str] = set()
    remainders: list[AcceptedImage] = []
    for group in by_anchor.values():
        rng.shuffle(group)
        group_val = len(group) // 4
        val_ids.update(item.image_id for item in group[:group_val])
        remainders.extend(group[group_val:])
    rng.shuffle(remainders)
    for item in remainders:
        if len(val_ids) >= desired_val:
            break
        val_ids.add(item.image_id)
    for item in images:
        assignments[item.image_id] = "val" if item.image_id in val_ids else "train"
    return assignments


def yolo_lines(
    rows: list[dict[str, str]], class_to_id: dict[str, int]
) -> list[str]:
    lines: list[str] = []
    for row in rows:
        target = SOURCE_TO_TARGET[row["LabelName"]]
        if target == "person":
            raise ValueError("Ảnh có person không được phép vào pilot riêng tư")
        xmin, xmax = float(row["XMin"]), float(row["XMax"])
        ymin, ymax = float(row["YMin"]), float(row["YMax"])
        width, height = xmax - xmin, ymax - ymin
        center_x, center_y = xmin + width / 2, ymin + height / 2
        if not (0 <= xmin < xmax <= 1 and 0 <= ymin < ymax <= 1):
            raise ValueError(f"BBox nguồn không hợp lệ: {row}")
        lines.append(
            f"{class_to_id[target]} {center_x:.8f} {center_y:.8f} "
            f"{width:.8f} {height:.8f}"
        )
    return sorted(set(lines))


def build_dataset(
    *,
    output: Path,
    schema: tuple[str, ...],
    accepted: list[AcceptedImage],
    assignments: dict[str, str],
    boxes: dict[str, list[dict[str, str]]],
    metadata: dict[str, dict[str, str]],
    rejected: list[dict[str, object]],
    seed: int,
    review_complete: bool,
) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Thư mục đích phải rỗng hoặc chưa tồn tại: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    class_to_id = {name: index for index, name in enumerate(schema)}
    manifest_rows: list[dict[str, str]] = []
    inventory_rows: list[dict[str, str]] = []
    class_counts: dict[str, int] = defaultdict(int)
    anchor_counts: dict[str, int] = defaultdict(int)
    split_counts: dict[str, int] = defaultdict(int)

    for item in accepted:
        split = assignments[item.image_id]
        name = f"oi_{item.image_id}"
        destination = output / "images" / split / f"{name}.jpg"
        shutil.copy2(item.cached_path, destination)
        lines = yolo_lines(boxes[item.image_id], class_to_id)
        if not lines:
            raise ValueError(f"Ảnh không có bbox target: {item.image_id}")
        (output / "labels" / split / f"{name}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        for line in lines:
            class_counts[schema[int(line.split()[0])]] += 1
        anchor_counts[item.anchor] += 1
        split_counts[split] += 1
        source = metadata[item.image_id]
        license_url = source.get("License", "").strip()
        original_url = source.get("OriginalURL", "").strip()
        landing_url = source.get("OriginalLandingURL", "").strip()
        manifest_rows.append(
            {
                "sample_id": f"obs_oi_{item.image_id}",
                "asset_relpath": f"images/{split}/{name}.jpg",
                "task": "obstacle",
                "group_id": f"grp_oi_{item.image_id}",
                "split": "test" if split == "val" else "development",
                "capture_session_id": "ses_openimages_v7_validation",
                "scene_id": f"scn_oi_{item.image_id}",
                "video_id": "",
                "frame_index": "",
                "source_type": "public_dataset",
                "source_origin": landing_url or original_url or IMAGE_URL.format(image_id=item.image_id),
                "license": license_url or "listed_CC_BY_2.0_review_required",
                "consent_status": "not_applicable",
                "contains_personal_data": "false",
                "capture_device": "unknown_source_device",
                "width_px": str(item.width),
                "height_px": str(item.height),
                "sha256": sha256_file(destination),
                "annotation_status": "accepted",
                "notes": (
                    "Open Images verified bbox; no Person bbox; "
                    + (
                        "privacy/contact-sheet review completed 2026-08-09"
                        if review_complete
                        else "manual privacy/contact-sheet review required before use"
                    )
                ),
            }
        )
        inventory_rows.append(
            {
                "image_id": item.image_id,
                "split": split,
                "anchor": item.anchor,
                "width": str(item.width),
                "height": str(item.height),
                "brightness": f"{item.brightness:.3f}",
                "blur_variance": f"{item.blur_variance:.3f}",
                "box_count": str(len(lines)),
                "license": license_url,
                "author": source.get("Author", "").strip(),
                "title": source.get("Title", "").strip(),
                "original_url": original_url,
                "landing_url": landing_url,
            }
        )

    with (output / "sample_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    inventory_fields = list(inventory_rows[0])
    with (output / "source_inventory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=inventory_fields)
        writer.writeheader()
        writer.writerows(inventory_rows)
    yaml_lines = [
        f"path: {output.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
        *(f"  {index}: {name}" for index, name in enumerate(schema)),
    ]
    (output / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    report = {
        "schema_version": "1.0.0",
        "created_at": "2026-08-09",
        "source": "Open Images V7 validation pixels + V5 validation boxes",
        "seed": seed,
        "accepted_images": len(accepted),
        "split_counts": dict(sorted(split_counts.items())),
        "anchor_counts": dict(sorted(anchor_counts.items())),
        "box_counts": dict(sorted(class_counts.items())),
        "uncovered_classes": [name for name in schema if class_counts.get(name, 0) == 0],
        "rejected_candidates": rejected,
        "privacy": {
            "person_box_filter": True,
            "manual_contact_sheet_review": (
                "completed_2026-08-09" if review_complete else "required_before_use"
            ),
            "manual_exclusion_count": len(
                [
                    item
                    for item in rejected
                    if str(item["reason"]).startswith("manual_review:")
                ]
            ),
            "repository_policy": "dataset remains under ignored data/local",
        },
    }
    (output / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if args.count < 80:
        raise ValueError("Pilot phải có tối thiểu 80 ảnh")
    project_root = Path(__file__).resolve().parent.parent
    output = (project_root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    config = (project_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    cache = (project_root / args.cache).resolve() if not args.cache.is_absolute() else args.cache.resolve()
    exclude_file = (
        (project_root / args.exclude_file).resolve()
        if not args.exclude_file.is_absolute()
        else args.exclude_file.resolve()
    )
    schema = load_schema(config)
    class_path = cache / "oidv7-class-descriptions-boxable.csv"
    box_path = cache / "validation-annotations-bbox.csv"
    metadata_path = cache / "validation-images-with-rotation.csv"
    download(CLASS_URL, class_path)
    download(BOX_URL, box_path)
    download(METADATA_URL, metadata_path)
    # Reading the class file ensures the downloaded metadata is not an HTML error page.
    class_text = class_path.read_text(encoding="utf-8")
    for source_name in ("Chair", "Table", "Couch", "Bed", "Cabinetry"):
        if f",{source_name}" not in class_text:
            raise ValueError(f"Thiếu lớp Open Images: {source_name}")
    metadata = read_metadata(metadata_path)
    boxes, candidates, person_images = read_boxes(box_path)
    accepted, rejected = select_images(
        count=args.count,
        seed=args.seed,
        candidates=candidates,
        person_images=person_images,
        metadata=metadata,
        cache=cache,
        excluded=read_exclusions(exclude_file),
        min_short_side=args.min_short_side,
        min_blur_variance=args.min_blur_variance,
    )
    build_dataset(
        output=output,
        schema=schema,
        accepted=accepted,
        assignments=assign_splits(accepted, args.seed),
        boxes=boxes,
        metadata=metadata,
        rejected=rejected,
        seed=args.seed,
        review_complete=args.review_complete,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
