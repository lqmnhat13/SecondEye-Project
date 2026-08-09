#!/usr/bin/env python3
"""Prepare a private Open Images review pool for missing indoor v1.1 classes.

The script downloads only official Open Images metadata and selected validation
pixels. It does not assign SecondEye's door state, glass-door type, stair
direction, or column boxes automatically; those candidates require visual
review. All generated pixels and detailed inventories stay under data/local.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import ssl
import urllib.request
from collections import defaultdict
from pathlib import Path

import certifi
import cv2
import numpy as np


CLASS_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv"
BOX_CLASS_URL = (
    "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
)
BOX_URL = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
LABEL_URL = (
    "https://storage.googleapis.com/openimages/v7/"
    "oidv7-val-annotations-human-imagelabels.csv"
)
METADATA_URL = (
    "https://storage.googleapis.com/openimages/2018_04/validation/"
    "validation-images-with-rotation.csv"
)
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg"

MIDS = {
    "person": "/m/01g317",
    "backpack_bag": "/m/01940j",
    "box": "/m/025dyy",
    "trash_bin": "/m/0bjyj5",
    "door_review": "/m/02dgv",
    "stairs_review": "/m/01lynh",
    "column_review": "/m/01_m7",
}
INDOOR_ANCHORS = {
    "/m/01mzpv",  # Chair
    "/m/04bcr3",  # Table
    "/m/02crq1",  # Couch
    "/m/03ssj5",  # Bed
    "/m/01s105",  # Cabinetry
    "/m/02dgv",  # Door
    "/m/01lynh",  # Stairs
}
LIMITS = {
    "person": 80,
    "backpack_bag": 40,
    "box": 80,
    "trash_bin": 40,
    "door_review": 197,
    "stairs_review": 100,
    "column_review": 70,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, default=Path("data/local/public_cache/openimages_v7")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/local/public_review/openimages_v11")
    )
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--min-short-side", type=int, default=480)
    parser.add_argument("--min-blur", type=float, default=25.0)
    return parser.parse_args()


def download(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "SecondEyeResearch/1.0"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=120, context=context) as response:
        with temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
    temporary.replace(destination)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_existing_ids(project_root: Path) -> set[str]:
    manifest = project_root / "data/local/indoor_dataset_v1_1/sample_manifest.csv"
    if not manifest.is_file():
        return set()
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        return {
            row["sample_id"].removeprefix("obs_oi_")
            for row in csv.DictReader(stream)
            if row.get("sample_id", "").startswith("obs_oi_")
        }


def read_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["ImageID"]: row for row in csv.DictReader(stream)}


def read_positive_labels(path: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    relevant = set(MIDS.values()) | INDOOR_ANCHORS
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["Confidence"] == "1" and row["LabelName"] in relevant:
                result[row["LabelName"]].add(row["ImageID"])
    return result


def read_boxes(path: Path) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    relevant = set(MIDS.values()) | INDOOR_ANCHORS
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["LabelName"] not in relevant:
                continue
            if row["IsDepiction"] == "1" or row["IsGroupOf"] == "1":
                continue
            result[row["ImageID"]].append(row)
    return result


def candidate_ids(
    labels: dict[str, set[str]],
    boxes: dict[str, list[dict[str, str]]],
    existing: set[str],
    seed: int,
) -> dict[str, list[str]]:
    rng = random.Random(seed)
    by_box_mid: dict[str, set[str]] = defaultdict(set)
    for image_id, rows in boxes.items():
        for row in rows:
            by_box_mid[row["LabelName"]].add(image_id)
    selected: dict[str, list[str]] = {}
    for group, mid in MIDS.items():
        pool = set(labels.get(mid, set())) | set(by_box_mid.get(mid, set()))
        if group == "person":
            indoor = set().union(*(labels.get(anchor, set()) for anchor in INDOOR_ANCHORS))
            indoor |= {
                image_id
                for image_id, rows in boxes.items()
                if any(row["LabelName"] in INDOOR_ANCHORS for row in rows)
            }
            pool &= indoor
        pool -= existing
        ordered = sorted(pool)
        rng.shuffle(ordered)
        selected[group] = ordered[: LIMITS[group]]
    return selected


def image_quality(path: Path) -> tuple[int, int, float, float] | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return width, height, float(gray.mean()), float(cv2.Laplacian(gray, cv2.CV_64F).var())


def fit_thumbnail(image: np.ndarray, width: int = 300, height: int = 220) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def draw_source_boxes(image: np.ndarray, rows: list[dict[str, str]], target_mid: str) -> None:
    height, width = image.shape[:2]
    for row in rows:
        if row["LabelName"] != target_mid:
            continue
        x1 = int(float(row["XMin"]) * width)
        x2 = int(float(row["XMax"]) * width)
        y1 = int(float(row["YMin"]) * height)
        y2 = int(float(row["YMax"]) * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), max(2, width // 500))


def write_sheets(
    output: Path,
    group: str,
    rows: list[dict[str, str]],
    image_dir: Path,
    boxes: dict[str, list[dict[str, str]]],
) -> None:
    sheet_dir = output / "sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    target_mid = MIDS[group]
    per_page = 20
    for page_start in range(0, len(rows), per_page):
        cells: list[np.ndarray] = []
        for row in rows[page_start : page_start + per_page]:
            image = cv2.imread(str(image_dir / f"{row['image_id']}.jpg"), cv2.IMREAD_COLOR)
            if image is None:
                continue
            draw_source_boxes(image, boxes.get(row["image_id"], []), target_mid)
            cell = fit_thumbnail(image)
            cv2.rectangle(cell, (0, 0), (299, 219), (80, 80, 80), 1)
            cv2.putText(
                cell,
                row["image_id"],
                (6, 213),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cells.append(cell)
        while len(cells) < per_page:
            cells.append(np.full((220, 300, 3), 245, dtype=np.uint8))
        sheet = np.vstack([np.hstack(cells[row : row + 5]) for row in range(0, 20, 5)])
        page = page_start // per_page + 1
        cv2.imwrite(str(sheet_dir / f"{group}_{page:02d}.jpg"), sheet)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    cache = (project_root / args.cache).resolve()
    output = (project_root / args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Review output phải rỗng hoặc chưa tồn tại: {output}")
    output.mkdir(parents=True, exist_ok=True)

    files = {
        "classes_all": (CLASS_URL, cache / "oidv7-class-descriptions.csv"),
        "classes_boxable": (BOX_CLASS_URL, cache / "oidv7-class-descriptions-boxable.csv"),
        "boxes": (BOX_URL, cache / "validation-annotations-bbox.csv"),
        "labels": (LABEL_URL, cache / "oidv7-val-annotations-human-imagelabels.csv"),
        "metadata": (METADATA_URL, cache / "validation-images-with-rotation.csv"),
    }
    for url, path in files.values():
        download(url, path)
    labels = read_positive_labels(files["labels"][1])
    boxes = read_boxes(files["boxes"][1])
    metadata = read_metadata(files["metadata"][1])
    groups = candidate_ids(labels, boxes, read_existing_ids(project_root), args.seed)

    image_dir = output / "images"
    image_dir.mkdir(parents=True)
    inventory: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    downloaded: set[str] = set()
    for group, image_ids in groups.items():
        accepted_for_group: list[dict[str, str]] = []
        for image_id in image_ids:
            source = metadata.get(image_id)
            if not source:
                rejected.append({"group": group, "image_id": image_id, "reason": "missing_metadata"})
                continue
            license_url = source.get("License", "").strip()
            if license_url != "https://creativecommons.org/licenses/by/2.0/":
                rejected.append({"group": group, "image_id": image_id, "reason": f"license:{license_url}"})
                continue
            image_path = image_dir / f"{image_id}.jpg"
            if image_id not in downloaded:
                try:
                    download(IMAGE_URL.format(image_id=image_id), image_path)
                except Exception as exc:
                    rejected.append({"group": group, "image_id": image_id, "reason": f"download:{exc}"})
                    continue
                downloaded.add(image_id)
            quality = image_quality(image_path)
            if quality is None:
                rejected.append({"group": group, "image_id": image_id, "reason": "decode"})
                continue
            width, height, brightness, blur = quality
            reason = ""
            if min(width, height) < args.min_short_side:
                reason = "short_side"
            elif not 20 <= brightness <= 235:
                reason = "brightness"
            elif blur < args.min_blur:
                reason = "blur"
            if reason:
                rejected.append({"group": group, "image_id": image_id, "reason": reason})
                continue
            row = {
                "group": group,
                "image_id": image_id,
                "source_mid": MIDS[group],
                "width": str(width),
                "height": str(height),
                "brightness": f"{brightness:.3f}",
                "blur_variance": f"{blur:.3f}",
                "source_box_count": str(
                    sum(item["LabelName"] == MIDS[group] for item in boxes.get(image_id, []))
                ),
                "license": license_url,
                "author": source.get("Author", "").strip(),
                "landing_url": source.get("OriginalLandingURL", "").strip(),
                "sha256": sha256_file(image_path),
                "review_status": "pending",
                "review_notes": "",
            }
            inventory.append(row)
            accepted_for_group.append(row)
        write_sheets(output, group, accepted_for_group, image_dir, boxes)

    fields = list(inventory[0])
    with (output / "candidate_inventory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(inventory)
    (output / "rejected.json").write_text(
        json.dumps(rejected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "source_license.json").write_text(
        json.dumps(
            {
                "dataset": "Open Images V7",
                "official_page": "https://storage.googleapis.com/openimages/web/download_v7.html",
                "image_license_required": "https://creativecommons.org/licenses/by/2.0/",
                "annotation_license": "https://creativecommons.org/licenses/by/4.0/",
                "candidate_rows": len(inventory),
                "unique_images": len({row["image_id"] for row in inventory}),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Review pool: {output}")
    print(f"Rows={len(inventory)} unique_images={len({row['image_id'] for row in inventory})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
