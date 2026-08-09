#!/usr/bin/env python3
"""Create box-level review sheets for ambiguous Open Images classes.

The preparation step creates image-level sheets. This companion script crops
each generic Door/Stairs annotation with context and assigns a stable box key,
so a reviewer can map it to the narrower SecondEye schema without guessing
which source box was inspected. Generated review artifacts remain in
``data/local``.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


BOX_URL_CACHE = Path("data/local/public_cache/openimages_v7/validation-annotations-bbox.csv")
GROUP_TO_MID = {
    "door_review": "/m/02dgv",
    "stairs_review": "/m/01lynh",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-root",
        type=Path,
        default=Path("data/local/public_review/openimages_v11"),
    )
    parser.add_argument("--boxes", type=Path, default=BOX_URL_CACHE)
    return parser.parse_args()


def thumbnail(image: np.ndarray, width: int = 300, height: int = 220) -> np.ndarray:
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


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    review_root = (project_root / args.review_root).resolve()
    box_path = (project_root / args.boxes).resolve()
    inventory_path = review_root / "candidate_inventory.csv"
    if not inventory_path.is_file():
        raise FileNotFoundError(f"Thiếu inventory: {inventory_path}")

    with inventory_path.open(encoding="utf-8", newline="") as stream:
        inventory = list(csv.DictReader(stream))
    ids_by_group = {
        group: {row["image_id"] for row in inventory if row["group"] == group}
        for group in GROUP_TO_MID
    }
    rows_by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    with box_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            for group, mid in GROUP_TO_MID.items():
                if row["ImageID"] not in ids_by_group[group] or row["LabelName"] != mid:
                    continue
                if row["IsDepiction"] == "1" or row["IsGroupOf"] == "1":
                    continue
                rows_by_group[group].append(row)

    output_dir = review_root / "bbox_review"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output bbox review phải rỗng: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    template_rows: list[dict[str, str]] = []

    for group, source_rows in rows_by_group.items():
        source_rows.sort(
            key=lambda row: (
                row["ImageID"],
                float(row["YMin"]),
                float(row["XMin"]),
            )
        )
        per_image_index: dict[str, int] = defaultdict(int)
        cells: list[np.ndarray] = []
        for source in source_rows:
            image_id = source["ImageID"]
            per_image_index[image_id] += 1
            box_index = per_image_index[image_id]
            box_key = f"{image_id}_b{box_index:02d}"
            image = cv2.imread(str(review_root / "images" / f"{image_id}.jpg"))
            if image is None:
                continue
            height, width = image.shape[:2]
            x1 = int(float(source["XMin"]) * width)
            x2 = int(float(source["XMax"]) * width)
            y1 = int(float(source["YMin"]) * height)
            y2 = int(float(source["YMax"]) * height)
            pad_x = max(20, int((x2 - x1) * 0.35))
            pad_y = max(20, int((y2 - y1) * 0.25))
            crop_x1, crop_x2 = max(0, x1 - pad_x), min(width, x2 + pad_x)
            crop_y1, crop_y2 = max(0, y1 - pad_y), min(height, y2 + pad_y)
            crop = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            cv2.rectangle(
                crop,
                (x1 - crop_x1, y1 - crop_y1),
                (x2 - crop_x1, y2 - crop_y1),
                (0, 255, 255),
                max(2, width // 500),
            )
            cell = thumbnail(crop)
            cv2.rectangle(cell, (0, 0), (299, 219), (80, 80, 80), 1)
            cv2.putText(
                cell,
                box_key,
                (5, 213),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cells.append(cell)
            template_rows.append(
                {
                    "group": group,
                    "box_key": box_key,
                    "image_id": image_id,
                    "source_box_index": str(box_index),
                    "xmin": source["XMin"],
                    "ymin": source["YMin"],
                    "xmax": source["XMax"],
                    "ymax": source["YMax"],
                    "decision": "pending",
                    "secondeye_class": "",
                    "review_notes": "",
                }
            )
        for start in range(0, len(cells), 20):
            page_cells = cells[start : start + 20]
            while len(page_cells) < 20:
                page_cells.append(np.full((220, 300, 3), 245, dtype=np.uint8))
            sheet = np.vstack(
                [np.hstack(page_cells[index : index + 5]) for index in range(0, 20, 5)]
            )
            page = start // 20 + 1
            cv2.imwrite(str(output_dir / f"{group}_{page:02d}.jpg"), sheet)

    fields = list(template_rows[0])
    with (output_dir / "bbox_review.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(template_rows)
    print(f"BBox review: {output_dir}")
    print(f"boxes={len(template_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
