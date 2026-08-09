#!/usr/bin/env python3
"""Range-extract ADE20K validation stair candidates for direction review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
from remotezip import RemoteZip

from prepare_ade20k_columns_v11 import (
    ARCHIVE_BYTES,
    ARCHIVE_ETAG,
    ARCHIVE_URL,
    draw_boxes,
    thumbnail,
)


# Official ADEChallengeData2016/objectInfo150.txt:
# 54 stairs/steps, 60 stairway/staircase, 97 escalator, 122 step/stair.
STAIR_CLASS_IDS = {54, 60, 97, 122}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/local/public_review/ade20k_stairs_v11"),
    )
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--min-component-area", type=float, default=0.002)
    return parser.parse_args()


def stair_boxes(mask: np.ndarray, min_area: float) -> list[tuple[float, float, float, float]]:
    binary = np.isin(mask, list(STAIR_CLASS_IDS)).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = mask.shape[:2]
    result = []
    for component in range(1, count):
        x, y, box_width, box_height, area = stats[component]
        if area / (width * height) < min_area or box_width < 12 or box_height < 12:
            continue
        result.append(
            (
                (x + box_width / 2) / width,
                (y + box_height / 2) / height,
                box_width / width,
                box_height / height,
            )
        )
    return result


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    output = (project_root / args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output phải rỗng: {output}")
    image_dir, mask_dir, sheet_dir = output / "images", output / "masks", output / "sheets"
    for directory in (image_dir, mask_dir, sheet_dir):
        directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    cells: list[np.ndarray] = []
    checked = 0
    with RemoteZip(ARCHIVE_URL) as archive:
        masks = sorted(
            name
            for name in archive.namelist()
            if name.startswith("ADEChallengeData2016/annotations/validation/")
            and name.endswith(".png")
        )
        random.Random(args.seed).shuffle(masks)
        for mask_name in masks:
            checked += 1
            mask_bytes = archive.read(mask_name)
            mask = cv2.imdecode(np.frombuffer(mask_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            boxes = stair_boxes(mask, args.min_component_area)
            if not boxes:
                continue
            stem = Path(mask_name).stem
            image_name = f"ADEChallengeData2016/images/validation/{stem}.jpg"
            image_bytes = archive.read(image_name)
            image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            (image_dir / f"{stem}.jpg").write_bytes(image_bytes)
            (mask_dir / f"{stem}.png").write_bytes(mask_bytes)
            height, width = image.shape[:2]
            rows.append(
                {
                    "source_id": stem,
                    "source_image_path": image_name,
                    "source_mask_path": mask_name,
                    "width": str(width),
                    "height": str(height),
                    "stairs_boxes_yolo": json.dumps(boxes, separators=(",", ":")),
                    "stairs_box_count": str(len(boxes)),
                    "image_sha256": digest(image_bytes),
                    "mask_sha256": digest(mask_bytes),
                    "review_status": "pending",
                    "secondeye_class": "",
                    "review_notes": "",
                }
            )
            annotated = image.copy()
            draw_boxes(annotated, boxes)
            cell = thumbnail(annotated)
            cv2.putText(
                cell,
                stem,
                (6, 213),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cells.append(cell)
            if len(rows) >= args.count:
                break
    if len(rows) < args.count:
        raise RuntimeError(f"Chỉ tìm được {len(rows)}/{args.count} ảnh stairs")
    with (output / "candidate_inventory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for start in range(0, len(cells), 20):
        page_cells = cells[start : start + 20]
        while len(page_cells) < 20:
            page_cells.append(np.full((220, 300, 3), 245, dtype=np.uint8))
        sheet = np.vstack(
            [np.hstack(page_cells[index : index + 5]) for index in range(0, 20, 5)]
        )
        cv2.imwrite(str(sheet_dir / f"stairs_{start // 20 + 1:02d}.jpg"), sheet)
    (output / "source_license.json").write_text(
        json.dumps(
            {
                "dataset": "ADE20K Scene Parsing Benchmark",
                "official_page": "https://ade20k.csail.mit.edu/",
                "terms": "https://ade20k.csail.mit.edu/terms",
                "archive": ARCHIVE_URL,
                "archive_bytes": ARCHIVE_BYTES,
                "archive_etag": ARCHIVE_ETAG,
                "images": "non-commercial research and educational use only",
                "annotations_and_software": "BSD-3-Clause",
                "privacy": "official ADE20K release states faces and license plates are blurred",
                "source_class_ids": sorted(STAIR_CLASS_IDS),
                "validation_masks_checked": checked,
                "accepted_candidates": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"ADE20K stairs review pool: {output}")
    print(f"checked={checked} accepted={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
