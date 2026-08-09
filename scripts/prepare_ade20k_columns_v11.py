#!/usr/bin/env python3
"""Range-download ADE20K validation samples containing the column class.

Only selected images and masks are extracted from MIT's official Scene Parsing
archive. ADE20K class 43 is ``column, pillar``. Semantic components are converted
to candidate YOLO boxes, but still require visual review before dataset import.
"""

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


ARCHIVE_URL = "https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
ARCHIVE_BYTES = 967_382_037
ARCHIVE_ETAG = "39a91415-5e46c686a7a37"
COLUMN_CLASS_ID = 43


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/local/public_review/ade20k_columns_v11")
    )
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--min-component-area", type=float, default=0.0015)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def column_boxes(mask: np.ndarray, min_area: float) -> list[tuple[float, float, float, float]]:
    binary = (mask == COLUMN_CLASS_ID).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = mask.shape[:2]
    boxes: list[tuple[float, float, float, float]] = []
    for component in range(1, count):
        x, y, box_width, box_height, area = stats[component]
        if area / (width * height) < min_area:
            continue
        if box_width < 5 or box_height < 20:
            continue
        boxes.append(
            (
                (x + box_width / 2) / width,
                (y + box_height / 2) / height,
                box_width / width,
                box_height / height,
            )
        )
    return boxes


def draw_boxes(image: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> None:
    height, width = image.shape[:2]
    for center_x, center_y, box_width, box_height in boxes:
        x1 = int((center_x - box_width / 2) * width)
        x2 = int((center_x + box_width / 2) * width)
        y1 = int((center_y - box_height / 2) * height)
        y2 = int((center_y + box_height / 2) * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), max(2, width // 500))


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
    output = (project_root / args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output phải rỗng hoặc chưa tồn tại: {output}")
    image_dir = output / "images"
    mask_dir = output / "masks"
    sheet_dir = output / "sheets"
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
            boxes = column_boxes(mask, args.min_component_area)
            if not boxes:
                continue
            stem = Path(mask_name).stem
            image_name = f"ADEChallengeData2016/images/validation/{stem}.jpg"
            image_bytes = archive.read(image_name)
            image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            image_path = image_dir / f"{stem}.jpg"
            mask_path = mask_dir / f"{stem}.png"
            image_path.write_bytes(image_bytes)
            mask_path.write_bytes(mask_bytes)
            height, width = image.shape[:2]
            rows.append(
                {
                    "source_id": stem,
                    "source_image_path": image_name,
                    "source_mask_path": mask_name,
                    "width": str(width),
                    "height": str(height),
                    "column_boxes_yolo": json.dumps(boxes, separators=(",", ":")),
                    "column_box_count": str(len(boxes)),
                    "image_sha256": sha256_bytes(image_bytes),
                    "mask_sha256": sha256_bytes(mask_bytes),
                    "review_status": "pending",
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
        raise RuntimeError(f"Chỉ tìm được {len(rows)}/{args.count} ảnh column")
    with (output / "candidate_inventory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for page_start in range(0, len(cells), 20):
        page_cells = cells[page_start : page_start + 20]
        while len(page_cells) < 20:
            page_cells.append(np.full((220, 300, 3), 245, dtype=np.uint8))
        sheet = np.vstack(
            [np.hstack(page_cells[index : index + 5]) for index in range(0, 20, 5)]
        )
        cv2.imwrite(str(sheet_dir / f"column_{page_start // 20 + 1:02d}.jpg"), sheet)
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
                "column_source_class_id": COLUMN_CLASS_ID,
                "validation_masks_checked": checked,
                "accepted_candidates": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"ADE20K column review pool: {output}")
    print(f"checked={checked} accepted={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
