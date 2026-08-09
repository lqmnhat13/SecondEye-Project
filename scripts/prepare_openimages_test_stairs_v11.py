#!/usr/bin/env python3
"""Prepare an official Open Images test-split stair pool for direction review."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from prepare_openimages_public_v11 import (
    MIDS,
    download,
    image_quality,
    read_existing_ids,
    read_metadata,
    sha256_file,
)


BOX_URL = "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv"
METADATA_URL = (
    "https://storage.googleapis.com/openimages/2018_04/test/"
    "test-images-with-rotation.csv"
)
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/test/{image_id}.jpg"
STAIRS_MID = MIDS["stairs_review"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, default=Path("data/local/public_cache/openimages_v7")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/local/public_review/openimages_test_stairs_v11"),
    )
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--min-short-side", type=int, default=480)
    parser.add_argument("--min-blur", type=float, default=25.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    cache = (project_root / args.cache).resolve()
    output = (project_root / args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output phải rỗng: {output}")
    output.mkdir(parents=True, exist_ok=True)
    box_path = cache / "test-annotations-bbox.csv"
    metadata_path = cache / "test-images-with-rotation.csv"
    download(BOX_URL, box_path)
    download(METADATA_URL, metadata_path)
    metadata = read_metadata(metadata_path)

    boxes_by_image: dict[str, list[dict[str, str]]] = {}
    with box_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["LabelName"] != STAIRS_MID:
                continue
            if row["IsDepiction"] == "1" or row["IsGroupOf"] == "1":
                continue
            boxes_by_image.setdefault(row["ImageID"], []).append(row)
    excluded = read_existing_ids(project_root)
    validation_inventory = project_root / "data/local/public_review/openimages_v11/candidate_inventory.csv"
    if validation_inventory.is_file():
        with validation_inventory.open(encoding="utf-8", newline="") as stream:
            excluded |= {row["image_id"] for row in csv.DictReader(stream)}
    image_ids = sorted(set(boxes_by_image) - excluded)
    random.Random(args.seed).shuffle(image_ids)

    image_dir = output / "images"
    image_dir.mkdir()
    inventory = []
    rejected = []
    for image_id in image_ids:
        source = metadata.get(image_id)
        if source is None:
            rejected.append({"image_id": image_id, "reason": "missing_metadata"})
            continue
        license_url = source.get("License", "").strip()
        if license_url != "https://creativecommons.org/licenses/by/2.0/":
            rejected.append({"image_id": image_id, "reason": f"license:{license_url}"})
            continue
        image_path = image_dir / f"{image_id}.jpg"
        try:
            download(IMAGE_URL.format(image_id=image_id), image_path)
        except Exception as exc:
            rejected.append({"image_id": image_id, "reason": f"download:{exc}"})
            continue
        quality = image_quality(image_path)
        if quality is None:
            rejected.append({"image_id": image_id, "reason": "decode"})
            continue
        width, height, brightness, blur = quality
        if min(width, height) < args.min_short_side or not 20 <= brightness <= 235 or blur < args.min_blur:
            rejected.append({"image_id": image_id, "reason": "quality"})
            continue
        inventory.append(
            {
                "group": "stairs_review",
                "image_id": image_id,
                "source_mid": STAIRS_MID,
                "width": str(width),
                "height": str(height),
                "brightness": f"{brightness:.3f}",
                "blur_variance": f"{blur:.3f}",
                "source_box_count": str(len(boxes_by_image[image_id])),
                "license": license_url,
                "author": source.get("Author", "").strip(),
                "landing_url": source.get("OriginalLandingURL", "").strip(),
                "sha256": sha256_file(image_path),
                "review_status": "pending",
                "review_notes": "",
            }
        )
        if len(inventory) >= args.count:
            break
    if not inventory:
        raise RuntimeError("Không có Open Images test stair candidates")
    with (output / "candidate_inventory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    (output / "rejected.json").write_text(
        json.dumps(rejected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "source_license.json").write_text(
        json.dumps(
            {
                "dataset": "Open Images V7 test pixels + V5 test boxes",
                "official_page": "https://storage.googleapis.com/openimages/web/download_v7.html",
                "image_license_required": "https://creativecommons.org/licenses/by/2.0/",
                "annotation_license": "https://creativecommons.org/licenses/by/4.0/",
                "candidate_rows": len(inventory),
                "test_box_metadata_etag": "1cc058a7003b4e73d47642276e9b123b",
                "test_image_metadata_etag": "d832feb775b3cb78077bf8ae350adce5",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Open Images test stairs pool: {output}")
    print(f"accepted={len(inventory)} source_images={len(boxes_by_image)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
