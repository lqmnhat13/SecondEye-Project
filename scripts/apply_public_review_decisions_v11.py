#!/usr/bin/env python3
"""Apply the recorded 2026-08-09 visual-review decisions to local CSV files.

Selections are expressed as contact-sheet page/cell positions so the review can
be audited against the generated sheets. Unlisted candidates are rejected. The
script changes only ignored files under ``data/local``.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


IMAGE_ACCEPTED = {
    "person": {
        1: [2, 12, 16, 18],
        2: [2, 3, 5, 6, 17],
        3: [9, 10, 11, 15],
        4: [2, 3, 4, 11, 12],
    },
    "backpack_bag": {
        1: [1, 3, 6, 9, 11, 12, 14, 15, 16, 17, 20],
        2: [2, 5, 6, 8, 9],
    },
    "box": {
        1: [1, 14, 16, 18],
        2: [1, 5, 6, 11, 12, 15, 18],
        3: [2, 3, 4, 7],
        4: [1, 2, 3, 10, 11, 12, 14],
    },
    "trash_bin": {
        1: [3, 4, 7, 8, 9, 11, 12, 15, 18, 20],
        2: [1, 2, 3, 7, 10],
    },
}

DOOR_DECISIONS = {
    "door_closed": {
        1: [1, 3, 5, 7, 11, 13, 16, 18, 20],
        2: [1, 6, 7, 8, 9, 16, 17, 18, 19, 20],
        3: [2, 3, 11, 12],
        4: [14, 15, 16, 17],
    },
    "glass_door": {
        1: [19],
        2: [4],
        3: [1, 4, 5],
        4: [12, 18],
        5: [9, 12],
        6: [4, 7, 16, 18, 19],
        8: [2, 8, 13],
        9: [4, 5, 6, 12, 14, 15],
        10: [1, 2, 3],
    },
    "doorway_open": {
        1: [6, 8, 9, 15, 17],
        3: [8, 9, 13],
        4: [13],
        5: [13, 15],
        6: [2, 15, 17],
        7: [1, 10, 16],
        8: [11],
        9: [9, 13],
    },
}

STAIRS_DECISIONS = {
    "stairs_up": {
        1: [1, 2, 3, 4, 7, 8, 9, 10, 12, 13, 16, 17, 18, 20],
        2: [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 17],
    },
    "stairs_down": {
        1: [14, 15, 19],
        2: [12, 16],
    },
}

ADE_COLUMN_ACCEPTED = {
    1: [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20],
    2: [2, 3, 4, 5, 6, 7, 9, 10],
}

ADE_STAIRS_DOWN = {1: [4], 2: [3, 17]}

OPENIMAGES_TEST_STAIRS_DOWN = {
    1: [18],
    2: [1, 18],
    4: [2, 4, 10, 19],
    5: [5, 6, 9],
    6: [9],
    7: [7, 9, 10, 15],
}


def keys_from_positions(rows: list[dict[str, str]], positions: dict[int, list[int]]) -> set[str]:
    selected: set[str] = set()
    for page, cells in positions.items():
        for cell in cells:
            index = (page - 1) * 20 + cell - 1
            if not 0 <= index < len(rows):
                raise IndexError(f"Review position ngoài phạm vi: page={page} cell={cell}")
            selected.add(rows[index]["box_key"])
    return selected


def update_image_review(root: Path) -> None:
    path = root / "image_review.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    accepted: set[tuple[str, str]] = set()
    for group, pages in IMAGE_ACCEPTED.items():
        group_rows = grouped[group]
        for page, cells in pages.items():
            for cell in cells:
                index = (page - 1) * 20 + cell - 1
                row = group_rows[index]
                if int(row.get("source_box_count", "0") or 0) == 0:
                    # image_review.csv may not carry this field; inventory check follows below.
                    pass
                accepted.add((group, row["image_id"]))
    inventory = {
        (row["group"], row["image_id"]): row
        for row in csv.DictReader((root / "candidate_inventory.csv").open(encoding="utf-8", newline=""))
    }
    for row in rows:
        key = (row["group"], row["image_id"])
        if key in accepted:
            if int(inventory[key]["source_box_count"]) <= 0:
                raise ValueError(f"Accepted image không có source bbox: {key}")
            row["decision"] = "accepted"
            row["review_notes"] = (
                "visual review 2026-08-09; semantic bbox verified; "
                + (
                    "indoor/distant/masked or face not identifiable; privacy-safe person selection"
                    if row["group"] == "person"
                    else "usable object; domain limitations recorded in quality report"
                )
            )
        else:
            row["decision"] = "rejected"
            row["review_notes"] = "not selected: wrong/ambiguous class, weak domain fit, privacy, or redundant"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def update_box_review(root: Path) -> None:
    path = root / "bbox_review" / "bbox_review.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    decisions: dict[tuple[str, str], str] = {}
    for target, positions in DOOR_DECISIONS.items():
        for key in keys_from_positions(grouped["door_review"], positions):
            decision_key = ("door_review", key)
            if decision_key in decisions:
                raise ValueError(f"Door bbox có hai quyết định: {key}")
            decisions[decision_key] = target
    for target, positions in STAIRS_DECISIONS.items():
        for key in keys_from_positions(grouped["stairs_review"], positions):
            decision_key = ("stairs_review", key)
            if decision_key in decisions:
                raise ValueError(f"Stairs bbox có hai quyết định: {key}")
            decisions[decision_key] = target
    for row in rows:
        target = decisions.get((row["group"], row["box_key"]))
        if target:
            row["decision"] = "accepted"
            row["secondeye_class"] = target
            row["review_notes"] = "visual review 2026-08-09; class follows camera viewpoint/schema v1"
        else:
            row["decision"] = "rejected"
            row["secondeye_class"] = ""
            row["review_notes"] = "wrong object, ambiguous state/direction, weak bbox, or privacy/domain rejection"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_ade_review(project_root: Path) -> None:
    root = project_root / "data/local/public_review/ade20k_columns_v11"
    inventory_path = root / "candidate_inventory.csv"
    with inventory_path.open(encoding="utf-8", newline="") as stream:
        inventory = list(csv.DictReader(stream))
    selected_indices = {
        (page - 1) * 20 + cell - 1
        for page, cells in ADE_COLUMN_ACCEPTED.items()
        for cell in cells
    }
    rows = []
    for index, source in enumerate(inventory):
        accepted = index in selected_indices
        rows.append(
            {
                "source_id": source["source_id"],
                "decision": "accepted" if accepted else "rejected",
                "accepted_box_indices": "" if accepted else "",
                "review_notes": (
                    "visual review 2026-08-09; architectural column/pillar confirmed"
                    if accepted
                    else "rejected: outdoor, wall edge/frame, weak component, or poor domain fit"
                ),
            }
        )
    with (root / "column_review.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_ade_stairs_review(project_root: Path) -> None:
    root = project_root / "data/local/public_review/ade20k_stairs_v11"
    with (root / "candidate_inventory.csv").open(encoding="utf-8", newline="") as stream:
        inventory = list(csv.DictReader(stream))
    selected = {
        (page - 1) * 20 + cell - 1
        for page, cells in ADE_STAIRS_DOWN.items()
        for cell in cells
    }
    rows = []
    for index, source in enumerate(inventory):
        accepted = index in selected
        rows.append(
            {
                "source_id": source["source_id"],
                "decision": "accepted" if accepted else "rejected",
                "secondeye_class": "stairs_down" if accepted else "",
                "accepted_box_indices": "",
                "review_notes": (
                    "visual review 2026-08-09; camera is above the descending flight"
                    if accepted
                    else "rejected: stairs_up, ambiguous direction, weak component, or wrong object"
                ),
            }
        )
    with (root / "stairs_review.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def update_openimages_test_stairs(project_root: Path) -> None:
    path = (
        project_root
        / "data/local/public_review/openimages_test_stairs_v11/bbox_review/bbox_review.csv"
    )
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = keys_from_positions(rows, OPENIMAGES_TEST_STAIRS_DOWN)
    for row in rows:
        if row["box_key"] in selected:
            row["decision"] = "accepted"
            row["secondeye_class"] = "stairs_down"
            row["review_notes"] = (
                "visual review 2026-08-09; camera is above the descending flight; bbox verified"
            )
        else:
            row["decision"] = "rejected"
            row["secondeye_class"] = ""
            row["review_notes"] = "stairs_up, ambiguous direction, wrong object, privacy, or weak bbox"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    root = project_root / "data/local/public_review/openimages_v11"
    update_image_review(root)
    update_box_review(root)
    write_ade_review(project_root)
    write_ade_stairs_review(project_root)
    update_openimages_test_stairs(project_root)
    print("Applied Open Images and ADE20K visual-review decisions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
