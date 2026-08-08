"""Audit the SecondEye sample manifest before training or evaluation.

The validator deliberately does not assign rows to splits. Split assignment is a
group-level research decision that must consider task and scenario balance.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence


REQUIRED_COLUMNS = {
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
}
DEFAULT_ALLOWED = {
    "task": {"obstacle", "ocr", "vqa"},
    "split": {"development", "test", "quarantine"},
    "source_type": {"self_collected", "public_dataset", "synthetic", "participant"},
    "consent_status": {"not_applicable", "pending", "approved", "withdrawn"},
    "annotation_status": {"pending", "in_review", "accepted", "rejected"},
}
ID_PREFIX = {"obstacle": "obs_", "ocr": "ocr_", "vqa": "vqa_"}
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    code: str
    message: str
    row: int | None = None

    def __str__(self) -> str:
        location = f" row={self.row}" if self.row is not None else ""
        return f"{self.severity.upper()} {self.code}{location}: {self.message}"


def load_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Load a UTF-8 CSV manifest and preserve its declared columns."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def load_allowed_values(config_path: Path | None) -> dict[str, set[str]]:
    """Load controlled vocabularies while retaining safe defaults."""
    allowed = {key: set(values) for key, values in DEFAULT_ALLOWED.items()}
    if config_path is None:
        return allowed
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    vocab = config.get("vocabulary", {})
    split = config.get("split", {})
    allowed["task"] = set(vocab.get("tasks", allowed["task"]))
    allowed["source_type"] = set(vocab.get("source_types", allowed["source_type"]))
    allowed["consent_status"] = set(
        vocab.get("consent_statuses", allowed["consent_status"])
    )
    allowed["annotation_status"] = set(
        vocab.get("annotation_statuses", allowed["annotation_status"])
    )
    allowed["split"] = set(split.get("allowed", allowed["split"]))
    development = float(split.get("development_fraction", 0.75))
    test = float(split.get("test_fraction", 0.25))
    if abs(development + test - 1.0) > 1e-9:
        raise ValueError("development_fraction + test_fraction must equal 1.0")
    return allowed


def _is_safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def _record_mapping(
    mapping: dict[str, set[str]], key: str, value: str
) -> None:
    if key:
        mapping[key].add(value)


def audit_manifest(
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, str]],
    *,
    allowed: Mapping[str, set[str]] | None = None,
    require_rows: bool = False,
) -> list[AuditIssue]:
    """Return all schema, privacy and group-leakage issues in a manifest."""
    issues: list[AuditIssue] = []
    missing = sorted(REQUIRED_COLUMNS - set(fieldnames))
    if missing:
        issues.append(AuditIssue("error", "MISSING_COLUMNS", ", ".join(missing)))
        return issues

    allowed_values = allowed or DEFAULT_ALLOWED
    materialized = list(rows)
    if not materialized:
        severity = "error" if require_rows else "warning"
        issues.append(AuditIssue(severity, "EMPTY_MANIFEST", "manifest contains no samples"))
        return issues

    sample_ids: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    scene_splits: dict[str, set[str]] = defaultdict(set)
    video_splits: dict[str, set[str]] = defaultdict(set)
    video_groups: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    hash_groups: dict[str, set[str]] = defaultdict(set)

    for csv_row, raw in enumerate(materialized, start=2):
        row = {key: (raw.get(key) or "").strip() for key in REQUIRED_COLUMNS}
        sample_id = row["sample_id"]
        task = row["task"]
        split = row["split"]
        group_id = row["group_id"]

        if not sample_id or not SAFE_ID.fullmatch(sample_id):
            issues.append(AuditIssue("error", "INVALID_SAMPLE_ID", sample_id or "blank", csv_row))
        elif sample_id in sample_ids:
            issues.append(AuditIssue("error", "DUPLICATE_SAMPLE_ID", sample_id, csv_row))
        else:
            sample_ids.add(sample_id)

        for field in ("task", "split", "source_type", "consent_status", "annotation_status"):
            if row[field] not in allowed_values[field]:
                issues.append(
                    AuditIssue("error", f"INVALID_{field.upper()}", row[field] or "blank", csv_row)
                )

        if task in ID_PREFIX and not sample_id.startswith(ID_PREFIX[task]):
            issues.append(
                AuditIssue("error", "TASK_ID_MISMATCH", f"{sample_id} does not match {task}", csv_row)
            )

        for field, prefix in (
            ("group_id", "grp_"),
            ("capture_session_id", "ses_"),
            ("scene_id", "scn_"),
        ):
            value = row[field]
            if not value.startswith(prefix) or not SAFE_ID.fullmatch(value):
                issues.append(AuditIssue("error", f"INVALID_{field.upper()}", value or "blank", csv_row))
        video_id = row["video_id"]
        if video_id and (not video_id.startswith("vid_") or not SAFE_ID.fullmatch(video_id)):
            issues.append(AuditIssue("error", "INVALID_VIDEO_ID", video_id, csv_row))

        if not _is_safe_relative_path(row["asset_relpath"]):
            issues.append(
                AuditIssue("error", "UNSAFE_ASSET_PATH", row["asset_relpath"] or "blank", csv_row)
            )
        if not row["source_origin"]:
            issues.append(AuditIssue("error", "MISSING_SOURCE_ORIGIN", "source_origin is required", csv_row))
        if not row["license"]:
            issues.append(AuditIssue("error", "MISSING_LICENSE", "license is required", csv_row))
        if not row["capture_device"]:
            issues.append(
                AuditIssue("error", "MISSING_CAPTURE_DEVICE", "use unknown_source_device if unavailable", csv_row)
            )

        personal = row["contains_personal_data"].lower()
        if personal not in {"true", "false"}:
            issues.append(AuditIssue("error", "INVALID_PERSONAL_DATA_FLAG", personal or "blank", csv_row))
        consent = row["consent_status"]
        source_type = row["source_type"]
        if source_type == "participant" and consent != "approved":
            issues.append(
                AuditIssue("error", "PARTICIPANT_WITHOUT_APPROVAL", f"consent_status={consent}", csv_row)
            )
        if consent in {"pending", "withdrawn"} and split != "quarantine":
            issues.append(
                AuditIssue("error", "CONSENT_REQUIRES_QUARANTINE", f"split={split}", csv_row)
            )
        if personal == "true" and consent not in {"approved", "not_applicable"}:
            issues.append(
                AuditIssue("error", "PERSONAL_DATA_NOT_CLEARED", f"consent_status={consent}", csv_row)
            )
        if personal == "true" and consent == "not_applicable":
            issues.append(
                AuditIssue(
                    "warning",
                    "PERSONAL_DATA_BASIS_REVIEW",
                    "document the permitted basis and access restrictions",
                    csv_row,
                )
            )

        for dimension in ("width_px", "height_px"):
            try:
                if int(row[dimension]) <= 0:
                    raise ValueError
            except ValueError:
                issues.append(AuditIssue("error", f"INVALID_{dimension.upper()}", row[dimension], csv_row))
        if row["frame_index"]:
            try:
                if int(row["frame_index"]) < 0:
                    raise ValueError
            except ValueError:
                issues.append(AuditIssue("error", "INVALID_FRAME_INDEX", row["frame_index"], csv_row))

        sha256 = row["sha256"].lower()
        if not SHA256.fullmatch(sha256):
            issues.append(AuditIssue("error", "INVALID_SHA256", sha256 or "blank", csv_row))

        _record_mapping(group_splits, group_id, split)
        _record_mapping(scene_splits, row["scene_id"], split)
        _record_mapping(video_splits, video_id, split)
        _record_mapping(video_groups, video_id, group_id)
        _record_mapping(hash_splits, sha256, split)
        _record_mapping(hash_groups, sha256, group_id)

    for key, values in group_splits.items():
        if len(values) > 1:
            issues.append(AuditIssue("error", "GROUP_SPLIT_LEAKAGE", f"{key}: {sorted(values)}"))
    for key, values in scene_splits.items():
        if len(values) > 1:
            issues.append(AuditIssue("error", "SCENE_SPLIT_LEAKAGE", f"{key}: {sorted(values)}"))
    for key, values in video_splits.items():
        if len(values) > 1:
            issues.append(AuditIssue("error", "VIDEO_SPLIT_LEAKAGE", f"{key}: {sorted(values)}"))
    for key, values in video_groups.items():
        if len(values) > 1:
            issues.append(AuditIssue("error", "VIDEO_GROUP_LEAKAGE", f"{key}: {sorted(values)}"))
    for key, values in hash_splits.items():
        if len(values) > 1:
            issues.append(AuditIssue("error", "HASH_SPLIT_LEAKAGE", f"{key}: {sorted(values)}"))
    for key, values in hash_groups.items():
        if len(values) > 1:
            issues.append(AuditIssue("error", "HASH_GROUP_LEAKAGE", f"{key}: {sorted(values)}"))
    return issues


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="UTF-8 sample manifest CSV")
    parser.add_argument("--config", type=Path, default=None, help="data protocol TOML")
    parser.add_argument("--require-rows", action="store_true", help="treat an empty manifest as an error")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        fieldnames, rows = load_manifest(args.manifest)
        allowed = load_allowed_values(args.config)
        issues = audit_manifest(fieldnames, rows, allowed=allowed, require_rows=args.require_rows)
    except (OSError, csv.Error, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"ERROR LOAD_FAILED: {exc}", file=sys.stderr)
        return 2

    for issue in issues:
        print(issue)
    errors = sum(issue.severity == "error" for issue in issues)
    warnings = sum(issue.severity == "warning" for issue in issues)
    print(f"AUDIT rows={len(rows)} errors={errors} warnings={warnings}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
