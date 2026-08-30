from __future__ import annotations

from pathlib import Path

import pytest

from secondeye.data.protocol import (
    REQUIRED_COLUMNS,
    audit_manifest,
    load_allowed_values,
)


def valid_row(**overrides: str) -> dict[str, str]:
    row = {
        "sample_id": "obs_000001",
        "asset_relpath": "data/raw/obstacle/obs_000001.jpg",
        "task": "obstacle",
        "group_id": "grp_rooma_ses01_scene01",
        "split": "development",
        "capture_session_id": "ses_01",
        "scene_id": "scn_rooma_01",
        "video_id": "",
        "frame_index": "",
        "source_type": "self_collected",
        "source_origin": "SecondEye controlled indoor collection",
        "license": "internal_research_only",
        "consent_status": "not_applicable",
        "contains_personal_data": "false",
        "capture_device": "MacBook FaceTime HD",
        "width_px": "1280",
        "height_px": "720",
        "sha256": "a" * 64,
        "annotation_status": "accepted",
        "notes": "",
    }
    row.update(overrides)
    return row


def issue_codes(rows: list[dict[str, str]]) -> set[str]:
    return {issue.code for issue in audit_manifest(sorted(REQUIRED_COLUMNS), rows)}


def test_valid_manifest_has_no_issues():
    assert audit_manifest(sorted(REQUIRED_COLUMNS), [valid_row()]) == []


def test_group_and_scene_cannot_cross_development_and_test():
    rows = [
        valid_row(),
        valid_row(
            sample_id="ocr_000002",
            task="ocr",
            split="test",
            sha256="b" * 64,
            asset_relpath="data/raw/ocr/ocr_000002.jpg",
        ),
    ]
    assert {"GROUP_SPLIT_LEAKAGE", "SCENE_SPLIT_LEAKAGE"} <= issue_codes(rows)


def test_same_video_cannot_be_assigned_to_two_groups():
    rows = [
        valid_row(video_id="vid_01", frame_index="0"),
        valid_row(
            sample_id="obs_000002",
            group_id="grp_rooma_ses01_scene02",
            scene_id="scn_rooma_02",
            video_id="vid_01",
            frame_index="30",
            sha256="b" * 64,
            asset_relpath="data/raw/obstacle/obs_000002.jpg",
        ),
    ]
    assert "VIDEO_GROUP_LEAKAGE" in issue_codes(rows)


def test_identical_asset_may_not_cross_groups():
    rows = [
        valid_row(),
        valid_row(
            sample_id="vqa_000002",
            task="vqa",
            group_id="grp_rooma_ses01_scene02",
            scene_id="scn_rooma_02",
            asset_relpath="data/raw/vqa/vqa_000002.jpg",
        ),
    ]
    assert "HASH_GROUP_LEAKAGE" in issue_codes(rows)


@pytest.mark.parametrize("consent_status", ["pending", "withdrawn", "not_applicable"])
def test_participant_data_requires_approved_consent(consent_status: str):
    codes = issue_codes(
        [
            valid_row(
                source_type="participant",
                consent_status=consent_status,
                contains_personal_data="true",
            )
        ]
    )
    assert "PARTICIPANT_WITHOUT_APPROVAL" in codes


def test_pending_consent_requires_quarantine():
    assert "CONSENT_REQUIRES_QUARANTINE" in issue_codes(
        [valid_row(consent_status="pending")]
    )


def test_nonparticipant_personal_data_requires_manual_review():
    issues = audit_manifest(
        sorted(REQUIRED_COLUMNS),
        [valid_row(contains_personal_data="true", consent_status="not_applicable")],
    )
    assert any(
        issue.code == "PERSONAL_DATA_BASIS_REVIEW" and issue.severity == "warning"
        for issue in issues
    )


def test_unsafe_asset_paths_are_rejected():
    assert "UNSAFE_ASSET_PATH" in issue_codes(
        [valid_row(asset_relpath="../private.jpg")]
    )


def test_config_defines_valid_split_fractions(tmp_path: Path):
    path = tmp_path / "data_protocol.toml"
    path.write_text(
        '[split]\nallowed = ["development", "test", "quarantine"]\n'
        "development_fraction = 0.75\ntest_fraction = 0.25\n",
        encoding="utf-8",
    )
    allowed = load_allowed_values(path)
    assert allowed["split"] == {"development", "test", "quarantine"}
