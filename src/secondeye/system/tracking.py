"""Small dependency-free tracker for temporal safety evidence.

The tracker intentionally tracks geometry, not just semantic labels.  That lets
an unknown obstacle keep the same identity if YOLO later assigns it a label (or
temporarily misses the label altogether).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _iou(first: list[float], second: list[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


@dataclass(slots=True)
class _Track:
    track_id: int
    bbox_xyxy: list[float]
    timestamp: float
    hits: int = 1
    misses: int = 0
    distance_m: float | None = None
    distance_timestamp: float | None = None
    approach_speed_mps: float | None = None


class DetectionTracker:
    """Greedy IoU tracker with smoothed approach speed and TTC."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.25,
        max_misses: int = 4,
        speed_smoothing: float = 0.5,
        min_approach_speed_mps: float = 0.05,
    ) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold phải nằm trong (0, 1]")
        if max_misses < 0:
            raise ValueError("max_misses không được âm")
        if not 0.0 <= speed_smoothing < 1.0:
            raise ValueError("speed_smoothing phải nằm trong [0, 1)")
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.speed_smoothing = speed_smoothing
        self.min_approach_speed_mps = min_approach_speed_mps
        self._next_id = 1
        self._tracks: dict[int, _Track] = {}

    def update(
        self, detections: list[dict[str, Any]], *, timestamp: float
    ) -> list[dict[str, Any]]:
        """Return copies of detections enriched with stable temporal fields."""
        items = [dict(item) for item in detections]
        candidates: list[tuple[float, int, int]] = []
        for index, item in enumerate(items):
            bbox = [float(value) for value in item["bbox_xyxy"]]
            for track_id, track in self._tracks.items():
                overlap = _iou(bbox, track.bbox_xyxy)
                if overlap >= self.iou_threshold:
                    candidates.append((overlap, index, track_id))
        candidates.sort(reverse=True)
        matched_items: set[int] = set()
        matched_tracks: set[int] = set()
        assignments: dict[int, int] = {}
        for _, index, track_id in candidates:
            if index in matched_items or track_id in matched_tracks:
                continue
            matched_items.add(index)
            matched_tracks.add(track_id)
            assignments[index] = track_id

        for track_id, track in list(self._tracks.items()):
            if track_id in matched_tracks:
                continue
            track.misses += 1
            if track.misses > self.max_misses:
                del self._tracks[track_id]

        for index, item in enumerate(items):
            bbox = [float(value) for value in item["bbox_xyxy"]]
            raw_distance = item.get("distance_m")
            distance = None if raw_distance is None else float(raw_distance)
            track_id = assignments.get(index)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                track = _Track(
                    track_id=track_id,
                    bbox_xyxy=bbox,
                    timestamp=timestamp,
                    distance_m=distance,
                    distance_timestamp=timestamp if distance is not None else None,
                )
                self._tracks[track_id] = track
            else:
                track = self._tracks[track_id]
                elapsed = (
                    timestamp - track.distance_timestamp
                    if track.distance_timestamp is not None
                    else 0.0
                )
                if (
                    elapsed > 0.0
                    and distance is not None
                    and track.distance_m is not None
                ):
                    instantaneous = (track.distance_m - distance) / elapsed
                    previous = track.approach_speed_mps
                    track.approach_speed_mps = (
                        instantaneous
                        if previous is None
                        else self.speed_smoothing * previous
                        + (1.0 - self.speed_smoothing) * instantaneous
                    )
                track.bbox_xyxy = bbox
                track.timestamp = timestamp
                track.hits += 1
                track.misses = 0
                if distance is not None:
                    track.distance_m = distance
                    track.distance_timestamp = timestamp

            speed = track.approach_speed_mps
            ttc = (
                track.distance_m / speed
                if track.distance_m is not None
                and speed is not None
                and speed >= self.min_approach_speed_mps
                else None
            )
            item["track_id"] = track.track_id
            item["track_hits"] = track.hits
            item["approach_speed_mps"] = None if speed is None else round(speed, 3)
            item["time_to_collision_s"] = None if ttc is None else round(ttc, 2)
        return items

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
