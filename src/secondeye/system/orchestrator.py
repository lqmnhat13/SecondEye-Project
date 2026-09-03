"""State, priority, cooldown and audio decisions for all SecondEye modules."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Callable


class SystemState(str, Enum):
    IDLE = "IDLE"
    OBSTACLE = "OBSTACLE"
    READ = "READ"
    SCENE = "SCENE"
    QUESTION = "QUESTION"
    ERROR = "ERROR"


class AlertPriority(IntEnum):
    INFO = 10
    SEMANTIC = 20
    ERROR = 30
    OBSTACLE = 40
    STOP = 100


@dataclass(frozen=True, slots=True)
class Alert:
    key: str
    text: str
    priority: AlertPriority
    state: SystemState
    track_id: int | None = None
    label: str | None = None
    direction: str | None = None
    distance_m: float | None = None


_VI_LABELS = {
    "person": "người",
    "chair": "ghế",
    "table": "bàn",
    "sofa": "ghế sofa",
    "bed": "giường",
    "backpack": "ba lô",
    "handbag": "túi xách",
    "suitcase": "va li",
    "bottle": "chai",
    "potted_plant": "chậu cây",
    "unknown_obstacle": "vật cản",
    "dropoff": "chỗ hụt",
}

_VI_DIRECTIONS = {
    "left": "bên trái",
    "center": "phía trước",
    "right": "bên phải",
}


def compose_obstacle_announcement(alerts: list[Alert]) -> str:
    """Combine simultaneous hazards into one utterance without dropping any."""
    if not alerts:
        return ""
    phrases: list[str] = []
    seen: set[tuple[str, str]] = set()
    for alert in alerts:
        label = alert.label or "unknown_obstacle"
        direction = alert.direction or "center"
        identity = (label, direction)
        if identity in seen:
            continue
        seen.add(identity)
        readable = _VI_LABELS.get(label, label)
        location = _VI_DIRECTIONS.get(direction, "phía trước")
        distance = (
            f", cách {alert.distance_m:.1f} mét" if alert.distance_m is not None else ""
        )
        phrases.append(f"{readable} {location}{distance}")
    return "Cẩn thận: " + "; ".join(phrases) + "."


class SystemOrchestrator:
    def __init__(
        self,
        *,
        cooldown_seconds: float = 4.0,
        confirmation_frames: int = 2,
        rearm_absent_frames: int = 3,
        max_evidence_gap_seconds: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds không được âm")
        if confirmation_frames <= 0:
            raise ValueError("confirmation_frames phải dương")
        if rearm_absent_frames <= 0:
            raise ValueError("rearm_absent_frames phải dương")
        if max_evidence_gap_seconds <= 0.0:
            raise ValueError("max_evidence_gap_seconds phải dương")
        self.cooldown_seconds = cooldown_seconds
        self.confirmation_frames = confirmation_frames
        self.rearm_absent_frames = rearm_absent_frames
        self.max_evidence_gap_seconds = max_evidence_gap_seconds
        self.clock = clock
        self.state = SystemState.IDLE
        self._last_emitted: dict[str, float] = {}
        self._near_streaks: dict[str, int] = {}
        self._active_near: set[str] = set()
        self._absent_streaks: dict[str, int] = {}
        self._last_evidence_at: float | None = None

    def transition(self, state: SystemState) -> None:
        self.state = state

    def obstacle_alerts(self, detections: list[dict[str, Any]]) -> list[Alert]:
        """Emit track-aware alerts while keeping hazard state through cooldown."""
        alerts: list[Alert] = []
        now = self.clock()
        self._last_evidence_at = now
        visible_near: dict[str, dict[str, Any]] = {}
        for detection in detections:
            if not detection.get("obstacle_candidate"):
                continue
            proximity = detection.get("proximity_zone", detection.get("depth_zone"))
            if proximity not in {"near", "emergency"}:
                continue
            label = str(detection.get("label", "unknown_obstacle"))
            direction = str(detection.get("direction", "center"))
            track_id = detection.get("track_id")
            key = (
                f"track:{int(track_id)}"
                if track_id is not None
                else f"near:{label}:{direction}"
            )
            visible_near[key] = detection
        for key, detection in visible_near.items():
            self._absent_streaks.pop(key, None)
            self._near_streaks[key] = self._near_streaks.get(key, 0) + 1
            emergency = (
                detection.get("proximity_zone", detection.get("depth_zone"))
                == "emergency"
                or detection.get("risk_level") == "emergency"
            )
            if not emergency and self._near_streaks[key] < self.confirmation_frames:
                continue
            if key in self._active_near:
                continue
            self._active_near.add(key)
            last = self._last_emitted.get(key)
            if last is not None and now - last < self.cooldown_seconds:
                continue
            self._last_emitted[key] = now
            label = str(detection.get("label", "unknown_obstacle"))
            direction = str(detection.get("direction", "center"))
            track_id = detection.get("track_id")
            distance = detection.get("distance_m")
            readable = _VI_LABELS.get(label, label)
            alerts.append(
                Alert(
                    key=key,
                    text=(
                        f"Cẩn thận, {readable} "
                        f"{_VI_DIRECTIONS.get(direction, 'phía trước')}."
                    ),
                    priority=AlertPriority.OBSTACLE,
                    state=SystemState.OBSTACLE,
                    track_id=None if track_id is None else int(track_id),
                    label=label,
                    direction=direction,
                    distance_m=None if distance is None else float(distance),
                )
            )
        known_keys = set(self._near_streaks) | self._active_near
        for key in known_keys - set(visible_near):
            absent = self._absent_streaks.get(key, 0) + 1
            if absent < self.rearm_absent_frames:
                self._absent_streaks[key] = absent
                continue
            self._absent_streaks.pop(key, None)
            self._near_streaks.pop(key, None)
            self._active_near.discard(key)
        alerts.sort(key=lambda item: item.priority, reverse=True)
        self.state = SystemState.OBSTACLE if self._near_streaks else SystemState.IDLE
        return alerts

    def evidence_unavailable(self) -> None:
        """Expire obstacle state if current metric evidence disappears too long."""
        if self._last_evidence_at is None:
            return
        if self.clock() - self._last_evidence_at <= self.max_evidence_gap_seconds:
            return
        self._near_streaks.clear()
        self._active_near.clear()
        self._absent_streaks.clear()
        self._last_evidence_at = None
        if self.state is SystemState.OBSTACLE:
            self.state = SystemState.IDLE

    def semantic_alert(self, text: str, state: SystemState) -> Alert:
        if state not in {SystemState.READ, SystemState.SCENE, SystemState.QUESTION}:
            raise ValueError("semantic alert cần trạng thái READ/SCENE/QUESTION")
        self.state = state
        return Alert(
            key=f"semantic:{state.value}",
            text=text,
            priority=AlertPriority.SEMANTIC,
            state=state,
        )

    def reset(self) -> None:
        self.state = SystemState.IDLE
        self._near_streaks.clear()
        self._active_near.clear()
        self._absent_streaks.clear()
        self._last_evidence_at = None
