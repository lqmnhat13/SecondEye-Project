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
}


class SystemOrchestrator:
    def __init__(
        self,
        *,
        cooldown_seconds: float = 4.0,
        confirmation_frames: int = 2,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds không được âm")
        if confirmation_frames <= 0:
            raise ValueError("confirmation_frames phải dương")
        self.cooldown_seconds = cooldown_seconds
        self.confirmation_frames = confirmation_frames
        self.clock = clock
        self.state = SystemState.IDLE
        self._last_emitted: dict[str, float] = {}
        self._near_streaks: dict[str, int] = {}

    def transition(self, state: SystemState) -> None:
        self.state = state

    def obstacle_alerts(self, detections: list[dict[str, Any]]) -> list[Alert]:
        """Emit only depth-confirmed near central obstacle candidates."""
        alerts: list[Alert] = []
        now = self.clock()
        visible_near: dict[str, str] = {}
        for detection in detections:
            if not detection.get("obstacle_candidate"):
                continue
            if detection.get("depth_zone") != "near":
                continue
            label = str(detection["label"])
            direction = str(detection.get("direction", "center"))
            key = f"near:{label}:{direction}"
            visible_near[key] = label
        for key, label in visible_near.items():
            self._near_streaks[key] = self._near_streaks.get(key, 0) + 1
            if self._near_streaks[key] < self.confirmation_frames:
                continue
            last = self._last_emitted.get(key)
            if last is not None and now - last < self.cooldown_seconds:
                continue
            self._last_emitted[key] = now
            readable = _VI_LABELS.get(label, label)
            alerts.append(
                Alert(
                    key=key,
                    text=f"Cảnh báo, có {readable} ở gần phía trước.",
                    priority=AlertPriority.OBSTACLE,
                    state=SystemState.OBSTACLE,
                )
            )
        for key in set(self._near_streaks) - set(visible_near):
            self._near_streaks.pop(key, None)
        alerts.sort(key=lambda item: item.priority, reverse=True)
        if alerts:
            self.state = SystemState.OBSTACLE
        return alerts

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
