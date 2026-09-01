"""Single-owner priority audio queue for every SecondEye feature."""

from __future__ import annotations

import heapq
import threading
from dataclasses import dataclass, field
from typing import Any

from .orchestrator import AlertPriority


@dataclass(order=True, slots=True)
class _QueuedSpeech:
    sort_priority: int
    sequence: int
    text: str = field(compare=False)
    priority: AlertPriority = field(compare=False)


class PriorityAudioManager:
    """Serialize TTS and let safety alerts interrupt semantic speech."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self._condition = threading.Condition()
        self._queue: list[_QueuedSpeech] = []
        self._sequence = 0
        self._active_priority: AlertPriority | None = None
        self._last_text: str | None = None
        self._last_error: str | None = None
        self._closed = False
        self._worker = threading.Thread(
            target=self._run, name="secondeye-audio-owner", daemon=True
        )
        self._worker.start()

    @property
    def last_text(self) -> str | None:
        with self._condition:
            return self._last_text

    @property
    def last_error(self) -> str | None:
        with self._condition:
            return self._last_error

    def _record_error(self, exc: Exception) -> None:
        with self._condition:
            self._last_error = f"{type(exc).__name__}: {exc}"

    def _stop_backend(self) -> None:
        try:
            self.backend.stop()
        except Exception as exc:  # keep the audio owner alive after backend faults
            self._record_error(exc)

    def submit(
        self,
        text: str,
        *,
        priority: AlertPriority = AlertPriority.INFO,
    ) -> bool:
        normalized = " ".join(str(text).strip().split())
        if not normalized:
            return False
        with self._condition:
            if self._closed:
                return False
            self._sequence += 1
            heapq.heappush(
                self._queue,
                _QueuedSpeech(-int(priority), self._sequence, normalized, priority),
            )
            self._last_text = normalized
            should_interrupt = (
                self._active_priority is not None and priority > self._active_priority
            )
            self._condition.notify()
        if should_interrupt:
            self._stop_backend()
        return True

    def repeat(self) -> bool:
        with self._condition:
            text = self._last_text
        return (
            False
            if text is None
            else self.submit(text, priority=AlertPriority.SEMANTIC)
        )

    def stop(self) -> None:
        with self._condition:
            self._queue.clear()
            self._active_priority = None
        self._stop_backend()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._queue.clear()
            self._condition.notify_all()
        self._stop_backend()
        self._worker.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or bool(self._queue))
                if self._closed:
                    return
                item = heapq.heappop(self._queue)
                self._active_priority = item.priority
            try:
                self.backend.speak(item.text, interrupt=True)
                self.backend.wait()
            except Exception as exc:
                self._record_error(exc)
                self._stop_backend()
            finally:
                with self._condition:
                    self._active_priority = None
