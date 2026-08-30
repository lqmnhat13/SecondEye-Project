"""Thread-safe JSONL session logging for reproducible local demos."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


class SessionLogger:
    def __init__(self, path: Path | None = None) -> None:
        self.session_id = uuid.uuid4().hex[:12]
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.path = (
            (path or Path("logs") / f"session_{stamp}_{self.session_id}.jsonl")
            .expanduser()
            .resolve()
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, event: str, payload: Any, *, success: bool = True) -> None:
        record = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "session_id": self.session_id,
            "event": event,
            "success": success,
            "payload": json_safe(payload),
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line)
