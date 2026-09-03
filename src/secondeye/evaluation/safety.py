"""Event-level safety metrics for replayed, human-annotated scenarios."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class SafetyMetrics:
    duration_minutes: float
    hazard_events: int
    detected_hazard_events: int
    hazard_event_recall: float | None
    critical_events: int
    detected_critical_events: int
    critical_event_recall: float | None
    false_alerts: int
    false_alerts_per_minute: float | None
    stale_alerts: int
    alert_latency_p50_ms: float | None
    alert_latency_p95_ms: float | None
    alert_latency_p99_ms: float | None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def evaluate_safety_rows(
    rows: Iterable[dict[str, Any]], *, max_source_age_ms: float = 750.0
) -> SafetyMetrics:
    """Evaluate frame rows without treating individual frames as independent hazards.

    Required fields are ``timestamp_s``, ``hazard_present`` and ``alert``.
    Hazard frames additionally need a stable ``hazard_id``. ``critical`` and
    ``source_age_ms`` are optional.
    """
    if max_source_age_ms <= 0.0:
        raise ValueError("max_source_age_ms phải dương")
    ordered = sorted(
        (dict(row) for row in rows), key=lambda row: float(row["timestamp_s"])
    )
    if not ordered:
        raise ValueError("evaluation cần ít nhất một row")
    events: dict[str, dict[str, Any]] = {}
    false_alerts = 0
    stale_alerts = 0
    for row in ordered:
        timestamp = float(row["timestamp_s"])
        hazard_present = bool(row.get("hazard_present", False))
        alerted = bool(row.get("alert", False))
        if alerted and float(row.get("source_age_ms", 0.0)) > max_source_age_ms:
            stale_alerts += 1
        if not hazard_present:
            if alerted:
                false_alerts += 1
            continue
        raw_id = row.get("hazard_id")
        if raw_id in (None, ""):
            raise ValueError("row có hazard_present=true phải có hazard_id")
        hazard_id = str(raw_id)
        event = events.setdefault(
            hazard_id,
            {
                "started_at": timestamp,
                "critical": False,
                "first_alert_at": None,
            },
        )
        event["started_at"] = min(float(event["started_at"]), timestamp)
        event["critical"] = bool(event["critical"] or row.get("critical", False))
        if alerted and event["first_alert_at"] is None:
            event["first_alert_at"] = timestamp

    duration_seconds = max(
        0.0, float(ordered[-1]["timestamp_s"]) - float(ordered[0]["timestamp_s"])
    )
    duration_minutes = duration_seconds / 60.0
    detected = [
        event for event in events.values() if event["first_alert_at"] is not None
    ]
    critical = [event for event in events.values() if event["critical"]]
    detected_critical = [
        event for event in critical if event["first_alert_at"] is not None
    ]
    latencies = [
        (float(event["first_alert_at"]) - float(event["started_at"])) * 1000.0
        for event in detected
    ]
    return SafetyMetrics(
        duration_minutes=round(duration_minutes, 4),
        hazard_events=len(events),
        detected_hazard_events=len(detected),
        hazard_event_recall=(len(detected) / len(events) if events else None),
        critical_events=len(critical),
        detected_critical_events=len(detected_critical),
        critical_event_recall=(
            len(detected_critical) / len(critical) if critical else None
        ),
        false_alerts=false_alerts,
        false_alerts_per_minute=(
            false_alerts / duration_minutes if duration_minutes > 0.0 else None
        ),
        stale_alerts=stale_alerts,
        alert_latency_p50_ms=_percentile(latencies, 50.0),
        alert_latency_p95_ms=_percentile(latencies, 95.0),
        alert_latency_p99_ms=_percentile(latencies, 99.0),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.expanduser().resolve().open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"dòng {line_number} không phải JSON object")
            rows.append(value)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--max-source-age-ms", type=float, default=750.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    metrics = evaluate_safety_rows(
        load_jsonl(args.input), max_source_age_ms=args.max_source_age_ms
    )
    payload = asdict(metrics)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
