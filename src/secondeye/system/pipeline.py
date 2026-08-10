"""Unified frame processing for the pretrained integration baseline."""

from __future__ import annotations

import time
from typing import Any

from secondeye.multimodal.depth import attach_depth_zones

from .orchestrator import SystemOrchestrator, SystemState


class SecondEyeSystem:
    """Compose adapters without coupling their model implementations."""

    def __init__(
        self,
        *,
        detector: Any,
        depth: Any | None = None,
        ocr: Any | None = None,
        vqa: Any | None = None,
        tts: Any | None = None,
        orchestrator: SystemOrchestrator | None = None,
    ) -> None:
        self.detector = detector
        self.depth = depth
        self.ocr = ocr
        self.vqa = vqa
        self.tts = tts
        self.orchestrator = orchestrator or SystemOrchestrator()

    def process_frame(self, image: Any, *, with_depth: bool = True) -> dict[str, object]:
        started = time.perf_counter()
        detection = self.detector.predict_bgr(image)
        detections = list(detection["detections"])
        depth_result: dict[str, object] | None = None
        if with_depth and self.depth is not None:
            depth_result = self.depth.predict_bgr(image)
            detections = attach_depth_zones(
                detections, depth_result["relative_inverse_depth"]
            )
        alerts = self.orchestrator.obstacle_alerts(detections)
        if self.tts is not None and alerts:
            self.tts.speak(alerts[0].text, interrupt=True)
        return {
            "schema_version": "1.0",
            "system": "SecondEye",
            "mode": "pretrained_integration",
            "state": self.orchestrator.state.value,
            "detection": {**detection, "detections": detections},
            "depth": self._serializable_depth(depth_result),
            "alerts": [
                {
                    "key": alert.key,
                    "text": alert.text,
                    "priority": int(alert.priority),
                    "state": alert.state.value,
                }
                for alert in alerts
            ],
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "limitations": [
                "Pretrained integration baseline, không phải hệ thống điều hướng đã kiểm định.",
                "Không hỗ trợ cửa, cầu thang, cột, tủ, hộp hoặc thùng rác.",
            ],
        }

    @staticmethod
    def _serializable_depth(result: dict[str, object] | None) -> dict[str, object] | None:
        if result is None:
            return None
        return {
            key: value
            for key, value in result.items()
            if key != "relative_inverse_depth"
        }

    def read_text(self, image: Any) -> dict[str, object]:
        if self.ocr is None:
            raise RuntimeError("OCR chưa được bật")
        result = self.ocr.read_bgr(image)
        text = str(result.get("transcript", "")).strip()
        if self.tts is not None:
            self.tts.speak(text or "Tôi không đọc được văn bản trong ảnh.")
        self.orchestrator.transition(SystemState.READ)
        return result

    def ask(self, image: Any, question: str) -> dict[str, object]:
        if self.vqa is None:
            raise RuntimeError("VQA chưa được bật")
        result = self.vqa.ask_bgr(image, question)
        answer = str(result["answer"])
        if self.tts is not None:
            self.tts.speak(answer)
        self.orchestrator.transition(SystemState.QUESTION)
        return result

    def stop_audio(self) -> None:
        if self.tts is not None:
            self.tts.stop()
        self.orchestrator.reset()
