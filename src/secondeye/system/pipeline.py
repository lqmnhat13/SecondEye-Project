"""Unified frame processing for the pretrained integration baseline."""

from __future__ import annotations

import time
from typing import Any

from secondeye.multimodal.depth import attach_depth_zones
from secondeye.multimodal.quality import assess_image_quality
from secondeye.multimodal.questions import (
    normalize_visual_question,
    plain_vietnamese,
)
from secondeye.multimodal.speech import localize_vqa_answer

from .localization import VI_DIRECTIONS, VI_LABELS
from .orchestrator import AlertPriority, SystemOrchestrator, SystemState

_UNSAFE_VQA_TERMS = (
    "dẫn đường",
    "đi đường nào",
    "có an toàn để đi",
    "băng qua",
    "navigate",
    "which way should i go",
    "safe to cross",
)

_SEMANTIC_DETECTION_CONFIDENCE = 0.45
_VI_NUMBERS = {
    0: "không",
    1: "một",
    2: "hai",
    3: "ba",
    4: "bốn",
    5: "năm",
    6: "sáu",
    7: "bảy",
    8: "tám",
    9: "chín",
    10: "mười",
}


def _semantic_detections(
    detection_result: dict[str, object],
) -> tuple[list[dict[str, object]], int]:
    accepted: list[dict[str, object]] = []
    discarded = 0
    for raw_item in detection_result.get("detections", []):
        item = dict(raw_item)
        confidence = item.get("confidence")
        if confidence is not None and float(confidence) < _SEMANTIC_DETECTION_CONFIDENCE:
            discarded += 1
            continue
        accepted.append(item)
    return accepted, discarded


def _scene_summary(
    detection_result: dict[str, object],
    *,
    directions: set[str] | None = None,
) -> tuple[str, bool, list[dict[str, object]], int]:
    detections, discarded = _semantic_detections(detection_result)
    if directions is not None:
        detections = [
            item for item in detections if str(item.get("direction")) in directions
        ]
    groups: dict[tuple[str, str, str | None], dict[str, object]] = {}
    for item in detections:
        label = str(item.get("label", ""))
        if not label:
            continue
        direction = str(item.get("direction", "center"))
        depth = item.get("depth_zone")
        depth_zone = None if depth in (None, "", "unknown") else str(depth)
        key = (label, direction, depth_zone)
        group = groups.setdefault(
            key,
            {
                "label": label,
                "direction": direction,
                "depth_zone": depth_zone,
                "count": 0,
                "max_confidence": None,
            },
        )
        group["count"] = int(group["count"]) + 1
        if item.get("confidence") is not None:
            confidence = float(item["confidence"])
            previous = group["max_confidence"]
            group["max_confidence"] = (
                confidence
                if previous is None
                else max(float(previous), confidence)
            )
    if not groups:
        return (
            "Tôi chưa nhận diện được vật thể đủ chắc chắn trong cảnh.",
            True,
            [],
            discarded,
        )

    depth_rank = {"near": 0, "medium": 1, "far": 2, None: 3}
    direction_rank = {"center": 0, "left": 1, "right": 1}
    ordered = sorted(
        groups.values(),
        key=lambda group: (
            depth_rank.get(group["depth_zone"], 3),
            direction_rank.get(str(group["direction"]), 2),
            -float(group["max_confidence"] or 0.0),
            str(group["label"]),
        ),
    )[:5]
    phrases: list[str] = []
    for group in ordered:
        count = int(group["count"])
        readable = VI_LABELS.get(str(group["label"]), str(group["label"]))
        prefix = f"{count} " if count > 1 else ""
        direction = VI_DIRECTIONS.get(str(group["direction"]), "trong ảnh")
        depth = group["depth_zone"]
        if depth == "near":
            location = f"ở gần {direction}"
        elif depth == "medium":
            location = f"ở khoảng cách trung bình {direction}"
        elif depth == "far":
            location = f"ở xa {direction}"
        else:
            location = direction
        phrases.append(f"{prefix}{readable} {location}")
    return "Tôi thấy " + ", ".join(phrases) + ".", False, ordered, discarded


class SecondEyeSystem:
    """Compose adapters without coupling their model implementations."""

    def __init__(
        self,
        *,
        detector: Any,
        depth: Any | None = None,
        ocr: Any | None = None,
        vqa: Any | None = None,
        translator: Any | None = None,
        tts: Any | None = None,
        orchestrator: SystemOrchestrator | None = None,
    ) -> None:
        self.detector = detector
        self.depth = depth
        self.ocr = ocr
        self.vqa = vqa
        self.translator = translator
        self.tts = tts
        self.orchestrator = orchestrator or SystemOrchestrator()

    def _speak(self, text: str, priority: AlertPriority) -> None:
        if self.tts is None or not text.strip():
            return
        submit = getattr(self.tts, "submit", None)
        if callable(submit):
            submit(text, priority=priority)
        else:
            self.tts.speak(text, interrupt=priority >= AlertPriority.OBSTACLE)

    def announce(self, text: str, priority: AlertPriority = AlertPriority.INFO) -> None:
        self._speak(text, priority)

    def process_frame(
        self, image: Any, *, with_depth: bool = True
    ) -> dict[str, object]:
        started = time.perf_counter()
        detection = self.detector.predict_bgr(image)
        depth_result: dict[str, object] | None = None
        if with_depth and self.depth is not None:
            depth_result = self.depth.predict_bgr(image)
        return self.fuse_detection_and_depth(
            detection,
            depth_result,
            started_at=started,
            depth_age_ms=0.0 if depth_result is not None else None,
        )

    def fuse_detection_and_depth(
        self,
        detection: dict[str, object],
        depth_result: dict[str, object] | None,
        *,
        started_at: float | None = None,
        depth_age_ms: float | None = None,
    ) -> dict[str, object]:
        """Fuse independently scheduled detection/depth results safely."""
        started = time.perf_counter() if started_at is None else started_at
        detections = list(detection["detections"])
        if depth_result is not None:
            detections = attach_depth_zones(
                detections, depth_result["relative_inverse_depth"]
            )
        alerts = self.orchestrator.obstacle_alerts(detections)
        if not alerts and self.orchestrator.state is SystemState.OBSTACLE:
            self.orchestrator.reset()
        if self.tts is not None and alerts:
            self._speak(alerts[0].text, AlertPriority.OBSTACLE)
        return {
            "schema_version": "1.0",
            "system": "SecondEye",
            "mode": "pretrained_integration",
            "state": self.orchestrator.state.value,
            "detection": {**detection, "detections": detections},
            "depth": self._serializable_depth(depth_result),
            "depth_age_ms": None if depth_age_ms is None else round(depth_age_ms, 2),
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
    def _serializable_depth(
        result: dict[str, object] | None,
    ) -> dict[str, object] | None:
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
        quality = assess_image_quality(image)
        if not quality.acceptable:
            result = {
                "schema_version": "1.0",
                "module": "ocr",
                "success": True,
                "abstained": True,
                "reason": quality.reason,
                "transcript": "",
                "quality": quality.as_dict(),
            }
            self._speak(quality.guidance_vi, AlertPriority.SEMANTIC)
            self.orchestrator.transition(SystemState.READ)
            return result
        result = self.ocr.read_bgr(image)
        text = str(result.get("transcript", "")).strip()
        confidences = [
            float(line["confidence"])
            for line in result.get("lines", [])
            if line.get("confidence") is not None
        ]
        mean_confidence = sum(confidences) / len(confidences) if confidences else None
        abstained = not text or (mean_confidence is not None and mean_confidence < 0.35)
        spoken = (
            text if not abstained else "Tôi không đọc được văn bản đủ rõ trong ảnh."
        )
        result = {
            **result,
            "abstained": abstained,
            "mean_confidence": None
            if mean_confidence is None
            else round(mean_confidence, 4),
            "quality": quality.as_dict(),
        }
        self._speak(spoken, AlertPriority.SEMANTIC)
        self.orchestrator.transition(SystemState.READ)
        return result

    def ask(
        self,
        image: Any,
        question: str,
        *,
        detection_result: dict[str, object] | None = None,
    ) -> dict[str, object]:
        question = " ".join(question.strip().split())
        if not question:
            raise ValueError("question không được rỗng")
        if any(term in question.lower() for term in _UNSAFE_VQA_TERMS):
            answer = "Tôi không thể dùng VQA để đưa ra chỉ dẫn di chuyển an toàn."
            result = {
                "schema_version": "1.0",
                "module": "vqa",
                "success": True,
                "question": question,
                "answer": answer,
                "spoken_answer_vi": answer,
                "abstained": True,
                "reason": "navigation_request_blocked",
                "localization_abstained": False,
            }
            self._speak(answer, AlertPriority.SEMANTIC)
            self.orchestrator.transition(SystemState.QUESTION)
            return result
        quality = assess_image_quality(image)
        if not quality.acceptable:
            answer = quality.guidance_vi
            result = {
                "schema_version": "1.0",
                "module": "vqa",
                "success": True,
                "question": question.strip(),
                "answer": answer,
                "spoken_answer_vi": answer,
                "abstained": True,
                "localization_abstained": False,
                "quality": quality.as_dict(),
            }
            self._speak(answer, AlertPriority.SEMANTIC)
            self.orchestrator.transition(SystemState.QUESTION)
            return result
        normalized = normalize_visual_question(question)
        normalization = normalized.as_dict()
        if normalized.intent == "unsupported":
            answer = (
                "Tôi chưa hỗ trợ dạng câu hỏi này. Hãy hỏi về số lượng, màu sắc, "
                "hành động, trang phục hoặc đồ vật phía trước."
            )
            result = {
                "schema_version": "1.0",
                "module": "vqa",
                "success": True,
                "question": question,
                "model_question": None,
                "question_normalization": normalization,
                "answer": answer,
                "spoken_answer_vi": answer,
                "abstained": True,
                "reason": "unsupported_vietnamese_question",
                "localization_abstained": False,
                "translation_used": False,
                "translation": None,
                "translation_error": None,
                "quality": quality.as_dict(),
            }
            self._speak(answer, AlertPriority.SEMANTIC)
            self.orchestrator.transition(SystemState.QUESTION)
            return result

        if normalized.intent in {"grounded_count", "grounded_scene"}:
            grounded = detection_result or self.detector.predict_bgr(image)
            if normalized.intent == "grounded_count":
                detections, discarded = _semantic_detections(grounded)
                count = (
                    len(detections)
                    if normalized.target_label == "object"
                    else sum(
                        str(item.get("label")) == normalized.target_label
                        for item in detections
                    )
                )
                if count:
                    number = _VI_NUMBERS.get(count, str(count))
                    readable = (
                        "vật thể"
                        if normalized.target_label == "object"
                        else VI_LABELS.get(
                            str(normalized.target_label), str(normalized.target_label)
                        )
                    )
                    answer = f"Tôi nhận diện được {number} {readable} trong ảnh."
                    abstained = False
                else:
                    answer = (
                        "Tôi chưa nhận diện được đối tượng được hỏi đủ chắc chắn "
                        "trong ảnh."
                    )
                    abstained = True
                object_groups = [
                    {
                        "label": normalized.target_label,
                        "count": count,
                    }
                ]
            else:
                plain_question = plain_vietnamese(question)
                front_only = "front" in question.casefold() or any(
                    marker in plain_question
                    for marker in ("phia truoc", "truoc mat")
                )
                answer, abstained, object_groups, discarded = _scene_summary(
                    grounded,
                    directions={"center"} if front_only else None,
                )
            result = {
                "schema_version": "1.0",
                "module": "grounded_visual_query",
                "success": True,
                "question": question,
                "model_question": normalized.model_question,
                "question_normalization": normalization,
                "answer": answer,
                "spoken_answer_vi": answer,
                "abstained": abstained,
                "reason": "no_grounded_detection" if abstained else None,
                "source": "pretrained_detection",
                "object_groups": object_groups,
                "discarded_low_confidence": discarded,
                "localization_abstained": False,
                "translation_used": False,
                "translation": None,
                "translation_error": None,
                "quality": quality.as_dict(),
            }
            self._speak(answer, AlertPriority.SEMANTIC)
            self.orchestrator.transition(SystemState.QUESTION)
            return result

        if self.vqa is None:
            raise RuntimeError("VQA chưa được bật")
        assert normalized.model_question is not None
        result = self.vqa.ask_bgr(image, normalized.model_question)
        answer = str(result["answer"])
        spoken_answer, localization_abstained = localize_vqa_answer(answer)
        translation: dict[str, object] | None = None
        translation_used = False
        translation_error: str | None = None
        if localization_abstained and not bool(result.get("abstained")):
            if self.translator is None:
                spoken_answer = "Chưa thể tải bộ dịch câu trả lời sang tiếng Việt."
            else:
                try:
                    translation = self.translator.translate(answer)
                    if translation.get("quality_assured") is not True:
                        raise RuntimeError("Bản dịch chưa được kiểm chứng chất lượng")
                    spoken_answer = str(translation["translation"]).strip()
                    localization_abstained = not bool(spoken_answer)
                    translation_used = not localization_abstained
                except Exception as exc:
                    translation_error = f"{type(exc).__name__}: {exc}"
                    spoken_answer = "Không dịch được câu trả lời sang tiếng Việt."
                    localization_abstained = True
        result = {
            **result,
            "question": question,
            "model_question": normalized.model_question,
            "question_normalization": normalization,
            "spoken_answer_vi": spoken_answer,
            "localization_abstained": localization_abstained,
            "translation_used": translation_used,
            "translation": translation,
            "translation_error": translation_error,
            "abstained": bool(result.get("abstained")) or localization_abstained,
            "quality": quality.as_dict(),
        }
        self._speak(spoken_answer, AlertPriority.SEMANTIC)
        self.orchestrator.transition(SystemState.QUESTION)
        return result

    def describe_scene(
        self,
        image: Any | None = None,
        *,
        detection_result: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Create a short Vietnamese description grounded in detector output."""
        if detection_result is None:
            if image is None:
                raise ValueError("describe_scene cần image hoặc detection_result")
            detection_result = self.detector.predict_bgr(image)
        text, abstained, groups, discarded = _scene_summary(detection_result)
        self.orchestrator.transition(SystemState.SCENE)
        self._speak(text, AlertPriority.SEMANTIC)
        return {
            "schema_version": "1.0",
            "module": "grounded_scene_description",
            "success": True,
            "description": text,
            "abstained": abstained,
            "source": "pretrained_detection",
            "object_groups": groups,
            "discarded_low_confidence": discarded,
            "limitations": [
                "Chỉ mô tả detection có confidence từ 0.45 và các lớp được cấu hình.",
                "Khoảng cách gần/trung bình/xa là tương đối, không phải mét.",
            ],
        }

    def repeat_audio(self) -> bool:
        repeat = getattr(self.tts, "repeat", None)
        return bool(repeat()) if callable(repeat) else False

    def stop_audio(self) -> None:
        if self.tts is not None:
            self.tts.stop()
        self.orchestrator.reset()

    def close(self) -> None:
        close = getattr(self.tts, "close", None)
        if callable(close):
            close()
