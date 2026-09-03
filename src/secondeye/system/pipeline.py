"""Unified frame processing for the pretrained integration baseline."""

from __future__ import annotations

import re
import time
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Callable

from secondeye.detection.geometry import (
    CameraIntrinsics,
    GeometryObstacleConfig,
    detect_geometry_obstacles,
    fuse_geometry_with_detections,
)
from secondeye.multimodal.depth import (
    DepthFusionConfig,
    attach_depth_zones,
    attach_metric_depth_zones,
)
from secondeye.multimodal.ocr import OcrConsensusConfig
from secondeye.multimodal.quality import assess_image_quality
from secondeye.multimodal.questions import (
    normalize_visual_question,
    plain_vietnamese,
)
from secondeye.multimodal.speech import localize_vqa_answer

from .localization import VI_DIRECTIONS, VI_LABELS
from .orchestrator import (
    AlertPriority,
    SystemOrchestrator,
    SystemState,
    compose_obstacle_announcement,
)
from .tracking import DetectionTracker

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
        if (
            confidence is not None
            and float(confidence) < _SEMANTIC_DETECTION_CONFIDENCE
        ):
            discarded += 1
            continue
        accepted.append(item)
    return accepted, discarded


def _bbox_iou(first: Any, second: Any) -> float:
    a = [float(value) for value in first]
    b = [float(value) for value in second]
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


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
                confidence if previous is None else max(float(previous), confidence)
            )
    if not groups:
        return (
            "Tôi chưa nhận diện được vật thể đủ chắc chắn trong cảnh.",
            True,
            [],
            discarded,
        )

    depth_rank = {"emergency": 0, "near": 1, "medium": 2, "far": 3, None: 4}
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
        if depth in {"near", "emergency"}:
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
        semantic_detector: Any | None = None,
        translator: Any | None = None,
        tts: Any | None = None,
        orchestrator: SystemOrchestrator | None = None,
        depth_fusion_config: DepthFusionConfig | None = None,
        geometry_config: GeometryObstacleConfig | None = None,
        tracker: DetectionTracker | None = None,
        emergency_ttc_seconds: float = 1.5,
        ocr_consensus_config: OcrConsensusConfig | None = None,
    ) -> None:
        self.detector = detector
        self.depth = depth
        self.ocr = ocr
        self.vqa = vqa
        self.semantic_detector = semantic_detector
        self.translator = translator
        self.tts = tts
        self.orchestrator = orchestrator or SystemOrchestrator()
        self.depth_fusion_config = depth_fusion_config or DepthFusionConfig()
        self.geometry_config = geometry_config or GeometryObstacleConfig()
        self.tracker = tracker or DetectionTracker()
        if emergency_ttc_seconds <= 0.0:
            raise ValueError("emergency_ttc_seconds phải dương")
        self.emergency_ttc_seconds = emergency_ttc_seconds
        self.ocr_consensus_config = ocr_consensus_config or OcrConsensusConfig()

    def warmup(self) -> None:
        """Warm latency-sensitive vision models before starting live capture."""
        self.detector.warmup()
        if self.depth is not None:
            depth_warmup = getattr(self.depth, "warmup", None)
            if callable(depth_warmup):
                depth_warmup()

    def runtime_manifest(self) -> dict[str, object]:
        detector_manifest = getattr(self.detector, "runtime_manifest", None)
        return {
            "detector": (
                detector_manifest()
                if callable(detector_manifest)
                else {"type": type(self.detector).__name__}
            ),
            "depth": (
                None
                if self.depth is None
                else {
                    "type": type(self.depth).__name__,
                    "model": getattr(self.depth, "model_name", None),
                    "model_revision": getattr(self.depth, "model_revision", None),
                    "depth_type": getattr(self.depth, "depth_type", None),
                    "device": getattr(self.depth, "device", None),
                }
            ),
            "semantic_detector": (
                None
                if self.semantic_detector is None
                else {"type": type(self.semantic_detector).__name__}
            ),
            "safety": {
                "confirmation_frames": self.orchestrator.confirmation_frames,
                "rearm_absent_frames": self.orchestrator.rearm_absent_frames,
                "cooldown_seconds": self.orchestrator.cooldown_seconds,
                "max_evidence_gap_seconds": (
                    self.orchestrator.max_evidence_gap_seconds
                ),
                "emergency_ttc_seconds": self.emergency_ttc_seconds,
                "relative_depth_alerts": False,
                "bbox_only_alerts": False,
            },
        }

    def warmup_frame(self, image: Any) -> None:
        """Warm shape-specific backend kernels with an actual camera frame."""
        self.detector.predict_bgr(image)
        if self.depth is not None:
            self.depth.predict_bgr(image)

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

    def _grounded_semantics(
        self,
        image: Any,
        detection_result: dict[str, object] | None,
    ) -> dict[str, object]:
        base = detection_result or self.detector.predict_bgr(image)
        if self.semantic_detector is None:
            return base
        expanded = self.semantic_detector.predict_bgr(image)
        merged = [dict(item) for item in base.get("detections", [])]
        for raw_item in expanded.get("detections", []):
            item = dict(raw_item)
            duplicate = any(
                str(existing.get("label")) == str(item.get("label"))
                and _bbox_iou(existing["bbox_xyxy"], item["bbox_xyxy"]) >= 0.5
                for existing in merged
            )
            if not duplicate:
                merged.append(item)
        return {
            **base,
            "detections": merged,
            "semantic_expansion": {
                "model": expanded.get("model"),
                "added_count": len(merged) - len(base.get("detections", [])),
                "latency_ms": expanded.get("latency_ms"),
            },
        }

    def process_frame(
        self,
        image: Any,
        *,
        with_depth: bool = True,
        captured_at: float | None = None,
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
            captured_at=captured_at,
        )

    def fuse_detection_and_depth(
        self,
        detection: dict[str, object],
        depth_result: dict[str, object] | None,
        *,
        started_at: float | None = None,
        depth_age_ms: float | None = None,
        captured_at: float | None = None,
        safety_enabled: bool = True,
        safety_age_check: Callable[[], bool] | None = None,
    ) -> dict[str, object]:
        """Fuse independently scheduled detection/depth results safely."""
        started = time.perf_counter() if started_at is None else started_at
        detections = list(detection["detections"])
        depth_usable = depth_result is not None and bool(
            depth_result.get("usable", True)
        )
        depth_type = None if depth_result is None else depth_result.get("depth_type")
        geometry: dict[str, object] | None = None
        metric_evaluable = bool(
            depth_usable
            and depth_result is not None
            and (depth_type == "metric" or "metric_depth_m" in depth_result)
        )
        if metric_evaluable and depth_result is not None:
            metric_depth = depth_result["metric_depth_m"]
            detections = attach_metric_depth_zones(
                detections,
                metric_depth,
                config=self.depth_fusion_config,
            )
            raw_intrinsics = depth_result.get("intrinsics")
            intrinsics = (
                CameraIntrinsics.from_mapping(raw_intrinsics)
                if isinstance(raw_intrinsics, dict)
                else None
            )
            geometry_obstacles, geometry = detect_geometry_obstacles(
                metric_depth,
                intrinsics=intrinsics,
                config=self.geometry_config,
                depth_config=self.depth_fusion_config,
            )
            detections = fuse_geometry_with_detections(detections, geometry_obstacles)
            metric_evaluable = bool(geometry.get("usable"))
        elif (
            depth_usable
            and depth_result is not None
            and "relative_inverse_depth" in depth_result
        ):
            detections = attach_depth_zones(
                detections,
                depth_result["relative_inverse_depth"],
                config=self.depth_fusion_config,
            )
        tracked_at = time.monotonic() if captured_at is None else captured_at
        detections = self.tracker.update(detections, timestamp=tracked_at)
        for item in detections:
            ttc = item.get("time_to_collision_s")
            if (
                item.get("safety_evaluable")
                and ttc is not None
                and float(ttc) <= self.emergency_ttc_seconds
            ):
                item["risk_level"] = "emergency"
                item["proximity_zone"] = "emergency"
                item["proximity_reason"] = "metric_geometry_low_ttc"
        within_age_limit = (
            True if safety_age_check is None else bool(safety_age_check())
        )
        risk_evidence_current = bool(
            safety_enabled and within_age_limit and metric_evaluable
        )
        alerts = []
        if risk_evidence_current:
            alertable = [item for item in detections if item.get("safety_evaluable")]
            alerts = self.orchestrator.obstacle_alerts(alertable)
        else:
            self.orchestrator.evidence_unavailable()
        if self.tts is not None and alerts:
            self._speak(compose_obstacle_announcement(alerts), AlertPriority.OBSTACLE)
        return {
            "schema_version": "2.0",
            "system": "SecondEye",
            "mode": "pretrained_integration",
            "state": self.orchestrator.state.value,
            "detection": {**detection, "detections": detections},
            "depth": self._serializable_depth(depth_result),
            "depth_used_for_alert": bool(alerts and metric_evaluable),
            "alert_evidence": (
                "metric_floor_geometry" if alerts and metric_evaluable else None
            ),
            "risk_evidence_current": risk_evidence_current,
            "geometry": geometry,
            "depth_age_ms": None if depth_age_ms is None else round(depth_age_ms, 2),
            "alerts": [
                {
                    "key": alert.key,
                    "text": alert.text,
                    "priority": int(alert.priority),
                    "state": alert.state.value,
                    "track_id": alert.track_id,
                    "label": alert.label,
                    "direction": alert.direction,
                    "distance_m": alert.distance_m,
                }
                for alert in alerts
            ],
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "limitations": [
                "Pretrained integration baseline, không phải hệ thống điều hướng đã kiểm định.",
                "Depth tương đối và bbox không được dùng làm bằng chứng cảnh báo.",
                "Mặt sàn không ước lượng được thì hệ thống chủ động từ chối cảnh báo.",
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
            if key not in {"relative_inverse_depth", "metric_depth_m"}
        }

    @staticmethod
    def _normalize_ocr_consensus_text(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        return re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE).strip()

    @staticmethod
    def _finalize_ocr_candidate(
        result: dict[str, object], quality: Any
    ) -> dict[str, object]:
        text = str(result.get("transcript", "")).strip()
        confidences = [
            float(line["confidence"])
            for line in result.get("lines", [])
            if line.get("confidence") is not None
        ]
        mean_confidence = sum(confidences) / len(confidences) if confidences else None
        return {
            **result,
            "abstained": not text
            or (mean_confidence is not None and mean_confidence < 0.35),
            "mean_confidence": (
                None if mean_confidence is None else round(mean_confidence, 4)
            ),
            "quality": quality.as_dict(),
        }

    def read_text(self, image: Any) -> dict[str, object]:
        return self.read_text_frames([image])

    def read_text_frames(
        self, images: list[Any] | tuple[Any, ...]
    ) -> dict[str, object]:
        """Read a short burst and speak only a temporally stable transcript."""
        if self.ocr is None:
            raise RuntimeError("OCR chưa được bật")
        frames = list(images)
        if not frames:
            raise ValueError("OCR cần ít nhất một frame")
        assessed = [assess_image_quality(image) for image in frames]
        ranked = sorted(
            (
                (index, quality)
                for index, quality in enumerate(assessed)
                if quality.acceptable
            ),
            key=lambda item: (
                item[1].sharpness,
                item[1].contrast,
                -abs(item[1].brightness - 128.0),
            ),
            reverse=True,
        )[: self.ocr_consensus_config.max_candidates]
        if not ranked:
            quality = assessed[-1]
            result = {
                "schema_version": "1.0",
                "module": "ocr",
                "success": True,
                "abstained": True,
                "reason": quality.reason,
                "transcript": "",
                "quality": quality.as_dict(),
                "frame_count": len(frames),
                "evaluated_frame_count": 0,
                "consensus_score": None,
            }
            self._speak(quality.guidance_vi, AlertPriority.SEMANTIC)
            self.orchestrator.transition(SystemState.READ)
            return result

        candidates: list[tuple[int, dict[str, object]]] = []
        errors: list[dict[str, object]] = []
        for index, quality in ranked:
            try:
                raw_result = self.ocr.read_bgr(frames[index])
            except Exception as exc:
                errors.append(
                    {
                        "frame_index": index,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                continue
            candidates.append(
                (index, self._finalize_ocr_candidate(raw_result, quality))
            )
        if not candidates:
            detail = "; ".join(
                f"frame {item['frame_index']}: {item['message']}" for item in errors
            )
            raise RuntimeError(f"OCR thất bại trên mọi frame: {detail}")

        nonempty = [
            (index, result)
            for index, result in candidates
            if self._normalize_ocr_consensus_text(str(result.get("transcript", "")))
        ]
        consensus_score = 1.0
        if len(ranked) > 1 and len(nonempty) < 2:
            selected_index, selected = nonempty[0] if nonempty else candidates[0]
            consensus_score = 0.0
        elif len(nonempty) >= 2:
            scored: list[tuple[float, int, dict[str, object]]] = []
            for index, result in nonempty:
                text = self._normalize_ocr_consensus_text(
                    str(result.get("transcript", ""))
                )
                similarities = [
                    SequenceMatcher(
                        None,
                        text,
                        self._normalize_ocr_consensus_text(
                            str(other.get("transcript", ""))
                        ),
                    ).ratio()
                    for other_index, other in nonempty
                    if other_index != index
                ]
                # One agreeing peer is enough to outvote a single unstable
                # frame in a three-frame burst. With two frames this is their
                # direct similarity.
                score = max(similarities)
                scored.append((score, index, result))
            consensus_score, selected_index, selected = max(
                scored,
                key=lambda item: (
                    item[0],
                    float(item[2].get("mean_confidence") or 0.0),
                    float(item[2]["quality"]["sharpness"]),
                ),
            )
        elif nonempty:
            selected_index, selected = nonempty[0]
        else:
            selected_index, selected = candidates[0]

        text = str(selected.get("transcript", "")).strip()
        abstained = bool(selected.get("abstained", False))
        abstention_reason = (
            "no_text" if not text else ("low_mean_confidence" if abstained else None)
        )
        if (
            len(ranked) > 1
            and consensus_score < self.ocr_consensus_config.minimum_consensus
        ):
            abstained = True
            abstention_reason = "ocr_temporal_disagreement"
        if not abstained:
            spoken = text
        elif abstention_reason == "ocr_temporal_disagreement":
            spoken = (
                "Văn bản chưa ổn định giữa các khung hình. "
                "Hãy giữ camera ổn định rồi thử lại."
            )
        else:
            spoken = "Tôi không đọc được văn bản đủ rõ trong ảnh."
        result = {
            **selected,
            "abstained": abstained,
            "abstention_reason": abstention_reason,
            "frame_count": len(frames),
            "evaluated_frame_count": len(candidates),
            "selected_frame_index": selected_index,
            "consensus_score": round(consensus_score, 4),
            "minimum_consensus": self.ocr_consensus_config.minimum_consensus,
            "candidate_summaries": [
                {
                    "frame_index": index,
                    "engine": candidate.get("engine"),
                    "has_text": bool(str(candidate.get("transcript", "")).strip()),
                    "mean_confidence": candidate.get("mean_confidence"),
                    "line_count": len(candidate.get("lines", [])),
                    "quality": candidate.get("quality"),
                }
                for index, candidate in candidates
            ],
            "candidate_errors": errors,
            "limitations": list(selected.get("limitations", []))
            + [
                "Đồng thuận nhiều frame chỉ đo độ ổn định; các frame có thể lặp lại cùng một lỗi OCR."
            ],
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
            grounded = self._grounded_semantics(image, detection_result)
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
                    marker in plain_question for marker in ("phia truoc", "truoc mat")
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
        if image is None and detection_result is None:
            raise ValueError("describe_scene cần image hoặc detection_result")
        if image is not None:
            detection_result = self._grounded_semantics(image, detection_result)
        assert detection_result is not None
        text, abstained, groups, discarded = _scene_summary(detection_result)
        self.orchestrator.transition(SystemState.SCENE)
        self._speak(text, AlertPriority.SEMANTIC)
        return {
            "schema_version": "1.0",
            "module": "grounded_scene_description",
            "success": True,
            "description": text,
            "abstained": abstained,
            "source": (
                "pretrained_detection_plus_open_vocabulary"
                if detection_result.get("semantic_expansion") is not None
                else "pretrained_detection"
            ),
            "object_groups": groups,
            "discarded_low_confidence": discarded,
            "limitations": [
                "Chỉ mô tả detection có confidence từ 0.45 và các lớp được cấu hình.",
                "Open-vocabulary (nếu bật) chỉ bổ sung ngữ nghĩa, không xác nhận an toàn.",
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
