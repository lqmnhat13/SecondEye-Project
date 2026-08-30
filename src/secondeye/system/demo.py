"""Unified, non-blocking camera MVP for the pretrained SecondEye stack."""

from __future__ import annotations

import queue
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .camera import AsyncVisionRuntime, LatestFrameCapture
from .localization import localize_state
from .orchestrator import AlertPriority, SystemState
from .overlay import UnicodeTextRenderer, draw_detection_overlays
from .session import SessionLogger


@dataclass(frozen=True, slots=True)
class SemanticCommand:
    kind: str
    frame: Any
    detection: dict[str, object] | None = None


def _plain_vietnamese(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    return "".join(
        character for character in normalized if unicodedata.category(character) != "Mn"
    )


class SemanticWorker:
    """Run OCR, VQA and microphone tasks without blocking safety inference."""

    def __init__(
        self,
        system: Any,
        logger: SessionLogger,
        *,
        default_question: str,
        recorder_factory: Callable[[], Any] | None = None,
        transcriber_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.system = system
        self.logger = logger
        self.default_question = default_question
        self.recorder_factory = recorder_factory
        self.transcriber_factory = transcriber_factory
        self._commands: queue.Queue[SemanticCommand | None] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._busy = False
        self._status = "Sẵn sàng"
        self._last_result: dict[str, object] | None = None
        self._recorder: Any | None = None
        self._transcriber: Any | None = None
        self._thread = threading.Thread(
            target=self._run, name="secondeye-semantic-worker", daemon=True
        )
        self._thread.start()

    def submit(self, command: SemanticCommand) -> bool:
        with self._lock:
            if self._busy or not self._commands.empty():
                self._status = "Tác vụ ngữ nghĩa đang chạy"
                return False
        try:
            self._commands.put_nowait(command)
            return True
        except queue.Full:
            return False

    def snapshot(self) -> tuple[bool, str, dict[str, object] | None]:
        with self._lock:
            return self._busy, self._status, self._last_result

    def close(self) -> None:
        try:
            self._commands.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    def _set_status(
        self,
        status: str,
        *,
        busy: bool,
        result: dict[str, object] | None = None,
    ) -> None:
        with self._lock:
            self._status = status
            self._busy = busy
            if result is not None:
                self._last_result = result

    def _microphone(self, command: SemanticCommand) -> dict[str, object]:
        if self.recorder_factory is None or self.transcriber_factory is None:
            raise RuntimeError("Push-to-talk chưa được cấu hình")
        self._recorder = self._recorder or self.recorder_factory()
        self._set_status("Đang nghe...", busy=True)
        with tempfile.TemporaryDirectory(prefix="secondeye_audio_") as directory:
            audio_path = self._recorder.record(Path(directory) / "command.wav")
            self._set_status("Đang nhận dạng giọng nói...", busy=True)
            self._transcriber = self._transcriber or self.transcriber_factory()
            stt = self._transcriber.transcribe(audio_path)
        transcript = str(stt.get("transcript", "")).strip()
        if not transcript:
            self.system.announce(
                "Tôi không nghe rõ yêu cầu. Hãy thử lại.", AlertPriority.ERROR
            )
            return {"stt": stt, "intent": "unknown", "abstained": True}
        plain = _plain_vietnamese(transcript)
        if any(word in plain for word in ("dung", "im lang", "stop")):
            self.system.stop_audio()
            intent_result: dict[str, object] = {"stopped": True}
            intent = "stop"
        elif "lap lai" in plain:
            intent_result = {"repeated": self.system.repeat_audio()}
            intent = "repeat"
        elif any(word in plain for word in ("doc chu", "doc van ban", "doc cho")):
            intent_result = self.system.read_text(command.frame)
            intent = "ocr"
        elif any(word in plain for word in ("mo ta", "xung quanh", "khung canh")):
            intent_result = self.system.describe_scene(
                command.frame, detection_result=command.detection
            )
            intent = "scene"
        else:
            intent_result = self.system.ask(command.frame, transcript)
            intent = "vqa"
        return {"stt": stt, "intent": intent, "result": intent_result}

    def _execute(self, command: SemanticCommand) -> dict[str, object]:
        if command.kind == "ocr":
            return self.system.read_text(command.frame)
        if command.kind == "scene":
            return self.system.describe_scene(
                command.frame, detection_result=command.detection
            )
        if command.kind == "vqa":
            return self.system.ask(command.frame, self.default_question)
        if command.kind == "microphone":
            return self._microphone(command)
        raise ValueError(f"Tác vụ không được hỗ trợ: {command.kind}")

    def _run(self) -> None:
        while True:
            command = self._commands.get()
            if command is None:
                return
            self._set_status(f"Đang chạy {command.kind}...", busy=True)
            self.logger.log("semantic_started", {"kind": command.kind})
            try:
                result = self._execute(command)
            except Exception as exc:
                result = {
                    "success": False,
                    "state": SystemState.ERROR.value,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
                self.logger.log("semantic_error", result, success=False)
                self.system.announce(
                    "Tác vụ vừa yêu cầu gặp lỗi. Hãy kiểm tra màn hình.",
                    AlertPriority.ERROR,
                )
                self._set_status(f"Lỗi: {exc}", busy=False, result=result)
            else:
                self.logger.log(command.kind, result)
                self._set_status(f"Hoàn tất {command.kind}", busy=False, result=result)


def run_mvp_demo(
    args: Any,
    *,
    cv2: Any,
    system: Any,
    recorder_factory: Callable[[], Any],
    transcriber_factory: Callable[[], Any],
) -> Path:
    logger = SessionLogger(args.log)
    logger.log(
        "session_started",
        {
            "mode": "pretrained_mvp",
            "camera": args.camera,
            "depth": args.depth,
            "semantic_device": args.semantic_device,
            "controls": "o:ocr s:scene v:vqa m:microphone r:repeat x:stop q:quit",
        },
    )
    system.detector.warmup()
    capture = LatestFrameCapture(
        cv2,
        args.camera,
        width=args.width,
        height=args.height,
        target_fps=args.camera_fps,
    ).start()
    runtime = AsyncVisionRuntime(
        system,
        capture.frames,
        detection_fps=args.detection_fps,
        depth_fps=args.depth_fps,
        max_depth_age_seconds=args.max_depth_age,
    ).start()
    semantic = SemanticWorker(
        system,
        logger,
        default_question=args.question,
        recorder_factory=recorder_factory,
        transcriber_factory=transcriber_factory,
    )
    system.announce("SecondEye đã sẵn sàng.", AlertPriority.INFO)
    text_renderer = UnicodeTextRenderer()
    window = "SecondEye MVP"
    last_logged_frame = -1
    display_fps = 0.0
    previous_display: float | None = None
    demo_started = time.monotonic()
    try:
        while True:
            packet = capture.frames.latest(copy_frame=True)
            if packet is None:
                time.sleep(0.01)
                continue
            now = time.monotonic()
            if args.max_seconds is not None and now - demo_started >= args.max_seconds:
                break
            if previous_display is not None and now > previous_display:
                instant = 1.0 / (now - previous_display)
                display_fps = (
                    instant if display_fps == 0.0 else 0.9 * display_fps + 0.1 * instant
                )
            previous_display = now
            payload = runtime.latest()
            fresh = bool(
                payload is not None
                and now - float(payload["completed_at"]) <= args.overlay_max_age
            )
            detections = (
                payload["detection"]["detections"]
                if payload is not None and fresh
                else []
            )
            overlays = draw_detection_overlays(cv2, packet.frame, detections)
            if payload is not None and int(payload["frame_id"]) != last_logged_frame:
                last_logged_frame = int(payload["frame_id"])
                logger.log("vision", payload)
            busy, semantic_status, _ = semantic.snapshot()
            state = "WARMING_UP" if payload is None else str(payload["state"])
            depth_status = "tắt"
            if args.depth:
                depth_status = (
                    "đang chờ"
                    if payload is None or payload.get("depth") is None
                    else f"{runtime.measured_depth_fps:.1f}Hz"
                )
            status = (
                f"{localize_state(state)} | hiển thị {display_fps:.1f} | "
                f"camera {capture.measured_fps:.1f} | nhận diện "
                f"{runtime.measured_detection_fps:.1f} | độ sâu {depth_status}"
            )
            overlays.extend(
                [
                    (status, (12, 8), (0, 255, 255), 22),
                    (
                        semantic_status,
                        (12, 38),
                        (255, 255, 0) if busy else (255, 255, 255),
                        20,
                    ),
                    (
                        "o đọc chữ | s mô tả | v hỏi ảnh | m nói | r lặp lại | "
                        "x dừng | q thoát",
                        (12, packet.frame.shape[0] - 28),
                        (255, 255, 255),
                        17,
                    ),
                ]
            )
            text_renderer.draw_bgr(packet.frame, overlays)
            cv2.imshow(window, packet.frame)
            key = cv2.waitKey(max(1, int(1000.0 / args.display_fps))) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("x"):
                system.stop_audio()
                logger.log("audio_stopped", {})
            elif key == ord("r"):
                logger.log("audio_repeat", {"accepted": system.repeat_audio()})
            elif key in (ord("o"), ord("s"), ord("v"), ord("m")):
                kind = {
                    ord("o"): "ocr",
                    ord("s"): "scene",
                    ord("v"): "vqa",
                    ord("m"): "microphone",
                }[key]
                accepted = semantic.submit(
                    SemanticCommand(
                        kind,
                        packet.frame.copy(),
                        None if payload is None or not fresh else payload["detection"],
                    )
                )
                logger.log("command", {"kind": kind, "accepted": accepted})
    finally:
        semantic.close()
        runtime.stop()
        capture.stop()
        system.stop_audio()
        system.close()
        cv2.destroyAllWindows()
        logger.log("session_ended", {})
    return logger.path
