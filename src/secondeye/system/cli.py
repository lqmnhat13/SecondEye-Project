"""One local CLI for the complete SecondEye pretrained MVP."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import threading
import sys
import time
from pathlib import Path
from typing import Any

from secondeye.detection.config import DEFAULT_CONFIG_PATH, load_detection_config
from secondeye.detection.model import PretrainedCocoDetector
from secondeye.detection.runtime import require_detection_runtime, write_json
from secondeye.multimodal import (
    DepthAnythingEstimator,
    FFmpegMicrophoneRecorder,
    MacOSTextToSpeech,
    PaddleOcrReader,
    PretrainedEnglishVietnameseTranslator,
    PretrainedVisualQuestionAnswering,
    WhisperSpeechToText,
    macos_voice_available,
)

from .audio import PriorityAudioManager
from .demo import run_mvp_demo
from .localization import localize_state
from .overlay import UnicodeTextRenderer, draw_detection_overlays
from .pipeline import SecondEyeSystem
from .camera import AsyncVisionRuntime, LatestFrameCapture
from .session import json_safe


DEMO_SEMANTIC_DEVICE = "cpu"


class _LazyOcrReader:
    def __init__(self) -> None:
        self._value: PaddleOcrReader | None = None
        self._lock = threading.Lock()

    def read_bgr(self, image: Any) -> dict[str, object]:
        with self._lock:
            self._value = self._value or PaddleOcrReader()
            value = self._value
        return value.read_bgr(image)


class _LazyVqa:
    def __init__(self, *, device: str) -> None:
        self.device = device
        self._value: PretrainedVisualQuestionAnswering | None = None
        self._lock = threading.Lock()

    def ask_bgr(self, image: Any, question: str) -> dict[str, object]:
        with self._lock:
            self._value = self._value or PretrainedVisualQuestionAnswering(
                device=self.device
            )
            value = self._value
        return value.ask_bgr(image, question)


def _json_safe(payload: object) -> object:
    return json_safe(payload)


def _build_system(args: argparse.Namespace) -> SecondEyeSystem:
    config = load_detection_config(args.config)
    detector = PretrainedCocoDetector(config)
    depth = (
        DepthAnythingEstimator(device=config.model.device)
        if getattr(args, "depth", False)
        else None
    )
    lazy_semantic = bool(getattr(args, "lazy_semantic", False))
    semantic_device = str(getattr(args, "semantic_device", config.model.device))
    ocr = (
        (_LazyOcrReader() if lazy_semantic else PaddleOcrReader())
        if getattr(args, "ocr", False)
        else None
    )
    vqa = (
        (
            _LazyVqa(device=semantic_device)
            if lazy_semantic
            else PretrainedVisualQuestionAnswering(device=semantic_device)
        )
        if getattr(args, "question", None)
        else None
    )
    translator = (
        PretrainedEnglishVietnameseTranslator(device=semantic_device)
        if vqa is not None
        else None
    )
    tts = (
        None
        if getattr(args, "no_tts", False)
        else MacOSTextToSpeech(
            voice=getattr(args, "voice", "Linh"), rate=getattr(args, "speech_rate", 165)
        )
    )
    if tts is not None and getattr(args, "priority_audio", False):
        tts = PriorityAudioManager(tts)
    return SecondEyeSystem(
        detector=detector,
        depth=depth,
        ocr=ocr,
        vqa=vqa,
        translator=translator,
        tts=tts,
    )


def command_doctor(args: argparse.Namespace) -> None:
    del args
    groups = {
        "detection": ("numpy", "cv2", "torch", "ultralytics"),
        "depth_vqa_stt": ("torch", "transformers"),
        "translation_en_vi": ("sentencepiece",),
        "ocr": ("paddleocr",),
    }
    modules: dict[str, bool] = {}
    details: dict[str, str] = {}
    for group, names in groups.items():
        missing = [name for name in names if importlib.util.find_spec(name) is None]
        if missing:
            modules[group] = False
            details[group] = "missing: " + ", ".join(missing)
            continue
        command = [sys.executable, "-c", ";".join(f"import {name}" for name in names)]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=20
            )
        except subprocess.TimeoutExpired:
            modules[group] = False
            details[group] = "import timeout"
        else:
            modules[group] = completed.returncode == 0
            error_lines = completed.stderr.strip().splitlines()
            details[group] = (
                "ok"
                if completed.returncode == 0
                else (error_lines[-1] if error_lines else "import failed")
            )
    modules.update(
        {
            "tts_macos": sys.platform == "darwin",
            "tts_voice_linh_vi_vn": macos_voice_available("Linh"),
        }
    )
    details.update(
        {
            "tts_macos": "ok" if modules["tts_macos"] else "requires macOS",
            "tts_voice_linh_vi_vn": "ok"
            if modules["tts_voice_linh_vi_vn"]
            else "voice Linh unavailable",
        }
    )
    print(
        json.dumps(
            {"success": all(modules.values()), "modules": modules, "details": details},
            indent=2,
        )
    )


def command_image(args: argparse.Namespace) -> None:
    cv2, _, _ = require_detection_runtime()
    source = args.source.expanduser().resolve()
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {source}")
    system = _build_system(args)
    payload: dict[str, Any] = {
        "frame": system.process_frame(image, with_depth=args.depth)
    }
    if args.ocr:
        payload["ocr"] = system.read_text(image)
    if args.question:
        payload["vqa"] = system.ask(image, args.question)
    safe = _json_safe(payload)
    if args.output:
        write_json(args.output, safe)
    print(json.dumps(safe, ensure_ascii=False, indent=2))


def command_camera(args: argparse.Namespace) -> None:
    if args.display_fps <= 0 or args.overlay_max_age <= 0:
        raise ValueError("display-fps và overlay-max-age phải dương")
    cv2, _, _ = require_detection_runtime()
    system = _build_system(args)
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
    text_renderer = UnicodeTextRenderer()
    window = "SecondEye camera"
    display_fps = 0.0
    previous_display = None
    try:
        while True:
            packet = capture.frames.latest(copy_frame=True)
            if packet is None:
                time.sleep(0.01)
                continue
            now = time.monotonic()
            if previous_display is not None and now > previous_display:
                instant = 1.0 / (now - previous_display)
                display_fps = (
                    instant if display_fps == 0.0 else 0.9 * display_fps + 0.1 * instant
                )
            previous_display = now
            payload = runtime.latest()
            annotated = packet.frame
            result_is_fresh = bool(
                payload is not None
                and now - float(payload["completed_at"]) <= args.overlay_max_age
            )
            detections = (
                payload["detection"]["detections"]
                if payload is not None and result_is_fresh
                else []
            )
            overlays = draw_detection_overlays(cv2, annotated, detections)
            state = "WARMING_UP" if payload is None else str(payload["state"])
            detection_fps = runtime.measured_detection_fps
            depth_status = "tắt"
            if args.depth:
                depth_status = (
                    "đang chờ"
                    if payload is None or payload.get("depth") is None
                    else f"{runtime.measured_depth_fps:.1f}Hz"
                )
            status = (
                f"{localize_state(state)} | hiển thị {display_fps:.1f} | "
                f"camera {capture.measured_fps:.1f} | nhận diện {detection_fps:.1f} "
                f"| độ sâu {depth_status}"
            )
            overlays.append((status, (12, 8), (0, 255, 255), 22))
            text_renderer.draw_bgr(annotated, overlays)
            cv2.imshow(window, annotated)
            key = cv2.waitKey(max(1, int(1000.0 / args.display_fps))) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("x"):
                system.stop_audio()
    finally:
        runtime.stop()
        capture.stop()
        system.stop_audio()
        system.close()
        cv2.destroyAllWindows()


def command_demo(args: argparse.Namespace) -> None:
    if args.display_fps <= 0 or args.overlay_max_age <= 0:
        raise ValueError("display-fps và overlay-max-age phải dương")
    if args.max_seconds is not None and args.max_seconds <= 0:
        raise ValueError("max-seconds phải dương")
    cv2, _, _ = require_detection_runtime()
    system = _build_system(args)
    path = run_mvp_demo(
        args,
        cv2=cv2,
        system=system,
        recorder_factory=lambda: FFmpegMicrophoneRecorder(
            device=args.microphone, duration_seconds=args.listen_seconds
        ),
        transcriber_factory=lambda: WhisperSpeechToText(device=args.semantic_device),
    )
    print(f"Session log: {path}")


def command_transcribe(args: argparse.Namespace) -> None:
    result = WhisperSpeechToText().transcribe(args.audio)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_speech_test(args: argparse.Namespace) -> None:
    speaker = MacOSTextToSpeech(voice=args.voice, rate=args.speech_rate)
    speaker.speak(args.text, interrupt=True)
    speaker.wait()


def _add_tts_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument(
        "--voice", default="Linh", help="Giọng macOS, mặc định Linh vi_VN"
    )
    parser.add_argument("--speech-rate", type=int, default=165)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Kiểm tra runtime mà không tải model")
    doctor.set_defaults(handler=command_doctor)

    image = subparsers.add_parser("image", help="Chạy các module trên một ảnh local")
    image.add_argument("--source", type=Path, required=True)
    image.add_argument("--depth", action="store_true")
    image.add_argument("--ocr", action="store_true")
    image.add_argument("--question")
    _add_tts_arguments(image)
    image.add_argument("--output", type=Path)
    image.set_defaults(handler=command_image)

    camera = subparsers.add_parser("camera", help="Chạy camera Mac/iPhone end-to-end")
    camera.add_argument("--camera", type=int, default=0)
    camera.add_argument("--depth", action="store_true")
    camera.add_argument("--width", type=int, default=1280)
    camera.add_argument("--height", type=int, default=720)
    camera.add_argument("--camera-fps", type=float, default=30.0)
    camera.add_argument("--display-fps", type=float, default=30.0)
    camera.add_argument("--detection-fps", type=float, default=12.0)
    camera.add_argument("--depth-fps", type=float, default=3.0)
    camera.add_argument("--max-depth-age", type=float, default=0.50)
    camera.add_argument("--overlay-max-age", type=float, default=0.75)
    _add_tts_arguments(camera)
    camera.set_defaults(handler=command_camera, ocr=False, question=None)

    demo = subparsers.add_parser(
        "demo", help="Chạy MVP camera, OCR, scene, VQA và giọng nói"
    )
    demo.add_argument("--camera", type=int, default=0)
    demo.add_argument("--depth", action=argparse.BooleanOptionalAction, default=True)
    demo.add_argument("--width", type=int, default=1280)
    demo.add_argument("--height", type=int, default=720)
    demo.add_argument("--camera-fps", type=float, default=30.0)
    demo.add_argument("--display-fps", type=float, default=30.0)
    demo.add_argument("--detection-fps", type=float, default=12.0)
    demo.add_argument("--depth-fps", type=float, default=3.0)
    demo.add_argument("--max-depth-age", type=float, default=1.50)
    demo.add_argument("--overlay-max-age", type=float, default=1.50)
    demo.add_argument(
        "--question",
        default="What objects are directly in front of me?",
        help="Câu hỏi dùng khi nhấn v",
    )
    demo.add_argument("--microphone", default="1", help="Chỉ số audio AVFoundation")
    demo.add_argument("--listen-seconds", type=float, default=4.0)
    demo.add_argument(
        "--max-seconds",
        type=float,
        help="Tự thoát sau số giây đã cho; dùng cho smoke test",
    )
    demo.add_argument("--log", type=Path)
    _add_tts_arguments(demo)
    demo.set_defaults(
        handler=command_demo,
        ocr=True,
        lazy_semantic=True,
        priority_audio=True,
        semantic_device=DEMO_SEMANTIC_DEVICE,
    )

    transcribe = subparsers.add_parser(
        "transcribe", help="STT một file audio bằng Whisper"
    )
    transcribe.add_argument("--audio", type=Path, required=True)
    transcribe.set_defaults(handler=command_transcribe)

    speech_test = subparsers.add_parser("speech-test", help="Thử giọng TTS tiếng Việt")
    speech_test.add_argument(
        "--text", default="Xin chào, SecondEye đã sẵn sàng hỗ trợ bạn."
    )
    speech_test.add_argument("--voice", default="Linh")
    speech_test.add_argument("--speech-rate", type=int, default=165)
    speech_test.set_defaults(handler=command_speech_test)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Lỗi: {exc}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
