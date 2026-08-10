"""One local CLI for the SecondEye pretrained integration baseline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

from secondeye.detection.config import DEFAULT_CONFIG_PATH, load_detection_config
from secondeye.detection.model import PretrainedCocoDetector
from secondeye.detection.runtime import require_detection_runtime, write_json
from secondeye.multimodal import (
    DepthAnythingEstimator,
    MacOSTextToSpeech,
    PaddleOcrReader,
    PretrainedVisualQuestionAnswering,
    WhisperSpeechToText,
    macos_voice_available,
)

from .pipeline import SecondEyeSystem
from .camera import AsyncVisionRuntime, LatestFrameCapture


def _json_safe(payload: object) -> object:
    if isinstance(payload, dict):
        return {key: _json_safe(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_json_safe(value) for value in payload]
    tolist = getattr(payload, "tolist", None)
    return tolist() if callable(tolist) else payload


def _build_system(args: argparse.Namespace) -> SecondEyeSystem:
    config = load_detection_config(args.config)
    detector = PretrainedCocoDetector(config)
    depth = (
        DepthAnythingEstimator(device=config.model.device)
        if getattr(args, "depth", False)
        else None
    )
    ocr = PaddleOcrReader() if getattr(args, "ocr", False) else None
    vqa = PretrainedVisualQuestionAnswering() if getattr(args, "question", None) else None
    tts = (
        None
        if getattr(args, "no_tts", False)
        else MacOSTextToSpeech(
            voice=getattr(args, "voice", "Linh"),
            rate=getattr(args, "speech_rate", 165),
        )
    )
    return SecondEyeSystem(detector=detector, depth=depth, ocr=ocr, vqa=vqa, tts=tts)


def command_doctor(args: argparse.Namespace) -> None:
    del args
    modules = {
        "detection": all(importlib.util.find_spec(name) for name in ("ultralytics", "cv2", "torch")),
        "depth_vqa_stt": importlib.util.find_spec("transformers") is not None,
        "ocr": importlib.util.find_spec("paddleocr") is not None,
        "tts_macos": sys.platform == "darwin",
        "tts_voice_linh_vi_vn": macos_voice_available("Linh"),
    }
    print(json.dumps({"success": all(modules.values()), "modules": modules}, indent=2))


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
    window = "SecondEye async camera - q:quit x:stop"
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
                display_fps = instant if display_fps == 0.0 else 0.9 * display_fps + 0.1 * instant
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
            for detection in detections:
                x1, y1, x2, y2 = (int(value) for value in detection["bbox_xyxy"])
                color = (0, 0, 255) if detection.get("depth_zone") == "near" else (0, 200, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                caption = f"{detection['label']} {detection['confidence']:.2f}"
                if detection.get("depth_zone"):
                    caption += f" {detection['depth_zone']}"
                cv2.putText(
                    annotated,
                    caption,
                    (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            state = "WARMING_UP" if payload is None else str(payload["state"])
            detection_fps = runtime.measured_detection_fps
            depth_status = "off"
            if args.depth:
                depth_status = (
                    "waiting"
                    if payload is None or payload.get("depth") is None
                    else f"{runtime.measured_depth_fps:.1f}Hz"
                )
            status = (
                f"{state} | display {display_fps:.1f} | camera {capture.measured_fps:.1f} "
                f"| det {detection_fps:.1f} | depth {depth_status}"
            )
            cv2.putText(
                annotated,
                status,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
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
        cv2.destroyAllWindows()


def command_transcribe(args: argparse.Namespace) -> None:
    result = WhisperSpeechToText().transcribe(args.audio)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_speech_test(args: argparse.Namespace) -> None:
    speaker = MacOSTextToSpeech(voice=args.voice, rate=args.speech_rate)
    speaker.speak(args.text, interrupt=True)
    speaker.wait()


def _add_tts_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--voice", default="Linh", help="Giọng macOS, mặc định Linh vi_VN")
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

    transcribe = subparsers.add_parser("transcribe", help="STT một file audio bằng Whisper")
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
