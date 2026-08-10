"""One local CLI for the SecondEye pretrained integration baseline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
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
)

from .pipeline import SecondEyeSystem


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
    depth = DepthAnythingEstimator(device=config.model.device) if args.depth else None
    ocr = PaddleOcrReader() if getattr(args, "ocr", False) else None
    vqa = PretrainedVisualQuestionAnswering() if getattr(args, "question", None) else None
    tts = None if args.no_tts else MacOSTextToSpeech()
    return SecondEyeSystem(detector=detector, depth=depth, ocr=ocr, vqa=vqa, tts=tts)


def command_doctor(args: argparse.Namespace) -> None:
    del args
    modules = {
        "detection": all(importlib.util.find_spec(name) for name in ("ultralytics", "cv2", "torch")),
        "depth_vqa_stt": importlib.util.find_spec("transformers") is not None,
        "ocr": importlib.util.find_spec("paddleocr") is not None,
        "tts_macos": sys.platform == "darwin",
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
    cv2, _, _ = require_detection_runtime()
    system = _build_system(args)
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Không mở được camera {args.camera}. Kiểm tra quyền Camera và Continuity Camera."
        )
    window = "SecondEye pretrained integration - q:quit x:stop"
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Camera không trả frame hợp lệ")
            payload = system.process_frame(frame, with_depth=args.depth)
            annotated = frame.copy()
            for detection in payload["detection"]["detections"]:
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
            cv2.putText(
                annotated,
                f"{payload['state']} | {payload['latency_ms']:.1f} ms | pretrained",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("x"):
                system.stop_audio()
    finally:
        capture.release()
        system.stop_audio()
        cv2.destroyAllWindows()


def command_transcribe(args: argparse.Namespace) -> None:
    result = WhisperSpeechToText().transcribe(args.audio)
    print(json.dumps(result, ensure_ascii=False, indent=2))


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
    image.add_argument("--no-tts", action="store_true")
    image.add_argument("--output", type=Path)
    image.set_defaults(handler=command_image)

    camera = subparsers.add_parser("camera", help="Chạy camera Mac/iPhone end-to-end")
    camera.add_argument("--camera", type=int, default=0)
    camera.add_argument("--depth", action="store_true")
    camera.add_argument("--no-tts", action="store_true")
    camera.set_defaults(handler=command_camera, ocr=False, question=None)

    transcribe = subparsers.add_parser("transcribe", help="STT một file audio bằng Whisper")
    transcribe.add_argument("--audio", type=Path, required=True)
    transcribe.set_defaults(handler=command_transcribe)
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
