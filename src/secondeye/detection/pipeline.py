"""Local YOLO26 CLI evolved from the original YOLO11 Colab notebook.

The commands validate local data, train, evaluate, export a verified ONNX
artifact, run image inference, and open a private local webcam window.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG_PATH, DetectionPipelineConfig, load_detection_config
from .dataset import (
    dataset_fingerprint,
    safe_extract_dataset,
    stats_as_dict,
    validate_dataset,
    write_dataset_yaml,
)
from .model import ObjectObstacleDetector
from .runtime import (
    ensure_class_schema,
    environment_manifest,
    file_sha256,
    git_commit,
    require_detection_runtime,
    seed_everything,
    select_device,
    write_json,
)


def _run_id(prefix: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}_{stamp}"


def _dataset_root(args: argparse.Namespace, config: DetectionPipelineConfig) -> Path:
    value = getattr(args, "dataset", None)
    return (value or config.paths.dataset_root).expanduser().resolve()


def _config_snapshot(config: DetectionPipelineConfig) -> dict[str, object]:
    return {
        "source": str(config.source_path),
        "sha256": file_sha256(config.source_path),
        "model": asdict(config.model),
        "pretrained_coco": asdict(config.pretrained_coco),
        "training": asdict(config.training),
        "export": asdict(config.export),
        "paths": {name: str(value) for name, value in asdict(config.paths).items()},
        "class_names": list(config.class_names),
        "candidate_classes": sorted(config.candidate_classes),
        "central_zone_fraction": config.central_zone_fraction,
    }


def _metric_summary(metrics: Any, split: str) -> dict[str, object]:
    box = metrics.box
    return {
        "split": split,
        "mAP50_95": float(box.map),
        "mAP50": float(box.map50),
        "mAP75": float(box.map75),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "per_class_mAP50_95": [float(value) for value in box.maps],
        "save_dir": str(metrics.save_dir),
    }


def _adapt_coco_result_to_second_eye(
    result: Any, config: DetectionPipelineConfig
) -> Any:
    """Keep explicit COCO mappings only; never infer unsupported SecondEye classes."""
    if result.boxes is None:
        return result
    mapping = dict(config.pretrained_coco.class_mapping)
    thresholds = dict(config.pretrained_coco.class_thresholds)
    display_names = dict(result.names)
    keep: list[int] = []
    for index, box in enumerate(result.boxes):
        class_id = int(box.cls.item())
        source_label = str(result.names[class_id])
        canonical_label = mapping.get(source_label)
        if canonical_label is None:
            continue
        threshold = thresholds[canonical_label]
        if float(box.conf.item()) >= threshold:
            keep.append(index)
        display_names[class_id] = canonical_label
    result.boxes = result.boxes[keep]
    result.names = display_names
    return result


def _pretrained_inference_floor(config: DetectionPipelineConfig) -> float:
    thresholds = [
        *(value for _, value in config.pretrained_coco.class_thresholds),
    ]
    return min(thresholds)


def _verify_onnx(path: Path, config: DetectionPipelineConfig) -> None:
    try:
        import numpy as np
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            'Thiếu dependency ONNX. Chạy: python -m pip install ".[detection]"'
        ) from exc
    _, _, yolo_class = require_detection_runtime()
    onnx.checker.check_model(onnx.load(str(path)))
    onnx_model = yolo_class(str(path), task="detect")
    ensure_class_schema(onnx_model.names, config.class_names)
    dummy = np.zeros(
        (config.model.image_size, config.model.image_size, 3), dtype=np.uint8
    )
    results = onnx_model.predict(
        source=dummy,
        imgsz=config.model.image_size,
        conf=config.model.confidence_threshold,
        iou=config.model.iou_threshold,
        device="cpu",
        verbose=False,
    )
    if len(results) != 1:
        raise RuntimeError(f"ONNX smoke test cần một result, nhận {len(results)}")


def _export_onnx(model: Any, config: DetectionPipelineConfig) -> Path:
    exported = Path(
        model.export(
            format="onnx",
            imgsz=config.model.image_size,
            dynamic=config.export.dynamic,
            simplify=config.export.simplify,
            opset=config.export.opset,
            device="cpu",
        )
    ).resolve()
    if not exported.is_file():
        raise FileNotFoundError(f"Ultralytics không tạo ONNX tại {exported}")
    _verify_onnx(exported, config)
    return exported


def _artifact_entry(path: Path) -> dict[str, object]:
    return {"sha256": file_sha256(path), "bytes": path.stat().st_size}


def command_prepare(args: argparse.Namespace, config: DetectionPipelineConfig) -> None:
    destination = (args.destination or config.paths.dataset_root).expanduser().resolve()
    dataset_root = safe_extract_dataset(args.archive, destination)
    print(f"Dataset đã giải nén an toàn tại: {dataset_root}")
    print("Chạy tiếp lệnh validate trước khi train.")


def command_validate(args: argparse.Namespace, config: DetectionPipelineConfig) -> None:
    dataset_root = _dataset_root(args, config)
    stats = validate_dataset(dataset_root, config.class_names)
    payload = {
        "dataset_root": str(dataset_root),
        "class_names": list(config.class_names),
        "splits": stats_as_dict(stats),
    }
    if args.output:
        write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def command_demo(args: argparse.Namespace, config: DetectionPipelineConfig) -> None:
    """Run the pretrained COCO model without presenting it as a SecondEye model."""
    cv2, torch, yolo_class = require_detection_runtime()
    source = args.source.expanduser().resolve()
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {source}")
    device = select_device(config.model.device, torch)
    model = yolo_class(config.model.base_weights)
    results = model.predict(
        source=image,
        conf=_pretrained_inference_floor(config),
        iou=config.model.iou_threshold,
        imgsz=config.model.image_size,
        device=device,
        verbose=False,
    )
    if len(results) != 1:
        raise RuntimeError(f"Demo cần một result, nhận {len(results)}")
    result = _adapt_coco_result_to_second_eye(results[0], config)
    detections = []
    canonical_ids = {name: index for index, name in enumerate(config.class_names)}
    if result.boxes is not None:
        for box in result.boxes:
            source_class_id = int(box.cls.item())
            label = str(result.names[source_class_id])
            detections.append(
                {
                    "class_id": canonical_ids[label],
                    "source_class_id": source_class_id,
                    "label": label,
                    "confidence": round(float(box.conf.item()), 4),
                    "bbox_xyxy": [
                        round(float(value), 2) for value in box.xyxy[0].cpu().tolist()
                    ],
                }
            )
    payload = {
        "schema_version": "1.0",
        "result_type": "pretrained_coco_indoor_integration_baseline",
        "model": config.model.base_weights,
        "device": device,
        "source_class_count": len(result.names),
        "supported_second_eye_classes": [
            target for _, target in config.pretrained_coco.class_mapping
        ],
        "unsupported_second_eye_classes": list(
            config.pretrained_coco.unsupported_second_eye_classes
        ),
        "detections": detections,
        "warning": (
            f"Đây là {config.model.base_weights} pretrained COCO với schema "
            "indoor_coco_baseline_v1; không phải model an toàn indoor đã fine-tune."
        ),
    }
    if args.output_json:
        write_json(args.output_json, payload)
    if args.output_image:
        output_image = args.output_image.expanduser().resolve()
        output_image.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_image), result.plot()):
            raise RuntimeError(f"Không ghi được ảnh: {output_image}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def command_train(args: argparse.Namespace, config: DetectionPipelineConfig) -> None:
    _, torch, yolo_class = require_detection_runtime()
    dataset_root = _dataset_root(args, config)
    stats = validate_dataset(dataset_root, config.class_names)
    run_id = args.name or _run_id("yolo26m_obstacles")
    staging_yaml = config.paths.runs_root / "dataset_configs" / f"{run_id}.yaml"
    dataset_yaml = write_dataset_yaml(
        dataset_root,
        config.class_names,
        staging_yaml,
        include_test=stats["test"].images > 0,
    )
    device = select_device(config.model.device, torch)
    seed_everything(config.training.seed, torch)
    config.paths.runs_root.mkdir(parents=True, exist_ok=True)

    model = yolo_class(config.model.base_weights)
    model.train(
        data=str(dataset_yaml),
        epochs=config.training.epochs,
        imgsz=config.model.image_size,
        batch=config.training.batch_size,
        device=device,
        workers=config.training.workers,
        patience=config.training.patience,
        pretrained=True,
        seed=config.training.seed,
        deterministic=config.training.deterministic,
        project=str(config.paths.runs_root),
        name=run_id,
        exist_ok=False,
        plots=True,
        cache=False,
    )
    if model.trainer is None:
        raise RuntimeError("Ultralytics không trả trainer sau khi train")
    run_dir = Path(model.trainer.save_dir).resolve()
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    if not best_pt.is_file():
        raise FileNotFoundError(f"Train hoàn tất nhưng thiếu {best_pt}")

    best_model = yolo_class(str(best_pt))
    ensure_class_schema(best_model.names, config.class_names)
    evaluation_split = "test" if stats["test"].images else "val"
    metrics = best_model.val(
        data=str(dataset_yaml),
        split=evaluation_split,
        imgsz=config.model.image_size,
        device=device,
        plots=True,
        project=str(config.paths.runs_root),
        name=f"{run_id}_{evaluation_split}",
    )
    metric_summary = _metric_summary(metrics, evaluation_split)
    exported_onnx = None if args.no_export else _export_onnx(best_model, config)

    artifact_dir = config.paths.artifact_root / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    copied: dict[str, Path] = {}
    for name, source in {
        "best.pt": best_pt,
        "last.pt": last_pt if last_pt.is_file() else None,
        "dataset.yaml": dataset_yaml,
        "model.onnx": exported_onnx,
    }.items():
        if source is None:
            continue
        destination = artifact_dir / name
        shutil.copy2(source, destination)
        copied[name] = destination
    write_json(artifact_dir / "metrics.json", metric_summary)
    copied["metrics.json"] = artifact_dir / "metrics.json"

    manifest = {
        "schema_version": "1.0",
        "artifact_type": "secondeye_object_obstacle_detector",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "git_commit": git_commit(),
        "config": _config_snapshot(config),
        "environment": environment_manifest(device),
        "dataset": {
            "root_name": dataset_root.name,
            "fingerprint": dataset_fingerprint(dataset_root),
            "splits": stats_as_dict(stats),
        },
        "evaluation": metric_summary,
        "artifacts": {name: _artifact_entry(path) for name, path in copied.items()},
        "limitations": [
            "Nếu evaluation.split là val, metric chưa phải đánh giá test độc lập.",
            "Detection 2D không xác nhận khoảng cách; hệ thống cảnh báo cần module depth.",
            "Model chỉ nên dùng như nguyên mẫu nghiên cứu hỗ trợ, không thay thế công cụ định hướng.",
        ],
    }
    write_json(artifact_dir / "manifest.json", manifest)
    print(f"Train hoàn tất: {run_dir}")
    print(f"Artifact đã lưu: {artifact_dir}")


def command_evaluate(args: argparse.Namespace, config: DetectionPipelineConfig) -> None:
    _, torch, yolo_class = require_detection_runtime()
    dataset_root = _dataset_root(args, config)
    stats = validate_dataset(dataset_root, config.class_names)
    split = args.split
    if split == "auto":
        split = "test" if stats["test"].images else "val"
    if split == "test" and not stats["test"].images:
        raise ValueError("Dataset không có test split")
    output_parent = (
        (args.output or Path("results/detection_evaluation.json")).resolve().parent
    )
    dataset_yaml = write_dataset_yaml(
        dataset_root,
        config.class_names,
        output_parent / "dataset.yaml",
        include_test=stats["test"].images > 0,
    )
    model = yolo_class(str(args.model.expanduser().resolve()))
    ensure_class_schema(model.names, config.class_names)
    device = select_device(config.model.device, torch)
    metrics = model.val(
        data=str(dataset_yaml),
        split=split,
        imgsz=config.model.image_size,
        device=device,
        plots=True,
    )
    summary = _metric_summary(metrics, split)
    output = args.output or Path("results/detection_evaluation.json")
    write_json(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_export(args: argparse.Namespace, config: DetectionPipelineConfig) -> None:
    _, _, yolo_class = require_detection_runtime()
    model_path = args.model.expanduser().resolve()
    model = yolo_class(str(model_path))
    ensure_class_schema(model.names, config.class_names)
    onnx_path = _export_onnx(model, config)
    output_dir = (
        (args.output_dir or config.paths.artifact_root / _run_id("export"))
        .expanduser()
        .resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    copied_pt = output_dir / "best.pt"
    copied_onnx = output_dir / "model.onnx"
    shutil.copy2(model_path, copied_pt)
    shutil.copy2(onnx_path, copied_onnx)
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": _config_snapshot(config),
        "artifacts": {
            "best.pt": _artifact_entry(copied_pt),
            "model.onnx": _artifact_entry(copied_onnx),
        },
        "onnx_verified": True,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(f"ONNX đã kiểm tra và lưu tại: {copied_onnx}")


def command_predict(args: argparse.Namespace, config: DetectionPipelineConfig) -> None:
    cv2, _, _ = require_detection_runtime()
    source = args.source.expanduser().resolve()
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {source}")
    detector = ObjectObstacleDetector(args.model, config)
    detector.warmup()
    payload, annotated = detector.predict_and_render_bgr(image)
    if args.output_json:
        write_json(args.output_json, payload)
    if args.output_image:
        output_image = args.output_image.expanduser().resolve()
        output_image.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_image), annotated):
            raise RuntimeError(f"Không ghi được ảnh: {output_image}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def command_camera(args: argparse.Namespace, config: DetectionPipelineConfig) -> None:
    cv2, _, _ = require_detection_runtime()
    detector = ObjectObstacleDetector(args.model, config)
    detector.warmup()
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Không mở được camera {args.camera}. Hãy cấp quyền Camera cho Terminal/Python."
        )
    window_name = "SecondEye local camera - q/Esc de thoat"
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Camera không trả frame hợp lệ")
            payload, annotated = detector.predict_and_render_bgr(frame)
            candidate_count = sum(
                bool(item["obstacle_candidate"]) for item in payload["detections"]
            )
            cv2.putText(
                annotated,
                f"{payload['latency_ms']:.1f} ms | candidates: {candidate_count}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def command_camera_demo(
    args: argparse.Namespace, config: DetectionPipelineConfig
) -> None:
    """Run the configured pretrained COCO model on a local camera without training."""
    cv2, torch, yolo_class = require_detection_runtime()
    device = select_device(config.model.device, torch)
    model = yolo_class(config.model.base_weights, task="detect")
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Không mở được camera {args.camera}. Hãy cấp quyền Camera cho Terminal/Python."
        )
    warning = (
        f"{config.model.base_weights} pretrained đang dùng đủ 15 lớp của "
        "indoor_coco_baseline_v1; đây không phải model an toàn indoor đã fine-tune."
    )
    print(f"CẢNH BÁO: {warning}")
    if config.pretrained_coco.unsupported_second_eye_classes:
        print(
            "CHƯA HỖ TRỢ: "
            + ", ".join(config.pretrained_coco.unsupported_second_eye_classes)
        )
    window_name = "SecondEye YOLO26 COCO demo - q/Esc de thoat"
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Camera không trả frame hợp lệ")
            results = model.predict(
                source=frame,
                conf=_pretrained_inference_floor(config),
                iou=config.model.iou_threshold,
                imgsz=config.model.image_size,
                device=device,
                verbose=False,
            )
            if len(results) != 1:
                raise RuntimeError(f"Demo camera cần một result, nhận {len(results)}")
            result = _adapt_coco_result_to_second_eye(results[0], config)
            detection_count = 0 if result.boxes is None else len(result.boxes)
            latency_ms = float(result.speed.get("inference", 0.0))
            annotated = result.plot()
            cv2.putText(
                annotated,
                f"COCO DEMO | {latency_ms:.1f} ms | detections: {detection_count}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Giải nén dataset ZIP an toàn")
    prepare.add_argument("--archive", type=Path, required=True)
    prepare.add_argument("--destination", type=Path)
    prepare.set_defaults(handler=command_prepare)

    validate = subparsers.add_parser(
        "validate", help="Kiểm tra dataset trước khi train"
    )
    validate.add_argument("--dataset", type=Path)
    validate.add_argument("--output", type=Path)
    validate.set_defaults(handler=command_validate)

    demo = subparsers.add_parser(
        "demo", help="Smoke test schema indoor COCO pretrained gồm 15 lớp"
    )
    demo.add_argument("--source", type=Path, required=True)
    demo.add_argument("--output-json", type=Path)
    demo.add_argument("--output-image", type=Path)
    demo.set_defaults(handler=command_demo)

    train = subparsers.add_parser(
        "train", help="Train, evaluate, export và đóng gói model"
    )
    train.add_argument("--dataset", type=Path)
    train.add_argument("--name")
    train.add_argument("--no-export", action="store_true")
    train.set_defaults(handler=command_train)

    evaluate = subparsers.add_parser("evaluate", help="Đánh giá best.pt trên val/test")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--dataset", type=Path)
    evaluate.add_argument("--split", choices=("auto", "val", "test"), default="auto")
    evaluate.add_argument("--output", type=Path)
    evaluate.set_defaults(handler=command_evaluate)

    export = subparsers.add_parser("export", help="Xuất và smoke-test ONNX")
    export.add_argument("--model", type=Path, required=True)
    export.add_argument("--output-dir", type=Path)
    export.set_defaults(handler=command_export)

    predict = subparsers.add_parser("predict", help="Nhận diện một ảnh local")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--source", type=Path, required=True)
    predict.add_argument("--output-json", type=Path)
    predict.add_argument("--output-image", type=Path)
    predict.set_defaults(handler=command_predict)

    camera = subparsers.add_parser("camera", help="Mở webcam local bằng OpenCV")
    camera.add_argument("--model", type=Path, required=True)
    camera.add_argument("--camera", type=int, default=0)
    camera.set_defaults(handler=command_camera)

    camera_demo = subparsers.add_parser(
        "camera-demo",
        help="Mở camera bằng schema indoor COCO pretrained, không cần train",
    )
    camera_demo.add_argument("--camera", type=int, default=0)
    camera_demo.set_defaults(handler=command_camera_demo)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_detection_config(args.config)
        args.handler(args, config)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Lỗi: {exc}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
