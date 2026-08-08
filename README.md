# SecondEye

SecondEye là nguyên mẫu nghiên cứu đa phương thức hỗ trợ người khiếm thị nhận thức môi trường. MVP được khóa ở ba kịch bản: cảnh báo vật cản, đọc văn bản tiếng Việt và hiểu/hỏi đáp về cảnh.

> **Cảnh báo an toàn:** SecondEye chỉ là công cụ hỗ trợ nghiên cứu. Hệ thống không thay thế gậy trắng, chó dẫn đường hoặc thiết bị điều hướng chuyên dụng; không được dùng để tự điều hướng ngoài đường hay trong tình huống nguy hiểm.

## Trạng thái

- Giai đoạn 0: bản nháp phạm vi, Charter, kế hoạch 10 tuần và checklist bảo vệ đã có.
- Giai đoạn 1: đã xác minh 20 nguồn, hoàn thành literature matrix, BibTeX và bản tổng hợp research gap; chuẩn trích dẫn của trường còn cần xác nhận.
- Giai đoạn 2: đã có [data card, annotation/collection/split protocol](docs/data/README.md), CSV templates và leakage validator; dữ liệu thực tế chưa thu, số mẫu được chấp nhận hiện là 0.
- Pipeline YOLO11 từ notebook đã được chuyển thành CLI chạy local: validate dữ liệu,
  train, đánh giá val/test, kiểm tra ONNX, predict ảnh và webcam. Chưa có checkpoint
  12 lớp vì dữ liệu thực tế chưa được cung cấp/huấn luyện.
- OCR, depth, VQA, speech và demo end-to-end: chưa triển khai.

Theo dõi chi tiết tại [PROJECT_STATUS.md](PROJECT_STATUS.md). Nguồn yêu cầu chính là `output/pdf/Cam_nang_KLTN_SecondEye.pdf`.

## Kiến trúc đích

- **Luồng an toàn chạy thường xuyên:** camera -> detection -> depth -> đánh giá nguy cơ -> cảnh báo ngắn.
- **Luồng ngữ nghĩa theo yêu cầu:** ảnh + lệnh -> OCR hoặc VQA/mô tả cảnh -> phản hồi.
- **Một bộ điều phối âm thanh:** ưu tiên, cooldown, timeout và giải quyết xung đột cho toàn bộ TTS.

## Thiết lập tái lập

Yêu cầu đã kiểm tra: Python 3.11 trên macOS Apple Silicon. Các nền tảng khác là dự kiến cho đến khi được chạy kiểm chứng.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[dev,detection]"
```

Sau khi sửa mã nguồn, chạy lại `python -m pip install ".[dev,detection]"` trước khi dùng
console script để bản wheel trong `.venv` được cập nhật.

Chạy unit test không cần tải model:

```bash
pytest
```

Kiểm tra schema, consent state và leakage trong manifest dữ liệu:

```bash
secondeye-validate-manifest data/local/sample_manifest.csv \
  --config configs/data_protocol.toml --require-rows
```

## Pipeline YOLO11 chạy local

Notebook Colab cũ đã được thay bằng lệnh `secondeye-detection`; không còn Google Drive,
JavaScript camera, shell magic hay URL Gradio công khai. Cấu hình mặc định nằm tại
`configs/yolo11_obstacles.toml`.

```bash
python -m pip install ".[dev,detection]"

# Smoke test local bằng pretrained COCO; không phải model SecondEye 12 lớp
python scripts/fetch_smoke_asset.py
secondeye-detection demo --source data/samples/ultralytics_bus.jpg

# Nếu dữ liệu đang là ZIP (thư mục đích phải chưa tồn tại hoặc rỗng)
secondeye-detection prepare \
  --archive /duong/dan/secondeye_obstacles.zip \
  --destination data/local/secondeye_obstacles_import

# Dùng đúng dataset root được lệnh prepare in ra
secondeye-detection validate \
  --dataset data/local/secondeye_obstacles_import/secondeye_obstacles

# Train -> evaluate test nếu có, nếu không dùng val -> export/smoke-test ONNX -> package
secondeye-detection train \
  --dataset data/local/secondeye_obstacles_import/secondeye_obstacles
```

Artifact mỗi lần train được lưu riêng trong `artifacts/object_obstacle/<run_id>/`, gồm
`best.pt`, `last.pt`, `model.onnx`, `dataset.yaml`, `metrics.json` và `manifest.json`.
Xem hướng dẫn đầy đủ tại `docs/local_yolo11_pipeline.md`.

## Cấu trúc repository

```text
configs/                 cấu hình đã version hóa
data/raw/                dữ liệu gốc local, không commit mặc định
data/annotations/        nhãn local, không commit mặc định
data/templates/          manifest và schema CSV mẫu, không chứa dữ liệu thật
data/samples/            ảnh smoke test có thể tải lại bằng script
docs/                    Charter, hướng dẫn local và protocol nghiên cứu
output/pdf/              cẩm nang yêu cầu gốc
outputs/                 sản phẩm bàn giao theo từng giai đoạn
src/secondeye/           mã nguồn theo mô-đun
tests/                   unit test
```

## Quy tắc nghiên cứu

- Không dùng test set để chọn prompt, ngưỡng hoặc siêu tham số.
- Mỗi kết quả phải ghi model/version, config, seed, thiết bị, input hash và latency.
- Không lưu ảnh/âm thanh người dùng dài hạn nếu chưa có mục đích, đồng thuận và thời hạn lưu trữ rõ ràng.
- Mọi số liệu chưa chạy phải ghi là **dự kiến** hoặc **chỗ trống cần thí nghiệm**.

## Giấy phép cần chốt

Pipeline hiện dùng Ultralytics YOLO theo điều kiện AGPL-3.0. Phù hợp để khảo sát học thuật
khi tuân thủ điều kiện tương ứng, nhưng phải rà soát lại trước khi phát hành mã nguồn,
demo công khai hoặc thương mại hóa.
