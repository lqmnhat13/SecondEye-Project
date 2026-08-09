# SecondEye

SecondEye là nguyên mẫu nghiên cứu đa phương thức hỗ trợ người khiếm thị nhận thức môi trường. MVP được khóa ở ba kịch bản: cảnh báo vật cản, đọc văn bản tiếng Việt và hiểu/hỏi đáp về cảnh.

> **Cảnh báo an toàn:** SecondEye chỉ là công cụ hỗ trợ nghiên cứu. Hệ thống không thay thế gậy trắng, chó dẫn đường hoặc thiết bị điều hướng chuyên dụng; không được dùng để tự điều hướng ngoài đường hay trong tình huống nguy hiểm.

## Trạng thái

- Giai đoạn 0: bản nháp phạm vi, Charter, kế hoạch 10 tuần và checklist bảo vệ đã có.
- Giai đoạn 1: đã xác minh 20 nguồn, hoàn thành literature matrix, BibTeX và bản tổng hợp research gap; chuẩn trích dẫn của trường còn cần xác nhận.
- Giai đoạn 2: schema indoor v1 gồm 15 lớp đã khóa; dataset public v1.1 tại
  `data/local/indoor_dataset_v1_1` có **276 ảnh/603 bbox**, phủ đủ 15 lớp và mỗi
  lớp có ít nhất 20 bbox. YOLO validator, manifest audit và visual/privacy review
  đều đã qua. Dự án không tự chụp và không đưa dataset lên Git public.
- Pipeline hiện dùng YOLO26m và được chuyển từ notebook YOLO11 thành CLI chạy local: validate dữ liệu,
  train, đánh giá val/test, kiểm tra ONNX, predict ảnh và webcam. Chưa có checkpoint
  15 lớp vì bước fine-tune/evaluate chưa chạy trên dataset v1.1 vừa khóa.
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

## Pipeline YOLO26 chạy local

Notebook Colab cũ đã được thay bằng lệnh `secondeye-detection`; không còn Google Drive,
JavaScript camera, shell magic hay URL Gradio công khai. Pipeline hiện dùng
`yolo26m.pt`; cấu hình mặc định nằm tại `configs/yolo26_obstacles.toml`.

```bash
python -m pip install ".[dev,detection]"

# Smoke test local bằng YOLO26m pretrained COCO; không phải model SecondEye 15 lớp
python scripts/fetch_smoke_asset.py
secondeye-detection demo --source data/samples/ultralytics_bus.jpg

# Camera iPhone bằng YOLO26m pretrained, không cần train; trên máy hiện tại iPhone là camera 1
secondeye-detection camera-demo --camera 1

# Tạo lại pilot công khai (pixels/labels vẫn ở data/local và bị Git bỏ qua)
python scripts/build_openimages_indoor_pilot.py --review-complete

# Kiểm tra pilot 80 ảnh
secondeye-detection validate --dataset data/local/indoor_pilot_v1

# Kiểm tra dataset public v1.1 đã khóa (276 ảnh/603 bbox)
secondeye-detection validate --dataset data/local/indoor_dataset_v1_1
secondeye-validate-manifest data/local/indoor_dataset_v1_1/sample_manifest.csv \
  --config configs/data_protocol.toml --require-rows

# Train -> evaluate test nếu có, nếu không dùng val -> export/smoke-test ONNX -> package
secondeye-detection train --dataset data/local/indoor_dataset_v1_1
```

`camera-demo` chỉ nhận 80 lớp COCO và luôn được đánh dấu là demo. Muốn nhận đúng
15 lớp SecondEye phải dùng checkpoint YOLO26m đã fine-tune với lệnh `camera
--model ...`; checkpoint YOLO11n cũ không bị ghi đè.

Artifact mỗi lần train được lưu riêng trong `artifacts/object_obstacle/<run_id>/`, gồm
`best.pt`, `last.pt`, `model.onnx`, `dataset.yaml`, `metrics.json` và `manifest.json`.
Xem hướng dẫn đầy đủ tại `docs/local_yolo26_pipeline.md`.

## Chính sách dữ liệu indoor v1.1

Dự án không tự chụp ảnh bằng Mac/iPhone và không thu dữ liệu người tham gia.
`data/local/indoor_dataset_v1_1` chỉ được mở rộng từ dataset công khai có nguồn,
phiên bản và giấy phép rõ ràng. Ảnh web không rõ quyền không được sử dụng.

Các lớp phổ biến được nhập từ annotation nguồn phù hợp; cửa mở/đóng/kính và cầu
thang lên/xuống đã được relabel theo từng bbox sau visual review. Nguồn sử dụng là
Open Images V7 validation/test và ADE20K validation. ADE20K giới hạn ảnh cho nghiên
cứu/giáo dục phi thương mại, nên model huấn luyện từ bản này không mặc định phù
hợp để thương mại hóa. Pixels, labels và manifest chi tiết vẫn ở `data/local/`,
không đưa lên GitHub public. Xem hồ sơ giấy phép tại
`docs/data/public_dataset_license_review_v1_1.md`.

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
