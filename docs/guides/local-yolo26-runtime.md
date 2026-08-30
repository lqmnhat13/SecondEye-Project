# Pipeline YOLO26 pretrained chạy local cho SecondEye

Cập nhật: 2026-08-30.

Tài liệu này mô tả detection runtime đang được dùng trong MVP. Phiên bản hiện
tại chạy `yolo26m.pt` pretrained COCO qua adapter có ánh xạ tường minh; dự án
không fine-tune và không yêu cầu custom checkpoint để hoàn thành MVP.

Các lệnh `train`, `evaluate`, `export`, `predict` và `camera` cho custom checkpoint
vẫn được giữ trong source như hạ tầng nghiên cứu tương lai/tương thích ngược.
Chúng không nằm trong quy trình cài đặt, demo hoặc đánh giá chính hiện tại.

## 1. Runtime chính

Thiết lập đã được kiểm tra trên macOS Apple Silicon và Python 3.11:

```bash
cd /Users/lenhat/Documents/SecondEye-Project
brew install ffmpeg
./setup_mvp.sh
source ~/Library/Caches/SecondEye/venv/bin/activate
secondeye doctor
```

Lần chạy đầu, Ultralytics và các adapter đa phương thức có thể tải model vào
cache local. Sau khi cache đủ, inference không gửi ảnh lên API.

Smoke test detection trên ảnh:

```bash
python scripts/fetch_smoke_asset.py
secondeye-detection demo \
  --source data/samples/ultralytics_bus.jpg \
  --output-json results/detection_smoke.json
```

Chạy toàn bộ stack trên ảnh hoặc camera:

```bash
secondeye image --source data/samples/ultralytics_bus.jpg --depth --no-tts
./run_mvp.sh --camera 0
```

## 2. Schema detection hiện tại

Runtime dùng schema `indoor_coco_baseline_v1` gồm 15 lớp có ánh xạ trực tiếp từ
COCO:

| YOLO26 COCO | SecondEye runtime |
|---|---|
| `person` | `person` |
| `chair` | `chair` |
| `dining table` | `table` |
| `couch` | `sofa` |
| `bed` | `bed` |
| `backpack` | `backpack` |
| `handbag` | `handbag` |
| `suitcase` | `suitcase` |
| `bottle` | `bottle` |
| `potted plant` | `potted_plant` |
| `tv` | `tv` |
| `laptop` | `laptop` |
| `toilet` | `toilet` |
| `sink` | `sink` |
| `refrigerator` | `refrigerator` |

Adapter loại mọi lớp không có mapping; không đổi tên gần đúng và không suy đoán
cửa, cầu thang, cột, tủ, hộp hoặc thùng rác. Vì vậy schema này là baseline tích
hợp, không phải taxonomy an toàn đầy đủ.

Sáu threshold đầu (`person`, `chair`, `table`, `sofa`, `bed`, `backpack`) kế
thừa calibration trên development benchmark 69 ảnh/68 bbox. Chín threshold còn
lại là mặc định provisional `0.35`, chưa phải kết quả accuracy đã calibration.

## 3. Từ detection đến cảnh báo

Detection 2D chỉ tạo ứng viên. Một vật chỉ phát cảnh báo khi:

1. lớp nằm trong `risk.candidate_classes`;
2. tâm bbox nằm trong vùng di chuyển trung tâm;
3. relative depth xác nhận band `near`;
4. cùng khóa vật thể/hướng được thấy trong đủ số frame xác nhận;
5. cảnh báo không còn trong thời gian cooldown.

`near/medium/far` là mức tương đối trong từng frame, không phải khoảng cách mét.
Khi depth tắt hoặc quá cũ, runtime không phát câu “ở gần”.

## 4. Output ổn định

Adapter trả JSON gồm:

- `schema_name`, model và device;
- `source_class_id` COCO và `class_id` trong schema runtime;
- label nguồn/label chuẩn, confidence và bbox;
- hướng trái/giữa/phải;
- `obstacle_candidate`, lý do và depth zone khi có;
- latency và giới hạn sử dụng.

Input là ảnh OpenCV BGR `uint8`. Camera runtime dùng latest-frame buffer để bỏ
frame cũ; detection/depth chạy ngoài UI và semantic worker không chặn luồng vision.

## 5. Dataset và custom checkpoint tương lai

Repo vẫn có validator dataset và các lệnh nghiên cứu:

```bash
secondeye-detection prepare --archive /duong/dan/dataset.zip
secondeye-detection validate --dataset /duong/dan/dataset
secondeye-detection train --dataset /duong/dan/dataset
secondeye-detection evaluate --model /duong/dan/best.pt --dataset /duong/dan/dataset
secondeye-detection export --model /duong/dan/best.pt
```

Các lệnh này không được hiểu là kế hoạch hiện tại. Chỉ dùng chúng nếu dự án sau
này chính thức mở một nhánh fine-tuning mới. Trước khi chạy cần có tối thiểu:

- taxonomy/version được chốt riêng, không trộn với schema COCO runtime;
- dataset đã dense-label, rà privacy/license và có human review;
- split development/test độc lập, manifest/hash và protocol metric;
- mục tiêu nghiên cứu giải thích vì sao pretrained baseline chưa đủ;
- tài nguyên, giấy phép và kế hoạch đánh giá custom checkpoint.

Dataset indoor v1.1/v1.2 lịch sử chưa đáp ứng các điều kiện này. Trạng thái đó
không chặn MVP pretrained và không phải việc cần hoàn tất trong phiên bản hiện tại.

## 6. Giới hạn an toàn và giấy phép

SecondEye là nguyên mẫu nghiên cứu hỗ trợ, không thay thế gậy trắng, chó dẫn
đường hoặc thiết bị định hướng chuyên dụng. Không dùng output để hướng dẫn giao
thông/cầu thang không giám sát. Trước khi phát hành hoặc thương mại hóa phải rà
điều kiện của Ultralytics, weights và toàn bộ dependency/model pretrained.
