# Pipeline YOLO11 chạy local cho SecondEye

Pipeline này là bản chuyển đổi từ notebook
`Bản_sao_của_SecondEye_Object_Obstacle_YOLO11_Colab(1).ipynb`. Mã chạy local nằm
trong `src/secondeye/detection/`; notebook không còn là entry point thực thi.

## 1. Cài môi trường

Thiết lập đã được kiểm tra trên macOS Apple Silicon và Python 3.11. Chạy từ thư mục gốc
project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[dev,detection]"
```

Lần chạy đầu, Ultralytics có thể tải `yolo11n.pt`. Train local tự chọn thiết bị theo thứ
tự CUDA -> Apple MPS -> CPU. Có thể sửa `model.device` trong
`configs/yolo11_obstacles.toml` nếu cần ép thiết bị.

Có thể smoke test cài đặt ngay, chưa cần dataset:

```bash
secondeye-detection demo --source data/samples/ultralytics_bus.jpg
```

Lệnh này cố ý ghi `result_type=pretrained_coco_demo_not_second_eye_model`: đây chỉ là
YOLO11n pretrained COCO 80 lớp, không phải model 12 lớp của SecondEye.

## 2. Chuẩn bị dataset

Dataset phải có cấu trúc:

```text
secondeye_obstacles/
├── images/
│   ├── train/
│   ├── val/
│   └── test/       # không bắt buộc nhưng rất nên có
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Mỗi dòng label dùng định dạng YOLO:

```text
class_id x_center y_center width height
```

Nếu dữ liệu là ZIP:

```bash
secondeye-detection prepare \
  --archive /duong/dan/secondeye_obstacles.zip \
  --destination data/local/secondeye_obstacles_import
```

Lệnh không xóa dữ liệu cũ. Thư mục đích phải chưa tồn tại hoặc rỗng; ZIP chứa đường dẫn
`../` hoặc symbolic link sẽ bị từ chối. Lệnh sẽ in ra dataset root thực tế, hỗ trợ cả ZIP
có hoặc không có một thư mục bao ngoài.

## 3. Kiểm tra dữ liệu

```bash
secondeye-detection validate \
  --dataset /duong/dan/toi/secondeye_obstacles \
  --output results/dataset_validation.json
```

Validation dừng sớm khi gặp:

- ảnh hỏng hoặc split train/val rỗng;
- label mồ côi, hai ảnh dùng chung stem hoặc annotation trùng;
- dòng label không đủ năm trường, NaN/infinity, class ID ngoài schema;
- bbox có kích thước không dương hoặc vượt biên ảnh chuẩn hóa;
- ảnh giống hệt xuất hiện ở nhiều split.

Ảnh không có file label hoặc label rỗng được xem là ảnh nền hợp lệ và được báo riêng.

## 4. Huấn luyện và lưu model

```bash
secondeye-detection train \
  --dataset /duong/dan/toi/secondeye_obstacles
```

Lệnh thực hiện tuần tự:

1. validate toàn bộ dataset;
2. tạo `dataset.yaml` với đường dẫn tuyệt đối;
3. fine-tune `yolo11n.pt` theo config;
4. lấy `best.pt` từ thư mục run do Ultralytics thực sự trả về;
5. đánh giá trên `test` nếu có, nếu không mới dùng `val`;
6. export ONNX trên CPU, chạy `onnx.checker` và một inference smoke test;
7. đóng gói artifact kèm checksum, metric, phiên bản môi trường và giới hạn sử dụng.

Mỗi lần chạy dùng thư mục riêng:

```text
artifacts/object_obstacle/<run_id>/
├── best.pt
├── last.pt
├── model.onnx
├── dataset.yaml
├── metrics.json
└── manifest.json
```

Không thêm `conf=0.25` khi tính mAP; ngưỡng confidence dùng cho ứng dụng không được làm
sai lệch protocol đánh giá. Nếu thiếu test split, `manifest.json` ghi rõ metric chỉ là
development/validation metric.

Có thể bỏ export trong một lần train thử:

```bash
secondeye-detection train --dataset /duong/dan/dataset --no-export
```

Export lại sau:

```bash
secondeye-detection export \
  --model artifacts/object_obstacle/<run_id>/best.pt
```

## 5. Đánh giá model đã lưu

```bash
secondeye-detection evaluate \
  --model artifacts/object_obstacle/<run_id>/best.pt \
  --dataset /duong/dan/toi/secondeye_obstacles \
  --split auto \
  --output results/yolo11_evaluation.json
```

`--split auto` ưu tiên test và chỉ dùng val nếu không có test. Pipeline kiểm tra thứ tự/tên
12 lớp trong checkpoint; `yolo11n.pt` pretrained COCO 80 lớp sẽ bị từ chối thay vì âm thầm
đánh giá sai class ID.

## 6. Predict ảnh và webcam local

```bash
secondeye-detection predict \
  --model artifacts/object_obstacle/<run_id>/best.pt \
  --source /duong/dan/anh.jpg \
  --output-json results/prediction.json \
  --output-image results/prediction.jpg
```

Mở camera local, không tạo URL public:

```bash
secondeye-detection camera \
  --model artifacts/object_obstacle/<run_id>/best.pt \
  --camera 0
```

Nhấn `q` hoặc `Esc` để thoát. macOS có thể yêu cầu cấp quyền Camera cho Terminal hoặc
ứng dụng Python đang chạy.

## 7. Tích hợp vào trợ lý đa phương thức

```python
from pathlib import Path

import cv2

from secondeye.detection import ObjectObstacleDetector, load_detection_config

config = load_detection_config()
detector = ObjectObstacleDetector(
    Path("artifacts/object_obstacle/<run_id>/best.pt"), config
)
detector.warmup()

frame_bgr = cv2.imread("input.jpg", cv2.IMREAD_COLOR)
result = detector.predict_bgr(frame_bgr)
```

Input NumPy được quy ước rõ là OpenCV BGR `uint8`, tránh lỗi đổi kênh màu của bản Gradio.
Output có `class_id`, `label`, bbox, hướng và cờ `obstacle_candidate`, nhưng luôn để
`depth_zone=None`. Module detection không tuyên bố vật thể đang gần hoặc sắp va chạm;
module depth và bộ điều phối cảnh báo phải quyết định phần đó.

## Giới hạn an toàn và giấy phép

Đây là nguyên mẫu nghiên cứu hỗ trợ, không thay thế gậy trắng, chó dẫn đường hoặc thiết bị
định hướng chuyên dụng. Trước khi phát hành hoặc thương mại hóa cần rà soát điều kiện giấy
phép Ultralytics/weights và cách phân phối toàn bộ ứng dụng.
