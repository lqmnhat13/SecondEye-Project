# Pipeline YOLO26 chạy local cho SecondEye

Pipeline này phát triển từ notebook
`Bản_sao_của_SecondEye_Object_Obstacle_YOLO11_Colab(1).ipynb`, nhưng runtime hiện
dùng YOLO26m. Mã chạy local nằm trong `src/secondeye/detection/`; notebook không
còn là entry point thực thi.

## 1. Cài môi trường

Thiết lập đã được kiểm tra trên macOS Apple Silicon và Python 3.11. Chạy từ thư mục gốc
project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[dev,detection]"
```

Lần chạy đầu, Ultralytics có thể tải `yolo26m.pt`. Train local tự chọn thiết bị theo thứ
tự CUDA -> Apple MPS -> CPU. Có thể sửa `model.device` trong
`configs/yolo26_obstacles.toml` nếu cần ép thiết bị.

Có thể smoke test cài đặt ngay, chưa cần dataset:

```bash
secondeye-detection demo --source data/samples/ultralytics_bus.jpg
```

Lệnh này cố ý ghi `result_type=pretrained_coco_demo_not_second_eye_model`: đây chỉ là
YOLO26m pretrained COCO 80 lớp, không phải model 15 lớp indoor của SecondEye.

Có thể dùng ngay YOLO26m pretrained với camera mà không train. Trên máy hiện tại,
camera iPhone Continuity có chỉ số 1:

```bash
secondeye-detection camera-demo --camera 1
```

`camera-demo` tự áp dụng confidence theo lớp đã calibration trên validation:
`person=0.29`, `chair=0.69`, `table_desk=0.28`, `sofa=0.31`, `bed=0.48`,
`backpack_bag=0.17`. Adapter chỉ xuất các lớp có mapping rõ ràng; mọi nhãn COCO
khác bị loại và không được đổi tên gần đúng hay dùng để suy đoán lớp SecondEye.
Đây là cấu hình riêng cho YOLO26m pretrained. Checkpoint SecondEye đã fine-tune
tiếp tục dùng ngưỡng `model.confidence_threshold=0.35` cho đến khi được
calibration độc lập.

Các lớp SecondEye chưa được COCO adapter hỗ trợ là: `cabinet`, `doorway_open`,
`door_closed`, `glass_door`, `stairs_up`, `stairs_down`, `box`, `trash_bin` và
`column`. Muốn nhận diện các lớp này phải dùng checkpoint 15 lớp đã fine-tune;
adapter tuyệt đối không giả lập kết quả từ một lớp COCO khác.

| YOLO26 COCO | SecondEye | Trạng thái |
|---|---|---|
| `person` | `person` | Hỗ trợ |
| `chair` | `chair` | Hỗ trợ |
| `dining table` | `table_desk` | Hỗ trợ |
| `couch` | `sofa` | Hỗ trợ |
| `bed` | `bed` | Hỗ trợ |
| `backpack` | `backpack_bag` | Hỗ trợ |

Trong JSON demo, `class_id` là ID chuẩn của schema SecondEye; `source_class_id`
là ID gốc COCO phục vụ truy vết. Hai loại ID không được dùng thay thế nhau.

Cửa, cầu thang, tủ và các lớp riêng SecondEye không thuộc đầy đủ vocabulary COCO;
chế độ này chỉ phục vụ so sánh nhanh, không thay thế checkpoint 15 lớp.

## 2. Chuẩn bị dataset

Dataset phải có cấu trúc:

```text
indoor_dataset_v1_1/
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
  --dataset data/local/indoor_dataset_v1_1 \
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
  --dataset data/local/indoor_dataset_v1_1
```

Lệnh thực hiện tuần tự:

1. validate toàn bộ dataset;
2. tạo `dataset.yaml` với đường dẫn tuyệt đối;
3. fine-tune `yolo26m.pt` theo config;
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
  --output results/yolo26_evaluation.json
```

`--split auto` ưu tiên test và chỉ dùng val nếu không có test. Pipeline kiểm tra thứ tự/tên
15 lớp trong checkpoint; `yolo26m.pt` pretrained COCO 80 lớp sẽ bị từ chối thay vì âm thầm
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

Lệnh `camera` ở đây kiểm tra checkpoint phải đúng schema 15 lớp SecondEye. Dùng
`camera-demo` nếu muốn chạy trực tiếp YOLO26m pretrained mà chưa fine-tune.

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
