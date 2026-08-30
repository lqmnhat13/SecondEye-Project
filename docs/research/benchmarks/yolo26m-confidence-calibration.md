# YOLO26m confidence calibration

Cập nhật: 2026-08-09

## Protocol

- Model: `yolo26m.pt` pretrained COCO, không fine-tune.
- Device: Apple MPS; `imgsz=640`, batch 1.
- Development set: 69 ảnh validation, 68 bbox thuộc sáu lớp chung.
- Mapping: `person`, `chair`, `dining table -> table_desk`, `couch -> sofa`,
  `bed`, `backpack -> backpack_bag`.
- Matching: IoU 0.50; quét confidence 0.01–0.95, bước 0.01.
- Objective triển khai: micro F1 cân bằng bỏ sót và false alert.

Đây là calibration trên validation, không phải test độc lập. Annotation có thể
không exhaustive cho mọi vật COCO nhìn thấy, nên false positive có thể bị tính
cao hơn thực tế.

## Global threshold

| Confidence | Precision | Recall | F1 | F2 | TP/FP/FN |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.379 | 0.691 | 0.490 | **0.593** | 47/77/21 |
| 0.35 | 0.413 | 0.632 | 0.500 | 0.572 | 43/61/25 |
| **0.41** | **0.424** | 0.618 | **0.503** | 0.566 | 42/57/26 |
| 0.45 | 0.409 | 0.559 | 0.472 | 0.521 | 38/55/30 |
| 0.50 | 0.409 | 0.529 | 0.462 | 0.500 | 36/52/32 |

Ngưỡng toàn cục F1 tốt nhất là 0.41. F2 tốt nhất là 0.25 nhưng tạo thêm 20 false
positive so với 0.41; không chọn làm mặc định khi chưa có depth/risk để chặn cảnh
báo sai.

## Selected per-class thresholds

| Canonical class | Confidence | TP | FP | FN |
|---|---:|---:|---:|---:|
| `person` | 0.29 | 13 | 30 | 3 |
| `chair` | 0.69 | 9 | 14 | 2 |
| `table_desk` | 0.28 | 5 | 1 | 15 |
| `sofa` | 0.31 | 4 | 1 | 1 |
| `bed` | 0.48 | 10 | 1 | 0 |
| `backpack_bag` | 0.17 | 4 | 0 | 2 |

Tổng hợp bộ ngưỡng theo lớp: precision 0.489, recall 0.662, F1 0.563 và F2
0.618 (45 TP, 47 FP, 23 FN). So với ngưỡng cũ 0.35, bộ calibration tăng cả
precision, recall và F1 trên development set.

Benchmark lịch sử này chỉ bao phủ sáu lớp có ánh xạ COCO rõ ràng. Runtime hiện
đã mở rộng schema tích hợp lên 15 lớp COCO có mapping trực tiếp: sáu lớp trong
bảng giữ threshold đã khảo sát, chín lớp bổ sung dùng giá trị provisional `0.35`.
Nhãn COCO ngoài mapping bị loại, không được đổi tên gần đúng hoặc dùng để suy
đoán lớp an toàn đặc thù.

Các ngưỡng trên chỉ áp dụng cho YOLO26m pretrained COCO. Dự án hiện không
fine-tune; nếu tương lai xuất hiện custom checkpoint, checkpoint đó phải có
calibration và báo cáo riêng thay vì kế thừa các số liệu trong tài liệu này.
