# Schema lớp indoor SecondEye v1

Phiên bản: **1.0.0 — khóa ngày 2026-08-09**
Phạm vi: nhà ở, lớp học, hành lang, thư viện và cầu thang có giám sát.

> Trạng thái từ 2026-08-30: đây là taxonomy nghiên cứu lịch sử cho một custom
> detector có thể được xem xét trong tương lai. MVP hiện tại không fine-tune và
> dùng schema pretrained `indoor_coco_baseline_v1` trong
> `configs/pretrained_indoor.toml`. Hai schema không được trộn class ID hoặc dùng
> thay thế nhau.

## Class ID đã khóa

| ID | Tên lớp | Quy tắc ngắn |
|---:|---|---|
| 0 | `person` | Người thật nhìn thấy; không gán tranh, tượng hoặc phản chiếu mơ hồ |
| 1 | `chair` | Ghế một người, gồm ghế học/ghế văn phòng |
| 2 | `table_desk` | Mặt bàn phục vụ đặt đồ, học tập hoặc làm việc |
| 3 | `sofa` | Ghế bọc dài/daybed đang thể hiện chức năng ngồi |
| 4 | `bed` | Giường/nệm thể hiện chức năng ngủ |
| 5 | `cabinet` | Tủ, kệ/tủ bếp, ngăn kéo lưu trữ |
| 6 | `doorway_open` | Lối cửa đang mở, bbox bao vùng thông hành nhìn thấy |
| 7 | `door_closed` | Cánh cửa đóng chặn lối, không dùng cho tủ |
| 8 | `glass_door` | Cửa kính có bằng chứng khung/tay nắm hoặc mép kính |
| 9 | `stairs_up` | Nhịp bậc đi lên theo hướng nhìn hiện tại |
| 10 | `stairs_down` | Nhịp bậc đi xuống theo hướng nhìn hiện tại |
| 11 | `backpack_bag` | Ba lô hoặc túi đủ lớn có thể nằm trên lối đi |
| 12 | `box` | Thùng/hộp rời, gồm thùng carton |
| 13 | `trash_bin` | Thùng/rổ dùng chứa rác, không gán hộp lưu trữ thông thường |
| 14 | `column` | Cột kết cấu độc lập; không gán khung cửa/tường nhô |

Thứ tự ID là giao diện giữa dataset, checkpoint và runtime. Không đổi thứ tự hoặc đổi tên trong v1.x.

## Các cặp khó và cách xử lý

- `doorway_open` / `door_closed`: cần thấy vùng thông hành hoặc cánh cửa chặn lối. Cửa bị cắt khung, tối hoặc che khuất chuyển review, không đoán.
- `glass_door` / cửa sổ: cần bằng chứng cửa như tay nắm, ray, bản lề hoặc lối đi. Phản xạ đơn lẻ không đủ.
- `stairs_up` / `stairs_down`: nhãn theo hướng nhìn của camera, không theo tên tầng. Nếu cùng ảnh có hai nhịp khác hướng, gán từng nhịp riêng.
- `table_desk` / `cabinet`: ưu tiên chức năng bề mặt làm việc cho bàn; phần lưu trữ đứng/âm tường là tủ. Có thể có hai bbox khi hai vật tách biệt rõ.
- `sofa` / `bed`: dựa vào chức năng cảnh và cấu trúc. Sofa bed đang mở để ngủ là `bed`; daybed dùng ngồi là `sofa`; trường hợp mơ hồ đưa review.
- `box` / `trash_bin`: hình dáng không đủ; cần dấu hiệu sử dụng. Thùng carton chứa đồ là `box`, thùng có túi rác/nắp/vị trí thu gom là `trash_bin`.
- `column` / khung cửa: cột phải là kết cấu đứng độc lập có thể cản lối; không gán nẹp hoặc cạnh tường.
- `person`: ảnh pilot public có người nhận dạng được bị loại để giảm rủi ro riêng tư. Nếu bổ sung từ nguồn public, phải có giấy phép phù hợp, privacy review và ưu tiên ảnh đã làm mờ mặt hoặc không nhận dạng được; không tự chụp người.

## Coverage pilot 80 ảnh

| Lớp có bbox | Train | Validation | Tổng |
|---|---:|---:|---:|
| `chair` | 27 | 10 | 37 |
| `table_desk` | 29 | 9 | 38 |
| `sofa` | 21 | 5 | 26 |
| `bed` | 36 | 10 | 46 |
| `cabinet` | 54 | 11 | 65 |
| **Tổng** | **167** | **45** | **212** |

Các lớp còn lại chưa có bbox trong pilot. Không được diễn giải việc schema đã khóa thành dataset đã đủ coverage.

## Coverage dataset public v1.1

| Lớp | Tổng bbox |
|---|---:|
| `person` | 77 |
| `chair` | 51 |
| `table_desk` | 76 |
| `sofa` | 27 |
| `bed` | 47 |
| `cabinet` | 69 |
| `doorway_open` | 20 |
| `door_closed` | 27 |
| `glass_door` | 26 |
| `stairs_up` | 27 |
| `stairs_down` | 23 |
| `backpack_bag` | 22 |
| `box` | 26 |
| `trash_bin` | 21 |
| `column` | 64 |
| **Tổng** | **603** |

Bản v1.1 có 276 ảnh: 207 train và 69 validation. Mọi lớp đều xuất hiện trong cả
hai split. Perceptual near-duplicate được gom cùng group trước khi chia; manifest
và YOLO validator đều đạt ngày 2026-08-09.

Các số liệu trên là trạng thái cấu trúc lịch sử. Semantic re-audit v1.2 sau đó
phát hiện sai miền/sai nhãn/thiếu dense annotation; không dùng v1.1 để train.

## Chính sách repository

- Pilot khóa nằm tại `data/local/indoor_pilot_v1/`; dataset public hiện hành nằm
  tại `data/local/indoor_dataset_v1_1/`. Cả hai bị `.gitignore` loại toàn bộ.
- Chỉ schema, script tái tạo, danh sách loại trừ và thống kê tổng hợp được phép commit.
- Trước mỗi lần push chạy `git status --short` và `git check-ignore data/local/indoor_pilot_v1/images/train/<file>.jpg`.
