# Kế hoạch dữ liệu indoor v1.1 chỉ dùng nguồn công khai

Cập nhật: 2026-08-09

Quyết định phạm vi: không tự chụp ảnh bằng camera Mac/iPhone và không thu dữ liệu
người tham gia. Bản v1.1 chỉ sử dụng dataset công khai hoặc dữ liệu synthetic có
provenance rõ ràng.

Trạng thái thực hiện: **hoàn thành bản dữ liệu v1.1** tại
`data/local/indoor_dataset_v1_1` với 276 ảnh/603 bbox; 15/15 lớp có ít nhất 20
bbox. Manifest SHA-256:
`600d8e9c4be184485cebac1685185ad8d97407ae7d6050d8a2140bf47aa3fb32`.

## Mục tiêu

- Giữ schema 15 lớp đã khóa.
- Tối thiểu 250 ảnh và khoảng 20 bbox cho mỗi lớp trước khi train v1.1.
- Bổ sung 10 lớp đang trống: `person`, `doorway_open`, `door_closed`, `glass_door`,
  `stairs_up`, `stairs_down`, `backpack_bag`, `box`, `trash_bin`, `column`.
- Không đưa pixels, labels hoặc manifest chi tiết lên GitHub public.

## Quy trình cho mỗi nguồn

1. Ghi tên dataset, phiên bản, trang chính thức, URL download, giấy phép ảnh và
   annotation, ngày truy cập và checksum file nguồn.
2. Chỉ tải từ trang chính thức hoặc kho do tác giả công bố; không lấy ảnh tùy ý
   từ kết quả tìm kiếm web.
3. Lọc cảnh indoor, ảnh hỏng, ảnh trùng, sản phẩm tách nền và dữ liệu nhạy cảm.
4. Giữ annotation nguồn cho lớp tương thích. Relabel thủ công các lớp có định
   nghĩa riêng của SecondEye.
5. Gom ảnh cùng nguồn/cảnh/burst và near-duplicate vào cùng group trước khi chia
   train/validation.
6. Reviewer kiểm tra toàn bộ validation và các cặp lớp dễ nhầm.
7. Chạy cả validator YOLO và manifest trước khi nhập vào bản làm việc.

## Lớp cần relabel đặc biệt

- `doorway_open` / `door_closed`: trạng thái lối đi không thể suy ra chỉ từ nhãn
  `Door` chung.
- `glass_door`: loại cửa sổ và phản chiếu không đủ bằng chứng.
- `stairs_up` / `stairs_down`: hướng được xác định theo góc nhìn camera; ảnh mơ hồ
  bị loại hoặc chuyển review.
- `box` / `trash_bin`, `column` / khung cửa: review theo schema v1.
- `person`: chỉ dùng nguồn có quyền phù hợp; ưu tiên ảnh không nhận dạng được hoặc
  đã làm mờ mặt mà vẫn giữ được hình thể cần cho detection.

## Cổng trước khi train

```bash
secondeye-detection validate --dataset data/local/indoor_dataset_v1_1
secondeye-validate-manifest \
  data/local/indoor_dataset_v1_1/sample_manifest.csv \
  --config configs/data_protocol.toml --require-rows
```

Ngoài số lượng, báo cáo phải ghi rõ source coverage, bbox mỗi lớp, tỷ lệ loại,
license, manifest hash và kết quả review validation.

Nguồn và điều kiện quyền sử dụng đã được rà soát tại
`public_dataset_license_review_v1_1.md`. Open Images validation/test được dùng có
điều kiện; ADE20K cung cấp `column` và một phần `stairs_down` trong phạm vi
nghiên cứu/giáo dục phi thương mại. Objects365 không được nhập vào v1.1.

## Giới hạn bắt buộc ghi trong luận văn

Kết quả chỉ đo trên dữ liệu công khai. Không có bằng chứng trực tiếp rằng model
tổng quát tốt sang camera Mac/iPhone, cách cầm camera của người dùng hoặc môi
trường nhà/trường học thực tế. Demo camera chỉ là minh họa kỹ thuật, không phải
đánh giá hiệu quả hoặc an toàn trong triển khai thực tế.
