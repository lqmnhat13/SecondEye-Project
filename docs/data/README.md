# Giai đoạn 2 - Dữ liệu và protocol

Phiên bản: 1.0.0
Cập nhật: 2026-08-09
Trạng thái: protocol và schema indoor v1 đã khóa; obstacle public v1.1 có
**276 ảnh/603 bbox**, phủ 15/15 lớp và mỗi lớp có ít nhất 20 bbox. YOLO validator
và manifest audit đều đạt; dữ liệu vẫn nằm ngoài Git.

## Sản phẩm

1. `data_card.md`: mục đích, thành phần dự kiến, nguồn gốc, giới hạn và quản trị dữ liệu.
2. `annotation_guide.md`: hướng dẫn gán nhãn obstacle, OCR và VQA.
3. `collection_protocol.md`: quy trình thu thập, nhập kho, kiểm tra và xử lý rút đồng thuận.
4. `split_protocol.md`: quy tắc chia development/test theo `group_id`, `scene_id` và `video_id`.
5. `../../configs/data_protocol.toml`: target, controlled vocabulary và tỷ lệ split được version hóa.
6. `../../data/templates/`: manifest và ba mẫu annotation CSV.
7. `../../src/secondeye/data/protocol.py`: validator schema, quyền sử dụng và leakage.
8. `indoor_public_data_plan_v1_1.md`: kế hoạch bổ sung 10 lớp từ dữ liệu công khai.
9. `indoor_schema_v1.md`: class ID khóa, quy tắc phân biệt lớp và coverage pilot.
10. `public_dataset_license_review_v1_1.md`: quyết định nguồn, giấy phép và nghĩa
    vụ attribution cho Open Images/ADE20K; lý do không nhập Objects365.

## Cổng trước khi nhập dữ liệu công khai

- [ ] GVHD duyệt phạm vi, target và split protocol.
- [ ] Điền ngày kết thúc dự án, thời hạn lưu trữ và người chịu trách nhiệm dữ liệu.
- [x] Chốt không tự chụp và không dùng dữ liệu người tham gia cho dataset v1.1.
- [x] Rà soát Open Images V7, ADE20K và Objects365; lưu quyết định và điều kiện
  phái sinh trong `public_dataset_license_review_v1_1.md`.
- [x] Chạy pilot công khai 80 ảnh, loại ảnh có người/ngoài trời/sản phẩm rời, kiểm tra bbox và leakage.
- [x] Tạo `indoor_dataset_v1_1` riêng từ pilot khóa và giữ toàn bộ dữ liệu ngoài Git.
- [x] Bổ sung public data cho 10 lớp còn thiếu; relabel bbox cửa/cầu thang theo
  schema SecondEye và chuyển toàn bộ nhãn sang YOLO.
- [x] Review toàn bộ ảnh/bbox mới được chấp nhận và validation hiện tại; ghi rõ
  đây chưa phải đánh giá camera đích và chưa có reviewer độc lập thứ hai.

Không chuyển dữ liệu `pending` hoặc `withdrawn` ra khỏi `quarantine`. SecondEye là công cụ nghiên cứu hỗ trợ, không thay thế gậy trắng, chó dẫn đường hoặc thiết bị điều hướng chuyên dụng.

## Tài liệu nền

- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010): cấu trúc motivation, composition, collection, use và maintenance.
- [NIST AI RMF 1.0](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10): quản trị rủi ro trong thiết kế, phát triển và đánh giá hệ thống AI.
- [Luật Bảo vệ dữ liệu cá nhân 91/2025/QH15](https://vanban.chinhphu.vn/?classid=1&docid=214590&pageid=27160&typegroup=): nguồn pháp lý chính thức hiện hành từ 2026-01-01. Protocol này không phải tư vấn pháp lý; yêu cầu của trường và phê duyệt nghiên cứu vẫn là cổng bắt buộc.
