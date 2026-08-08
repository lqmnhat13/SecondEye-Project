# Giai đoạn 2 - Dữ liệu và protocol

Phiên bản: 0.1.0  
Cập nhật: 2026-08-04  
Trạng thái: bộ protocol hoàn thành; dữ liệu thực tế chưa được thu thập.

## Sản phẩm

1. `data_card.md`: mục đích, thành phần dự kiến, nguồn gốc, giới hạn và quản trị dữ liệu.
2. `annotation_guide.md`: hướng dẫn gán nhãn obstacle, OCR và VQA.
3. `collection_protocol.md`: quy trình thu thập, nhập kho, kiểm tra và xử lý rút đồng thuận.
4. `split_protocol.md`: quy tắc chia development/test theo `group_id`, `scene_id` và `video_id`.
5. `../../configs/data_protocol.toml`: target, controlled vocabulary và tỷ lệ split được version hóa.
6. `../../data/templates/`: manifest và ba mẫu annotation CSV.
7. `../../src/secondeye/data/protocol.py`: validator schema, quyền sử dụng và leakage.

## Cổng trước khi thu dữ liệu

- [ ] GVHD duyệt phạm vi, target và split protocol.
- [ ] Điền ngày kết thúc dự án, thời hạn lưu trữ và người chịu trách nhiệm dữ liệu.
- [ ] Chốt có hay không dữ liệu người tham gia.
- [ ] Nếu có người tham gia: có phê duyệt/miễn trừ phù hợp, consent form dễ tiếp cận và quy trình rút đồng thuận.
- [ ] Kiểm tra camera/micro và vùng lưu trữ mã hóa trên máy demo.
- [ ] Chạy thử 10–20 mẫu development không có dữ liệu cá nhân; sửa guide trước khi thu tập chính.

Không chuyển dữ liệu `pending` hoặc `withdrawn` ra khỏi `quarantine`. SecondEye là công cụ nghiên cứu hỗ trợ, không thay thế gậy trắng, chó dẫn đường hoặc thiết bị điều hướng chuyên dụng.

## Tài liệu nền

- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010): cấu trúc motivation, composition, collection, use và maintenance.
- [NIST AI RMF 1.0](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10): quản trị rủi ro trong thiết kế, phát triển và đánh giá hệ thống AI.
- [Luật Bảo vệ dữ liệu cá nhân 91/2025/QH15](https://vanban.chinhphu.vn/?classid=1&docid=214590&pageid=27160&typegroup=): nguồn pháp lý chính thức hiện hành từ 2026-01-01. Protocol này không phải tư vấn pháp lý; yêu cầu của trường và phê duyệt nghiên cứu vẫn là cổng bắt buộc.

