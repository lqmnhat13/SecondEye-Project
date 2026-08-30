# Dữ liệu và protocol đánh giá

Phiên bản: 1.1.0
Cập nhật: 2026-08-30
Trạng thái: dự án hiện dùng model pretrained và không fine-tune. Protocol dữ liệu
được duy trì để xây dựng tập đánh giá, kiểm soát leakage/quyền riêng tư và lưu
lịch sử nhánh taxonomy an toàn. Obstacle public v1.1 có **276 ảnh/603 bbox**
trước semantic re-audit; v1.2 là staging nghiên cứu tương lai, không phải blocker
và không phải dependency của MVP hiện tại. Toàn bộ dữ liệu vẫn nằm ngoài Git.

## Tài liệu được giữ

1. `data_card.md`: mục đích, thành phần dự kiến, nguồn gốc, giới hạn và quản trị dữ liệu.
2. `annotation_guide.md`: hướng dẫn gán nhãn obstacle, OCR và VQA.
3. `collection_protocol.md`: quy trình thu thập, nhập kho, kiểm tra và xử lý rút đồng thuận.
4. `split_protocol.md`: quy tắc chia development/test theo `group_id`, `scene_id` và `video_id`.
5. `../../../src/secondeye/data/protocol.py`: validator manifest, quyền sử dụng và leakage.
6. `indoor_schema_v1.md`: taxonomy lịch sử, quy tắc phân biệt lớp và coverage pilot.
7. `public_dataset_license_review_v1_1.md`: quyết định nguồn, giấy phép và nghĩa
    vụ attribution cho Open Images/ADE20K; lý do không nhập Objects365.
8. `indoor_dataset_v1_2_review.md`: kết quả semantic re-audit và điều kiện human
    review nếu sau này dự án mở lại nhánh custom detector/fine-tuning.

## Việc còn lại cho dữ liệu đánh giá hoặc nhánh nghiên cứu tương lai

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

Các mục chưa hoàn tất trong phần này không ngăn chạy MVP pretrained. Chúng chỉ
ngăn việc dùng dữ liệu để công bố metric cuối, phát hành dataset hoặc huấn luyện
custom checkpoint trong tương lai.

## Tài liệu nền

- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010): cấu trúc motivation, composition, collection, use và maintenance.
- [NIST AI RMF 1.0](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10): quản trị rủi ro trong thiết kế, phát triển và đánh giá hệ thống AI.
- [Luật Bảo vệ dữ liệu cá nhân 91/2025/QH15](https://vanban.chinhphu.vn/?classid=1&docid=214590&pageid=27160&typegroup=): nguồn pháp lý chính thức hiện hành từ 2026-01-01. Protocol này không phải tư vấn pháp lý; yêu cầu của trường và phê duyệt nghiên cứu vẫn là cổng bắt buộc.
