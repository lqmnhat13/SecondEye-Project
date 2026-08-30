# Kế hoạch triển khai 10 tuần

Kế hoạch dùng tuần tương đối vì chưa có ngày bảo vệ chính thức. Trạng thái trong
tài liệu này được cập nhật theo hiện trạng MVP ngày 2026-08-30.

Quyết định phạm vi cập nhật ngày 2026-08-30: toàn bộ kế hoạch chính dùng model
pretrained. Fine-tuning/custom checkpoint không phải milestone, không phải cổng
chất lượng và không phải điều kiện trả lời ba RQ. Dữ liệu vẫn cần cho đánh giá
độc lập, nhưng không được hiểu là dataset bắt buộc để huấn luyện model.

| Tuần | Mục tiêu | Đầu ra kiểm chứng được | Cổng chất lượng |
|---|---|---|---|
| 1 | Chốt đề tài và tổng quan | Charter, 3 RQ, 3 user stories, 24 nguồn đã rà đến 2026-08-30 | GVHD chốt phạm vi, thiết bị, protocol sơ bộ |
| 2 | Dữ liệu và baseline 1 | Data card, annotation guide, split theo group; detection và OCR | Smoke test + log mẫu; không dùng test để tuning |
| 3 | Baseline 2 và đóng phạm vi | Depth, VQA, speech; benchmark trên máy demo; ADR cấu hình cuối | Chọn đúng một cấu hình chính/mô-đun; backlog hóa phần nâng cao |
| 4 | Tích hợp và khóa tính năng | 3 pipeline end-to-end, config và log thống nhất | Feature freeze; lỗi mô-đun nâng cao không kéo chậm MVP |
| 5 | Điều phối và accessibility | State machine, một audio orchestrator, nút lớn/push-to-talk | Test priority/cooldown/STOP/repeat; không phụ thuộc màu |
| 6 | Kiểm thử và tối ưu | Unit/integration/failure tests; xử lý camera/micro/mạng lỗi | Báo cáo P50/P95 sơ bộ; đóng lỗi nghiêm trọng |
| 7 | Khóa dữ liệu đánh giá, chạy thí nghiệm | Test set manifest/hash; pretrained baseline và ablation | Config/commit/device được ghi; dự đoán cấp mẫu đầy đủ; không huấn luyện lại model |
| 8 | Phân tích | Metrics, CI nếu phù hợp, biểu đồ, >=20 failure cases | Trả lời RQ bằng dữ liệu thật; tách kết quả và suy luận |
| 9 | Viết và ổn định demo | Chương 1-5, hình/bảng thật, video dự phòng | Khóa code giữa tuần; clean-environment run lần 1 |
| 10 | Hoàn thiện và bảo vệ | Chương 6, bản cuối, slide, script, Q&A, checklist demo | Hai rehearsal có bấm giờ; clean-environment run lần 2 |

## Kế hoạch 7 ngày đầu

| Ngày | Việc | Trạng thái hiện tại |
|---|---|---|
| 1 | Chốt MVP, thiết bị và Charter | Hoàn thành bản nháp; còn xác nhận deadline/offline |
| 2 | Chốt 3 RQ và 3 user stories | Hoàn thành bản nháp |
| 3 | Tạo repo và môi trường Python | Hoàn thành; package, setup script và runtime local đã có |
| 4 | Detection trên ảnh và webcam | Hoàn thành baseline; ảnh và live camera smoke đã có |
| 5 | OCR trên ảnh development tiếng Việt | Adapter và smoke đã có; benchmark tập đánh giá còn thiếu |
| 6 | Depth tại ba vùng khoảng cách | Adapter/fusion đã có; calibration và benchmark còn thiếu |
| 7 | Tổng hợp baseline và biên bản chốt | MVP tích hợp đã có; báo cáo benchmark chính thức còn thiếu |

## Nếu chỉ còn 8 tuần

Gộp tuần 1-2 và 5-6; viết Chương 1-3 từ tuần 3. Giữ quyết định không fine-tune;
bỏ mobile và user study chính thức. Không cắt test set độc lập, một ablation và
phân tích failure cases trên stack pretrained.
