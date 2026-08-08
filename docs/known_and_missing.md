# Thông tin đã biết và còn thiếu

Cập nhật: 2026-08-04.

## Đã biết

- Nguồn yêu cầu chính: cẩm nang 18 trang, phiên bản 1.1 ngày 03/08/2026.
- MVP bắt buộc gồm cảnh báo vật cản, OCR tiếng Việt và mô tả/VQA; có TTS.
- Kiến trúc hai luồng và một bộ điều phối âm thanh duy nhất là bắt buộc.
- Ba RQ, protocol dữ liệu, ablation và nhóm metrics đã được cẩm nang định nghĩa.
- Thiết bị hiện có: MacBook Pro M1, RAM 16 GB, GPU Metal, camera tích hợp và iPhone camera.
- Repo ban đầu chưa có mã nguồn hay commit; chỉ có cẩm nang và tệp tạo/render cẩm nang.
- Baseline ưu tiên Python/OpenCV/Gradio; YOLO nhỏ, PaddleOCR Việt, Depth Anything V2 nhỏ.

## Thiếu nhưng chưa chặn baseline local

- Ngày bắt đầu, deadline nộp luận văn và ngày bảo vệ thực tế.
- Demo cuối bằng camera laptop, iPhone Continuity Camera hay thiết bị khác.
- Offline là bắt buộc hay chỉ mong muốn; có được dùng VLM API trong demo/thí nghiệm không.
- Ngân sách API/cloud và mức cho phép gửi ảnh ra dịch vụ ngoài.
- Khả năng tiếp cận người khiếm thị, người giám sát và quy trình phê duyệt đạo đức của trường.
- Chuẩn trích dẫn, template luận văn, thời lượng và template slide của trường.
- Repo sẽ công khai theo AGPL-3.0 hay cần thay baseline/giấy phép khác trước phát hành.
- Thiết bị micro/loa, quyền camera và chất lượng mạng tại phòng bảo vệ chưa được thử thực tế.

## Điểm quyết định phải chốt đúng lúc

1. Trước baseline VQA: offline/API, riêng tư và ngân sách.
2. Trước khóa protocol: deadline, ngưỡng chấp nhận và chuẩn trích dẫn của trường.
3. Trước thu thập/user study: tiếp cận người dùng, đồng thuận, lưu trữ và phê duyệt đạo đức.
4. Trước phát hành/demo công khai: nghĩa vụ giấy phép của Ultralytics và dữ liệu.

