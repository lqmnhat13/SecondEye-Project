# Project Charter - SecondEye

Phiên bản tài liệu: 0.3

Cập nhật: 2026-08-30

Đồng bộ với mã nguồn: 0.3.0 (`de67894`)

Trạng thái: phạm vi triển khai hiện tại dùng model pretrained, không fine-tune.

## Vấn đề và người dùng

Người khiếm thị cần tiếp nhận nhanh thông tin thị giác về vật cản phía trước, chữ tiếng Việt và nội dung cảnh. SecondEye nghiên cứu cách kết hợp nhiều mô-đun AI thành một hệ thống phản hồi đủ đúng, đủ nhanh và có cơ chế từ chối khi thiếu bằng chứng.

Người dùng mục tiêu là người mù và người có thị lực kém trong các tác vụ có kiểm soát, ưu tiên đứng yên hoặc môi trường trong nhà an toàn. Không suy diễn trải nghiệm của người sáng mắt bịt mắt thành trải nghiệm tương đương.

## Ba user stories đã chốt cho MVP

1. **Cảnh báo vật cản:** Là người dùng đang di chuyển trong môi trường an toàn có giám sát, tôi muốn nghe một cảnh báo rất ngắn khi có vật cản nguy cơ ở vùng trước mặt để có thêm thông tin và tự quyết định hành động.
2. **Đọc văn bản:** Là người dùng đang đứng yên, tôi muốn chụp biển/nhãn/tài liệu tiếng Việt và nghe nội dung được đọc lại, đồng thời được báo khi ảnh không đủ rõ hoặc không có chữ.
3. **Hiểu cảnh:** Là người dùng đang đứng yên, tôi muốn yêu cầu mô tả ngắn hoặc hỏi một câu về ảnh và nhận câu trả lời chỉ dựa trên bằng chứng nhìn thấy, có thể là “không xác định được”.

## Câu hỏi nghiên cứu

- **RQ1:** Detection kết hợp depth có cải thiện phát hiện vật cản nguy hiểm so với chỉ detection không?
- **RQ2:** Hệ thống có đáp ứng ba tác vụ cốt lõi với độ chính xác và thời gian phản hồi chấp nhận được không?
- **RQ3:** Prompt an toàn, phản hồi ngắn và cơ chế từ chối có làm giảm hallucination của VQA không?

## MVP và kiến trúc bắt buộc

- Camera -> detection -> depth/risk -> cảnh báo.
- Ảnh -> OCR tiếng Việt -> TTS.
- Ảnh + câu hỏi/yêu cầu -> mô tả hoặc VQA -> TTS.
- Luồng an toàn tách khỏi luồng ngữ nghĩa theo yêu cầu.
- Chỉ một bộ điều phối được phát âm thanh; có priority, cooldown, timeout và xử lý xung đột.
- Ghi timestamp, session, mode, input, model/version, latency, prediction/confidence, final response và error code.

## Quyết định triển khai hiện tại

- MVP dùng các model pretrained và adapter local; không huấn luyện lại detector,
  depth, OCR, VQA, STT hoặc model dịch.
- Detection dùng schema tích hợp `indoor_coco_baseline_v1`; chỉ công bố các lớp
  COCO có ánh xạ trực tiếp và không suy diễn cửa, cầu thang hoặc vật cản ngoài schema.
- OCR trên macOS ưu tiên Apple Vision `vi-VN`, fallback PaddleOCR; mô tả cảnh và
  câu hỏi số lượng/đồ vật được grounded từ detection, BLIP chỉ xử lý một số mẫu
  thuộc tính được hỗ trợ.
- Runtime MVP chạy local sau khi cache model, dùng push-to-talk, microphone
  `auto`, TTS macOS và JSONL session log; không gửi ảnh tới API đa phương thức.
- Fine-tuning không phải tiêu chí đóng MVP, không phải điều kiện để chạy thí
  nghiệm hiện tại và không nằm trên critical path của khóa luận.
- Dataset/taxonomy an toàn 15 lớp trước đây được lưu như tài sản nghiên cứu lịch
  sử. Chỉ khởi động lại nhánh này trong tương lai khi có mục tiêu, nguồn lực,
  human review và protocol đánh giá mới được duyệt.

## Không thực hiện trong MVP

- Ứng dụng Android hoàn chỉnh và fine-tune/tạo custom checkpoint.
- Khoảng cách tuyệt đối chính xác theo mét với mọi camera.
- Điều hướng giao thông/cầu thang không giám sát hoặc thay thế thiết bị hỗ trợ chính.
- Chạy mọi mô-đun liên tục theo thời gian thực.

## Thiết bị đã kiểm tra

- MacBook Pro Apple M1, CPU 8 lõi, GPU 8 lõi có Metal 3, RAM 16 GB.
- Camera FaceTime HD và camera iPhone có thể dùng qua Continuity Camera.
- macOS 15.7.7, Python 3.11.9.
- Micro/loa, push-to-talk và TTS đã có smoke log local; vẫn cần rehearsal chính
  thức trong điều kiện phòng demo.

## Dữ liệu dự kiến và đạo đức

- Vật cản: 200-500 ảnh/video ngắn; nhãn class, bbox, direction, risk và ground truth khoảng cách nếu có.
- OCR: 150-300 ảnh tiếng Việt; transcript chuẩn, điều kiện chụp, góc và độ khó.
- VQA: 100-200 ảnh, 2-3 câu hỏi/ảnh; đáp án, answerability và danh sách chi tiết hallucination.
- Chỉ thu dữ liệu người tham gia sau đồng thuận, phương án bảo vệ dữ liệu, môi trường an toàn, giám sát và phê duyệt cần thiết.

## Tiêu chí đóng MVP

Các tiêu chí bắt buộc từ cẩm nang:

- Ba kịch bản chạy end-to-end trên thiết bị demo và có log.
- Detection và OCR được đánh giá trên test set độc lập.
- Có ít nhất một ablation; ưu tiên cả detection vs detection+depth và VQA prompt.
- Báo cáo P50/P95 latency trên thiết bị thật; lưu dự đoán cấp mẫu.
- VQA có cơ chế từ chối; phân tích ít nhất 20 failure cases.
- Có README/config/seed/model version để tái lập và có video demo dự phòng.

Ngưỡng định lượng là **dự kiến, cần GVHD phê duyệt trước khi khóa protocol**: ưu tiên recall nguy hiểm hơn precision; đặt ngưỡng latency riêng cho cảnh báo và tác vụ theo yêu cầu; không lựa chọn ngưỡng bằng test set.

## Rủi ro chính

- Detection COCO không bao phủ ổ gà, cửa kính, bậc thềm và nhiều vật cản đặc thù.
- Depth monocular chỉ cho gần/trung bình/xa nếu chưa hiệu chuẩn.
- VQA pretrained có thể hallucinate; model tải lần đầu qua mạng nhưng inference
  hiện tại chạy local sau khi cache đủ.
- Ultralytics/weights áp dụng AGPL-3.0; cần quyết định giấy phép trước phát hành/deploy.
- Deadline, yêu cầu offline, khả năng tiếp cận người dùng và phê duyệt đạo đức chưa được xác nhận.
- Nếu mở rộng bằng fine-tuning trong tương lai, dữ liệu cũ phải qua lại semantic
  audit và human review; không được xem là train-ready chỉ vì validator cấu trúc đạt.

## Chủ sở hữu và mốc

- Sinh viên: quyết định phạm vi/đạo đức/chi phí, thu thập dữ liệu, xác minh nội dung cá nhân trong luận văn.
- Codex: triển khai repo, baseline, test, protocol, phân tích, bản thảo và tài liệu bảo vệ; không tự tạo dữ liệu hoặc kết quả.
- Mốc tương đối: xem `roadmap.md`; ngày bắt đầu/deadline thật còn cần xác nhận.
