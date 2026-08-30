# Khoảng trống nghiên cứu và đóng góp dự kiến

Trạng thái: bản nháp cập nhật ngày 2026-08-30 với 24 nguồn; cần giảng viên hướng
dẫn phê duyệt trước khi đưa vào tuyên bố đóng góp chính thức.

## Tổng hợp bằng chứng

Các nghiên cứu VizWiz chứng minh dữ liệu do người khiếm thị tạo khác đáng kể các benchmark thị giác thông thường: ảnh có thể mờ, bố cục không chuẩn, câu hỏi mang tính hội thoại và đôi khi không thể trả lời [S02-S06]. Các nghiên cứu độ tin cậy cho thấy confidence thô không đủ để quyết định trả lời, LVLM thường hallucinate và có thể không nhất quán giữa câu hỏi đối chứng [S07-S09].

Detection và depth đã có các mô hình mạnh, nhẹ [S10-S14], nhưng detection lớp COCO không đồng nghĩa “vật cản nguy hiểm”, còn monocular depth cơ sở thường chỉ cung cấp depth tương đối. Với OCR, PaddleOCR có pipeline thực dụng và đa ngôn ngữ [S15-S16], trong khi các dataset tiếng Việt gần đây chủ yếu tập trung OCR-VQA trên scene text hoặc bìa sách [S17-S18], chưa trực tiếp đại diện ba loại ảnh SecondEye dự kiến: biển báo, nhãn sản phẩm và tài liệu chụp bằng camera người dùng.

Cập nhật năm 2026 làm rõ hai điểm. Diary study với 20 người BLV ghi nhận MLLM
vẫn trả lời sai và từ chối trong sử dụng hằng ngày, cho thấy conversational
assistance cần hành vi hỗ trợ theo mục tiêu chứ không chỉ caption tốt [S21].
Quality score về visual fidelity/contrastiveness giúp người không xem ảnh đánh
giá đúng hơn độ tin cậy của dự đoán VLM [S22], trong khi probe hidden-state có
thể hỗ trợ early abstention nhưng phụ thuộc kiến trúc và chưa phù hợp để tuyên
bố đã có trong MVP [S23]. AutoViVQA mở rộng không gian câu hỏi tiếng Việt nhưng
là dữ liệu xây dựng tự động trên ảnh tuyển chọn, không thay bằng chứng camera
mục tiêu hoặc user study [S24].

## Khoảng trống có thể bảo vệ

1. **Khoảng trống tích hợp an toàn:** Chưa thấy trong 24 nguồn một nghiên cứu đánh giá đồng thời ba tác vụ vật cản, OCR tiếng Việt và VQA trong một prototype có tách luồng an toàn/luồng ngữ nghĩa và chỉ một bộ điều phối âm thanh.
2. **Khoảng trống đánh giá nguy cơ:** Benchmark detection/depth thường chấm đối tượng hoặc depth; chúng không trực tiếp kiểm tra quyết định cảnh báo dựa trên loại vật thể, vùng di chuyển, depth và temporal evidence trong dữ liệu mục tiêu.
3. **Khoảng trống tiếng Việt end-to-end:** Công trình tiếng Việt đã có OCR-VQA
   và VQA sinh tự động quy mô lớn [S17, S18, S24], nhưng chưa trả lời CER/WER,
   task success, latency và lỗi TTS của một trợ lý camera-OCR-speech trên thiết
   bị demo SecondEye.
4. **Khoảng trống độ tin cậy theo tác vụ:** POPE, HallusionBench, Reliable VQA
   và các quality/probe mới [S22, S23] cung cấp khung đo tốt, nhưng chưa thay thế
   đánh giá rule/grounding/fail-safe thực tế bằng rubric đúng-một phần-sai-nguy
   hiểm trên câu hỏi và ảnh mục tiêu của SecondEye.
5. **Khoảng trống bằng chứng trên thiết bị:** Số liệu do paper/vendor công bố không cho biết P50/P95 end-to-end trên Mac M1, cũng không bao gồm camera, điều phối, mạng và TTS.

Các câu trên là tổng hợp trong phạm vi 24 nguồn đã kiểm tra đến ngày 2026-08-30,
không phải tuyên bố rằng tuyệt đối không có nghiên cứu khác. Trước khi nộp luận
văn cần chạy thêm backward/forward citation search và kiểm tra version of record
cho các preprint mới.

## Đóng góp dự kiến

1. **Kiến trúc hệ thống:** Prototype hai luồng với state machine, risk fusion và audio orchestrator có priority/cooldown/timeout. Đóng góp nằm ở thiết kế và kiểm chứng tích hợp, không tuyên bố tạo mô hình nền tảng mới.
2. **Protocol đánh giá tái lập:** Ba tập development/test tách theo scene/video group, data card và annotation guide cho vật cản, OCR tiếng Việt và VQA; lưu prediction cấp mẫu, config, seed, model version và latency breakdown.
3. **Bằng chứng thực nghiệm cho ba RQ:**
   - RQ1: detection so với detection + depth, ưu tiên recall nguy hiểm/cảnh báo sai/latency.
   - RQ2: precision/recall/F1, CER/WER, VQA/task success và P50/P95 end-to-end.
   - RQ3: prompt cơ bản so với prompt ngắn-an toàn và prompt có từ chối; đo hallucination, abstention và consistency.
4. **Phân tích an toàn và failure taxonomy:** Ít nhất 20 lỗi được phân nhóm theo ánh sáng, rung/mờ, vật nhỏ/che khuất, lớp ngoài COCO, chữ khó, câu hỏi không thể trả lời, speech và tích hợp.
5. **Artifact tái lập:** Mã nguồn, config, logs, script metric, README, bảng/hình sinh từ kết quả thật và demo dự phòng.

## Những gì không được tuyên bố

- SecondEye không thay thế gậy trắng, chó dẫn đường hoặc thiết bị điều hướng chuyên dụng.
- Depth tương đối không phải khoảng cách mét chính xác.
- Kết quả trên ảnh mẫu/vendor không phải hiệu năng SecondEye.
- Không tuyên bố hiệu quả với người khiếm thị nếu chưa có đồng thuận, phê duyệt và user study phù hợp.
- Không gọi việc ghép model có sẵn là đóng góp thuật toán mới nếu không có phương pháp và ablation chứng minh.
