# Checklist sẵn sàng bảo vệ

Chỉ đánh dấu khi có đường dẫn bằng chứng và đã chạy kiểm tra; không đánh dấu theo kế hoạch.

## Phạm vi và bằng chứng nghiên cứu

- [ ] Ba kịch bản MVP chạy end-to-end trên thiết bị demo.
- [ ] RQ1 có kết quả detection-only và detection+depth trên cùng test set.
- [ ] RQ2 có task success và P50/P95 latency của cả ba tác vụ.
- [ ] RQ3 có prompt baseline, prompt an toàn và prompt từ chối trên cùng VQA test set.
- [ ] Detection và OCR được đánh giá trên test set độc lập đã khóa.
- [ ] Có ít nhất một ablation hợp lệ; cấu hình chỉ khác thành phần đang khảo sát.
- [ ] Có dự đoán cấp mẫu, metrics script và manifest/hash test set.
- [ ] Có ít nhất 20 failure cases được phân nhóm và thảo luận.
- [ ] Không có số liệu, trích dẫn hoặc phản hồi người dùng bịa đặt.

## Hệ thống và an toàn

- [ ] Luồng an toàn và luồng ngữ nghĩa tách rời; VQA/OCR không nghẽn cảnh báo.
- [ ] Chỉ một audio orchestrator phát TTS; priority/cooldown/xung đột có test.
- [ ] VQA từ chối khi ảnh/câu hỏi không đủ bằng chứng.
- [ ] Camera che/tối, không chữ, micro/mạng lỗi và VLM timeout có phản hồi rõ.
- [ ] Giao diện có nút lớn, push-to-talk, dừng, lặp lại và không phụ thuộc màu.
- [ ] Demo và luận văn nêu rõ SecondEye không thay thế gậy trắng/thiết bị chuyên dụng.
- [ ] Không user study nếu thiếu đồng thuận, bảo vệ dữ liệu, giám sát và phê duyệt.

## Tái lập

- [ ] README chạy được từ môi trường sạch.
- [ ] Python/dependency/model/config/seed/commit/device được ghi.
- [ ] Không có API key, PII hoặc đường dẫn riêng tư trong mã/log/tài liệu.
- [ ] Mỗi bảng/hình luận văn truy được về script và kết quả thật.
- [ ] Đã chạy clean-environment test hai lần.

## Luận văn và bảo vệ

- [ ] Sáu chương trả lời trực tiếp 3 RQ và phân biệt kết quả/suy luận/tương lai.
- [ ] Có kiến trúc hai luồng, pipeline, state machine và audio priority.
- [ ] Có literature matrix, ablation, latency, failure cases, an toàn/đạo đức/giới hạn.
- [ ] Slide khớp thời lượng bảo vệ; mỗi số liệu có nguồn.
- [ ] Có script thuyết trình, Q&A phản biện và hai rehearsal có bấm giờ.
- [ ] Có dữ liệu demo cố định, checklist demo và video dự phòng.

