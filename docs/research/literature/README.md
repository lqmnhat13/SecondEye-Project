# Giai đoạn 1 - Nghiên cứu liên quan

Cập nhật: 2026-08-30.

## Sản phẩm

- `sources.json`: nguồn dữ liệu có cấu trúc cho 24 tài liệu đã xác minh.
- `references.bib`: BibTeX trung tính; cần đổi style theo quy định của trường khi nhận template.
- `research-gap.md`: tổng hợp khoảng trống và đóng góp dự kiến, không tuyên bố novelty tuyệt đối.
- `literature-matrix.xlsx`: workbook để lọc, đối chiếu RQ và viết Chương 2. Chỉ
  dùng khi workbook đã được tái tạo từ `sources.json` cùng ngày xác minh; bản
  local ngày 2026-08-04 hiện cần regenerate trước khi dùng.

## Quy trình xác minh

1. Ưu tiên trang hội nghị/tạp chí, arXiv của tác giả, tổ chức tiêu chuẩn hoặc tài liệu chính thức.
2. Xác nhận tiêu đề, tác giả, năm, venue và DOI/ID hoặc URL.
3. Chỉ ghi số liệu khi xuất hiện trong abstract, bảng, project page chính thức hoặc toàn văn gốc.
4. Hạn chế được lấy từ phạm vi/thiết kế của nguồn; các suy luận cho SecondEye được đặt riêng ở cột `Liên hệ SecondEye`.
5. Đợt nền xác minh 20 nguồn: 2026-08-04.
6. Đợt cập nhật 2026-08-30 tìm trên arXiv, CVF Open Access và ACL Anthology,
   ưu tiên nguồn sơ cấp năm 2026 về BLV visual assistance, VLM reliability và
   VQA tiếng Việt; bổ sung S21-S24.

## Lưu ý phiên bản

- VizWiz-FewShot có thống kê hơi khác giữa bản xuất bản và bản arXiv cập nhật; matrix dùng số của arXiv và ghi rõ điều này.
- Reliable VQA có số coverage khác nhẹ giữa các phiên bản; matrix dùng số trên trang ECVA của bản ECCV.
- ViTextVQA xuất hiện dưới preprint 2024 và version of record 2026; matrix giữ năm dataset/preprint là 2024 và ghi DOI bản tạp chí.
- YOLO26 và PaddleOCR 3.0 là nguồn rất mới/technical report. Kết quả của nhà phát triển không được thay cho benchmark SecondEye trên Mac M1.
- CHI 2026 diary study S21 bổ sung bằng chứng sử dụng thực tế từ 20 người BLV;
  không được dùng tỷ lệ sai/từ chối của hệ thống trong nghiên cứu đó như kết quả
  của SecondEye.
- ACL 2026 S22 cho thấy quality score bám bằng chứng giúp giảm tin nhầm câu trả
  lời sai; HALP S23 là hướng early-abstention tương lai, chưa được triển khai
  trong MVP 0.3.0.
- AutoViVQA S24 là preprint/dataset sinh tự động; dùng để mở rộng taxonomy câu
  hỏi tiếng Việt, không thay test set camera mục tiêu hoặc human review.

## Ranh giới diễn giải

- Literature matrix không chứa kết quả thí nghiệm của SecondEye.
- Số liệu từ nguồn khác không được dùng để tuyên bố latency/accuracy trên thiết bị demo.
- `Khoảng trống nghiên cứu` là tổng hợp có điều kiện, cần GVHD phê duyệt và tiếp tục cập nhật nếu tìm thấy công trình gần hơn.
- Trước khi nộp bản cuối phải chạy lại backward/forward citation search và kiểm
  tra version of record của các preprint 2026.
