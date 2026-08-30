# Giai đoạn 1 - Nghiên cứu liên quan

Cập nhật: 2026-08-04.

## Sản phẩm

- `sources.json`: nguồn dữ liệu có cấu trúc cho 20 tài liệu đã xác minh.
- `references.bib`: BibTeX trung tính; cần đổi style theo quy định của trường khi nhận template.
- `research-gap.md`: tổng hợp khoảng trống và đóng góp dự kiến, không tuyên bố novelty tuyệt đối.
- `literature-matrix.xlsx`: workbook để lọc, đối chiếu RQ và viết Chương 2.

## Quy trình xác minh

1. Ưu tiên trang hội nghị/tạp chí, arXiv của tác giả, tổ chức tiêu chuẩn hoặc tài liệu chính thức.
2. Xác nhận tiêu đề, tác giả, năm, venue và DOI/ID hoặc URL.
3. Chỉ ghi số liệu khi xuất hiện trong abstract, bảng, project page chính thức hoặc toàn văn gốc.
4. Hạn chế được lấy từ phạm vi/thiết kế của nguồn; các suy luận cho SecondEye được đặt riêng ở cột `Liên hệ SecondEye`.
5. Ngày xác minh cho toàn bộ nguồn: 2026-08-04.

## Lưu ý phiên bản

- VizWiz-FewShot có thống kê hơi khác giữa bản xuất bản và bản arXiv cập nhật; matrix dùng số của arXiv và ghi rõ điều này.
- Reliable VQA có số coverage khác nhẹ giữa các phiên bản; matrix dùng số trên trang ECVA của bản ECCV.
- ViTextVQA xuất hiện dưới preprint 2024 và version of record 2026; matrix giữ năm dataset/preprint là 2024 và ghi DOI bản tạp chí.
- YOLO26 và PaddleOCR 3.0 là nguồn rất mới/technical report. Kết quả của nhà phát triển không được thay cho benchmark SecondEye trên Mac M1.

## Ranh giới diễn giải

- Literature matrix không chứa kết quả thí nghiệm của SecondEye.
- Số liệu từ nguồn khác không được dùng để tuyên bố latency/accuracy trên thiết bị demo.
- `Khoảng trống nghiên cứu` là tổng hợp có điều kiện, cần GVHD phê duyệt và tiếp tục cập nhật nếu tìm thấy công trình gần hơn.
