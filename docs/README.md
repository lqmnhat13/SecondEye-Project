# Tài liệu SecondEye

Cập nhật: 2026-08-30.

Thư mục tài liệu được tổ chức theo mục đích sử dụng. Điểm bắt đầu cho người mới
là `README.md` ở thư mục gốc, sau đó đọc tài liệu trong `current/` và `guides/`.

## Đang sử dụng

- `guides/complete-usage-guide.md`: hướng dẫn sử dụng đầy đủ từ cài đặt, kiến
  trúc logic, toàn bộ CLI đến xử lý lỗi; đây là tài liệu vận hành chính.
- `current/project-charter.md`: phạm vi, RQ, quyết định dùng model pretrained và
  tiêu chí đóng MVP.
- `current/roadmap.md`: kế hoạch triển khai và trạng thái công việc.
- `current/readiness-checklist.md`: checklist thí nghiệm, demo và bảo vệ.
- `guides/local-yolo26-runtime.md`: cách chạy detection/MVP hiện tại; các lệnh
  custom checkpoint được đánh dấu rõ là hạ tầng tương lai.

## Nghiên cứu và bằng chứng

- `research/benchmarks/`: benchmark development đã chạy; không phải test độc lập.
- `research/literature/`: nguồn, BibTeX, research gap và literature matrix.
- `research/data/`: data card/protocol đang dùng cho đánh giá và hồ sơ taxonomy
  an toàn lịch sử. Dữ liệu 15 lớp không phải dependency của MVP pretrained.

## Tài liệu nội bộ

- `private/thesis/`: phiếu đăng ký và đề cương nguồn. Có thông tin cá nhân; không
  đưa lên repository công khai. Phiếu đăng ký là hồ sơ lịch sử; đề cương bản 1.1
  đã được đồng bộ với MVP mã nguồn 0.3.0 ngày 2026-08-30.
- `references/`: cẩm nang/yêu cầu làm khóa luận. `kltn-handbook.pdf` là snapshot
  nguồn nhận ngày 2026-08-03; phải đối chiếu lại với nhà trường trước khi nộp nếu
  có bản phát hành mới.
- `design/`: sơ đồ kiến trúc nguồn `.drawio` và bản PNG. Sơ đồ thể hiện kiến trúc
  runtime 0.3.0 và vẫn giữ ranh giới an toàn/nghiên cứu; README/mã nguồn là thẩm
  quyền cuối cùng cho hành vi thực tế.

## Chính sách

- Không lưu kết quả sinh tự động trùng với nguồn Markdown nếu chúng đã lỗi thời.
- `logs/` và `results/` ở thư mục gốc là artifact chạy máy, không phải tài liệu.
- Model/dataset/output lớn tiếp tục nằm ngoài Git theo `.gitignore`.
- Khi tạo lại DOCX/PDF từ Markdown, tên file phải có ngày hoặc phiên bản và nội
  dung phải được kiểm tra với phạm vi pretrained-only hiện hành.
- Workbook `research/literature/literature-matrix.xlsx` phải được tái tạo từ
  `sources.json` sau mỗi đợt literature search; không dùng workbook có ngày xác
  minh cũ hơn nguồn JSON để viết Chương 2.
