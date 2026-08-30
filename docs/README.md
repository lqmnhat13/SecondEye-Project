# Tài liệu SecondEye

Cập nhật: 2026-08-30.

Thư mục tài liệu được tổ chức theo mục đích sử dụng. Điểm bắt đầu cho người mới
là `README.md` ở thư mục gốc, sau đó đọc tài liệu trong `current/` và `guides/`.

## Đang sử dụng

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
  đưa lên repository công khai.
- `references/`: cẩm nang/yêu cầu làm khóa luận.
- `design/`: sơ đồ kiến trúc nguồn `.drawio` và bản PNG. Sơ đồ thể hiện kiến trúc
  mục tiêu nghiên cứu; README/mã nguồn là thẩm quyền cho hành vi runtime thực tế.

## Chính sách

- Không lưu kết quả sinh tự động trùng với nguồn Markdown nếu chúng đã lỗi thời.
- `logs/` và `results/` ở thư mục gốc là artifact chạy máy, không phải tài liệu.
- Model/dataset/output lớn tiếp tục nằm ngoài Git theo `.gitignore`.
- Khi tạo lại DOCX/PDF từ Markdown, tên file phải có ngày hoặc phiên bản và nội
  dung phải được kiểm tra với phạm vi pretrained-only hiện hành.
