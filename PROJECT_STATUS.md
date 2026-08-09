# Trạng thái dự án SecondEye

Cập nhật: 2026-08-09
Nguồn yêu cầu: `output/pdf/Cam_nang_KLTN_SecondEye.pdf`, phiên bản 1.1.

## Quy ước

| Trạng thái | Ý nghĩa |
|---|---|
| Hoàn thành | Đã có sản phẩm và đã chạy kiểm tra tương ứng |
| Đang làm | Đang triển khai hoặc cần thêm bằng chứng để đóng |
| Chưa làm | Chưa bắt đầu |
| Bị chặn | Không thể quyết định đúng nếu thiếu thông tin/ủy quyền |
| Ngoài phạm vi | Đã quyết định không thực hiện trong phiên bản hiện tại |

## Bảng công việc

| Giai đoạn | Công việc | Trạng thái | Bằng chứng / điều kiện đóng |
|---|---|---|---|
| 0 | Đọc cẩm nang và kiểm kê repository | Hoàn thành | Đã đọc/trích xuất và render đủ 18 trang; đã kiểm kê repo và máy demo |
| 0 | Project Charter bản nháp | Hoàn thành | `docs/project_charter.md` |
| 0 | Kế hoạch 10 tuần | Hoàn thành | `docs/plan_10_weeks.md` |
| 0 | Checklist tiêu chí bảo vệ | Hoàn thành | `docs/defense_readiness_checklist.md` |
| 0 | Xác nhận deadline, offline/API và user study | Bị chặn | Cần chủ dự án/GVHD xác nhận; không chặn baseline local |
| 1 | Literature matrix 15-20 nguồn đã xác minh | Hoàn thành | 20 nguồn; `outputs/phase1/literature_matrix.xlsx`, `docs/literature/sources.json` và `docs/literature/references.bib` |
| 1 | Research gap và đóng góp dự kiến | Hoàn thành | `docs/literature/research_gap.md`; diễn giải có ranh giới, chưa tuyên bố novelty tuyệt đối |
| 1 | Định dạng trích dẫn theo chuẩn của trường | Bị chặn | Metadata/BibTeX đã có; cần mẫu hoặc tên chuẩn trích dẫn chính thức của trường |
| 2 | Data card, annotation guide, collection protocol và split | Hoàn thành | `docs/data/`; split 75/25 theo group, scene/video/hash không được chéo split |
| 2 | Manifest templates và leakage validator | Hoàn thành | `data/templates/`, `configs/data_protocol.toml`; 12 unit test mới đã qua |
| 2 | Pilot obstacle public không có dữ liệu cá nhân | Hoàn thành | 80 ảnh/212 bbox; 60 development, 20 validation; schema 15 lớp v1, provenance, privacy review và hai validator đều đạt |
| 2 | Bổ sung và gán nhãn obstacle/OCR/VQA đầy đủ | Đang làm | Obstacle public v1.1 đã khóa: 276 ảnh/603 bbox, đủ 15 lớp, mỗi lớp >=20 bbox, YOLO/manifest validator đạt; OCR/VQA vẫn là 0 |
| 2 | Dữ liệu người tham gia | Ngoài phạm vi | V1.1 đã chốt chỉ dùng public/synthetic data; không tự chụp và không thu người tham gia |
| 3 | Detection YOLO26 local | Đang làm | Pipeline mặc định đã chuyển sang YOLO26m; checkpoint YOLO11n cũ được giữ làm baseline; còn benchmark pretrained, fine-tune YOLO26m và đánh giá trên test set độc lập |
| 3 | OCR baseline | Chưa làm | PaddleOCR tiếng Việt, 20 ảnh development ban đầu |
| 3 | Depth baseline | Chưa làm | Depth Anything V2 nhỏ, ba vùng gần/trung bình/xa |
| 3 | VQA baseline | Bị chặn | Cần quyết định local/API dựa trên offline, riêng tư và chi phí |
| 3 | Speech baseline | Chưa làm | Nhấn-để-nói và TTS hệ điều hành trước |
| 4 | Ba pipeline end-to-end | Chưa làm | Obstacle, Read, Scene/Question đều chạy và có log |
| 4 | State machine và bộ điều phối âm thanh duy nhất | Chưa làm | IDLE/OBSTACLE/READ/SCENE/QUESTION, priority + cooldown |
| 5 | Test set khóa, baseline, ablation | Chưa làm | Lưu dự đoán cấp mẫu và config/seed/commit |
| 6 | Luận văn sáu chương | Chưa làm | Kết quả thật phân biệt với dự kiến/chỗ trống |
| 7 | Slide, kịch bản, Q&A, demo dự phòng | Chưa làm | Hai lần rehearsal có bấm giờ và clean-room run |

## Công việc gần nhất cần tiếp tục

1. Benchmark YOLO26m pretrained COCO trên ảnh/camera, sau đó fine-tune trên `data/local/indoor_dataset_v1_1` khi cần đủ schema 15 lớp; lưu checkpoint và metrics có fingerprint dataset.
2. Chạy benchmark ảnh/camera trên Mac M1; không dùng validation hiện tại để tuyên bố hiệu quả cuối, cần test set độc lập.
3. Chọn nguồn OCR tiếng Việt công khai hoặc synthetic có provenance; OCR/VQA hiện vẫn là 0.
