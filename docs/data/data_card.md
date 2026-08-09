# Data card - SecondEye MVP

Phiên bản: 1.1.0
Cập nhật: 2026-08-09
Trạng thái: **obstacle public v1.1 đã khóa**; **276 ảnh, 603 bbox**, đủ 15/15
lớp và mỗi lớp có ít nhất 20 bbox. Dự án không tự chụp ảnh; OCR/VQA vẫn là **0**.

## 1. Tóm tắt

SecondEye dự kiến tạo ba tập đánh giá nhỏ cho nguyên mẫu hỗ trợ người khiếm thị:

| Thành phần | Quy mô dự kiến | Đơn vị | Mục đích chính |
|---|---:|---|---|
| Obstacle | tối thiểu 250 | ảnh hoặc frame đã chọn | RQ1, detection/risk và failure analysis |
| OCR tiếng Việt | 150–300 | ảnh | RQ2, CER/WER và task success |
| VQA/scene | 100–200 | ảnh, 2–3 câu hỏi/ảnh | RQ2–RQ3, answerability và hallucination |

Dataset obstacle v1.1 có 207 ảnh train và 69 ảnh validation, kế thừa pilot 80
ảnh và bổ sung 196 ảnh đã review từ Open Images V7 validation/test và ADE20K
validation. Đây là tập development để fine-tune/so sánh cấu hình; chưa phải test
set độc lập chứng minh hiệu quả cuối hay an toàn điều hướng.

Quy mô OCR/VQA trong bảng vẫn là target thiết kế; quy mô obstacle là dữ liệu đã
có. Tập obstacle phục vụ nghiên cứu có kiểm soát trên Mac M1; không chứng minh
an toàn ngoài đường hoặc khả năng điều hướng độc lập.

## 2. Động cơ và RQ

- **RQ1:** so sánh detection-only với detection + depth/risk trên cùng test set khóa.
- **RQ2:** đo độ chính xác, task success và P50/P95 latency của ba tác vụ.
- **RQ3:** so sánh prompt cơ bản với prompt ngắn/an toàn/có từ chối trên cùng VQA test set.

Data card này áp dụng cách tài liệu hóa motivation, composition, collection, use và maintenance của [Datasheets for Datasets](https://arxiv.org/abs/1803.09010).

## 3. Sử dụng phù hợp và không phù hợp

### Phù hợp

- Development: sửa code, chọn ngưỡng, prompt và tiền xử lý.
- Validation khóa: so sánh cấu hình trong giai đoạn phát triển và failure cases.
- Demo có kiểm soát trong nhà, người vận hành đứng yên hoặc có giám sát.

### Không phù hợp

- Huấn luyện hệ thống điều hướng ngoài đường hoặc tuyên bố chứng nhận an toàn.
- Suy ra trải nghiệm của toàn bộ cộng đồng người mù/thị lực kém từ mẫu nhỏ.
- Nhận dạng danh tính, khuôn mặt, cảm xúc, bệnh lý hoặc đặc điểm nhân khẩu học.
- Dùng nhãn `distance_band` như khoảng cách tuyệt đối theo mét nếu không có phép đo độc lập.
- Công bố hay tải dữ liệu có thông tin cá nhân lên dịch vụ ngoài phạm vi đã được đồng thuận.

## 4. Thành phần và đơn vị dữ liệu

Một dòng trong `sample_manifest.csv` là một cặp **asset–task**. Cùng ảnh dùng cho nhiều tác vụ có thể có nhiều `sample_id`, nhưng phải giữ cùng `group_id`, `scene_id`, split và hash.

### Obstacle

- Ảnh độc lập hoặc frame chọn từ video/burst.
- Nhãn: bbox, lớp thô/canonical, hướng, hazard, distance band, occlusion và truncation.
- `hazard_label` không đồng nghĩa với lớp COCO; phải xét vị trí, lối đi và khả năng va chạm trong bối cảnh có kiểm soát.

### OCR tiếng Việt

- Ảnh biển báo trong nhà, nhãn hàng, menu/tờ rơi/tài liệu do dự án sở hữu hoặc được phép sử dụng.
- Nhãn vùng chữ, transcript UTF-8 có dấu, layout, hướng và độ đọc được.
- Chuỗi nhạy cảm phải được loại hoặc làm mờ trước khi đưa vào development/test.

### VQA/scene

- 2–3 câu hỏi tiếng Việt/ảnh; bao gồm câu có thể trả lời và không thể trả lời.
- Nhãn reference answer, answerability, bằng chứng cần thấy, hallucination trap và câu từ chối an toàn.
- Không đặt câu hỏi yêu cầu suy đoán danh tính, ý định, sức khỏe hoặc thuộc tính nhạy cảm.

## 5. Nguồn gốc dự kiến

| Nguồn | Trạng thái | Điều kiện nhập tập |
|---|---|---|
| Tự thu trong môi trường an toàn | Không dùng trong v1.1 | Quyết định phạm vi ngày 2026-08-09: bỏ hoàn toàn ảnh tự chụp |
| Open Images V7 validation/test pixels + verified boxes | Đã dùng | Chỉ nhận ảnh có metadata CC BY 2.0; giữ author, landing URL, license, hash; bbox/relabel annotation theo CC BY 4.0 |
| ADE20K validation | Đã dùng | Semantic class 43 cho `column`, các class stair cho bổ sung `stairs_down`; ảnh chỉ cho nghiên cứu/giáo dục phi thương mại, annotation BSD-3-Clause |
| Dataset công khai khác | Chưa dùng | Lưu URL, phiên bản, giấy phép và điều kiện phái sinh trước khi nhập version mới |
| Dữ liệu tổng hợp | Cho phép có đánh dấu | Không thay thế đánh giá trên ảnh camera thật |
| Người tham gia | Không dùng trong v1.1 | Quyết định phạm vi: public/synthetic data only |

Không gộp ảnh lấy tùy ý từ Internet. Mỗi asset phải có `source_origin` và `license` hoặc điều kiện sử dụng cụ thể.

## 6. Đại diện và kế hoạch coverage

Mục tiêu không phải cân bằng nhân khẩu học vì dự án không thu nhãn nhân khẩu học nếu không cần thiết. Coverage tập trung vào điều kiện tác vụ:

- Camera: metadata thiết bị nguồn có thể thiếu; không tuyên bố đại diện camera Mac/iPhone đích.
- Ánh sáng: đủ sáng, thiếu sáng trong nhà, ngược sáng.
- Góc/độ khó: chính diện, xiên, rung/mờ, che khuất.
- Obstacle: trái/giữa/phải; gần/trung bình/xa/không xác định; lớp COCO và vật cản ngoài COCO.
- OCR: chữ in/biển/nhãn; cỡ chữ và nền đa dạng; tiếng Việt có dấu.
- VQA: mô tả, thuộc tính, đọc chữ, đếm đơn giản, câu không answerable.

Các lát cắt thực tế sẽ được báo cáo từ manifest sau thu thập; không được điền số dự kiến thành số quan sát.

## 7. Gán nhãn và đảm bảo chất lượng

- Guide được pilot trên 10–20 mẫu development trước khi nhập batch public lớn.
- Tất cả nhãn test và nhãn an toàn/hazard phải có reviewer độc lập hoặc adjudication được ghi lại.
- VQA test: reviewer kiểm tra answerability, reference answer và hallucination trap.
- OCR test: reviewer đối chiếu transcript với ảnh gốc ở mức phóng đại phù hợp.
- Bbox phải nằm trong kích thước ảnh; transcript giữ Unicode; ID không chứa dữ liệu cá nhân.
- Không sửa guide sau khi xem kết quả test mà không tăng version và chạy lại đánh giá liên quan.

## 8. Split và leakage

- Split obstacle v1.1: 75% train, 25% validation theo **group**, không theo dòng/ảnh.
- Toàn bộ cùng scene, burst, video, near-duplicate hoặc cùng asset qua nhiều tác vụ phải ở cùng split.
- Validation hiện tại dùng để chọn cấu hình; cần tạo test độc lập trước khi báo cáo
  kết quả cuối của khóa luận.
- Manifest test được khóa bằng hash trước thí nghiệm cuối.

Chi tiết tại `split_protocol.md`; validator trong `src/secondeye/data/protocol.py` bắt lỗi group/scene/video/hash chéo split.

## 9. Dữ liệu cá nhân, đồng thuận và an toàn

- Mặc định tránh thu khuôn mặt, giọng nói, tên, địa chỉ, biển số, màn hình tài khoản và tài liệu cá nhân.
- Consent record được lưu riêng, không đặt trong repository hoặc tên file.
- Dữ liệu `pending`/`withdrawn` chỉ ở `quarantine`; dữ liệu người tham gia yêu cầu `approved` trước khi dùng.
- Người tham gia có thể dừng mà không cần giải thích; quy trình rút đồng thuận phải ánh xạ từ mã giả danh sang asset để xóa.
- Không yêu cầu người sáng mắt bịt mắt để mô phỏng trải nghiệm người mù; không thu cảnh đi bộ ngoài đường trong MVP.
- Không thu dữ liệu người tham gia trong v1.1; mọi mở rộng tương lai cần protocol và phê duyệt riêng.

Luật Bảo vệ dữ liệu cá nhân 91/2025/QH15 có hiệu lực từ 2026-01-01 theo [Cổng thông tin Chính phủ](https://vanban.chinhphu.vn/?classid=1&docid=214590&pageid=27160&typegroup=). Việc áp dụng cụ thể phải được xác nhận với nhà trường/người phụ trách; tài liệu này không phải tư vấn pháp lý.

## 10. Phân phối, bảo mật và bảo trì

- Raw data phải được lưu local trên vùng được mã hóa bằng cơ chế của hệ điều hành và bị `.gitignore` loại trừ; trạng thái mã hóa phải được kiểm tra trước collection.
- Chỉ manifest đã khử định danh và thống kê tổng hợp mới được cân nhắc commit.
- Không tải raw data lên API/VQA cloud trước khi quyết định riêng tư/chi phí và rà soát đồng thuận.
- Version theo `major.minor.patch`; mọi thay đổi mẫu/split/nhãn phải có changelog.
- Mỗi bản phát hành lưu manifest hash, annotation hash, protocol version và ngày khóa.
- Khi nhận yêu cầu rút dữ liệu, xóa asset, annotation dẫn xuất và bản sao; tăng version và ghi tác động đến kết quả.

## 11. Giới hạn đã biết

- Quy mô nhỏ, dự kiến thiên về trong nhà và thiết bị demo cụ thể.
- Nhãn hazard phụ thuộc ngữ cảnh; inter-rater agreement có thể thấp.
- Khoảng cách từ ảnh đơn không có hiệu chuẩn chỉ là band tương đối.
- OCR/VQA tiếng Việt có thể không đại diện vùng miền, chữ viết tay hoặc tài liệu phức tạp.
- Không có ảnh từ camera triển khai đích, vì vậy metric public không chứng minh khả năng tổng quát trên camera Mac/iPhone hoặc trong nhà/trường học thực tế.
- Dataset public có thể khác góc nhìn, độ phân giải, ánh sáng và phân bố cảnh của người dùng mục tiêu.

## 12. Trường còn phải điền

- Data owner/controller: **TBD**.
- Ngày kết thúc dự án và thời hạn lưu trữ: **TBD**.
- Mẫu consent dễ tiếp cận và kênh rút đồng thuận: **TBD**.
- Pilot obstacle: **80 ảnh/212 bbox; 60 development, 20 validation; 17 ảnh bị loại thủ công vì người/ngoài trời/sản phẩm rời và các ứng viên mờ bị loại tự động**.
- Dataset công khai thực dùng: **Open Images V7 validation/test pixels, V5 bbox;
  ADE20K Scene Parsing validation**. Điều kiện chi tiết tại
  `public_dataset_license_review_v1_1.md`.
- Obstacle v1.1: **276 ảnh/603 bbox; 207 train, 69 validation; 15/15 lớp, mọi
  lớp >=20 bbox; manifest SHA-256
  `600d8e9c4be184485cebac1685185ad8d97407ae7d6050d8a2140bf47aa3fb32`**.
- Agreement hai annotator: **chưa chạy**; bbox nguồn đã verified và các lớp
  SecondEye đặc thù đã có một lượt visual review, nhưng chưa có reviewer độc lập
  thứ hai. Không được mô tả đây là double annotation.
