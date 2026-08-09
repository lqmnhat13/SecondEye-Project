# Annotation guide - SecondEye MVP

Phiên bản: 1.0.0
Cập nhật: 2026-08-09
Áp dụng cho: obstacle, OCR tiếng Việt và VQA/scene.

## 1. Nguyên tắc chung

1. Chỉ gán nhãn những gì có bằng chứng trong ảnh; không đoán danh tính, ý định hoặc thuộc tính nhạy cảm.
2. Dùng `unknown`/`unanswerable` khi bằng chứng không đủ; không ép annotator chọn câu trả lời chắc chắn.
3. Không xem prediction của model trước khi tạo ground truth ban đầu.
4. Development và test dùng cùng định nghĩa nhãn; test được reviewer kiểm tra trước khi khóa.
5. Không đổi nhãn để làm model trông tốt hơn. Mọi adjudication ghi lý do và version guide.
6. ID annotator là mã giả danh; không ghi họ tên/email trong CSV.

## 2. Quy trình một mẫu

1. Kiểm tra quyền sử dụng, consent và dữ liệu nhạy cảm trong manifest.
2. Kiểm tra ảnh mở được, đúng hướng, hash và kích thước.
3. Gán nhãn độc lập theo tác vụ, trạng thái `in_review`.
4. Reviewer kiểm tra quy tắc và bất đồng; không được xem model prediction.
5. Sau adjudication, đặt `review_status=accepted` và cập nhật manifest `annotation_status=accepted`.
6. Mẫu không đủ quyền, hỏng hoặc không thể adjudicate chuyển `quarantine`/`rejected`, không xóa dấu vết quyết định.

## 3. Obstacle annotation

### 3.1 Đơn vị và bbox

- Một dòng cho mỗi vật thể/vùng cản có thể ảnh hưởng tác vụ.
- Bbox dùng pixel `xmin,ymin,xmax,ymax`, gốc ở góc trên trái; `0 <= xmin < xmax <= width` và tương tự cho y.
- Bbox ôm phần nhìn thấy của vật thể. Nếu che khuất, không tưởng tượng phần bị che.
- Vật thể phi cấu trúc như dây/cành có thể dùng bbox bao vùng ảnh hưởng và ghi lớp thô cụ thể.

### 3.2 Lớp

- Schema v1 khóa đúng 15 lớp và class ID tại `indoor_schema_v1.md`.
- `class_raw`: mô tả nhìn thấy bằng tiếng Việt, ví dụ `ghế`, `thùng carton`, `cửa kính`.
- `class_canonical`: chỉ nhận một trong 15 nhãn đã khóa; vật ngoài taxonomy không được ép vào lớp gần giống và phải ghi vào backlog schema.
- Generic `Door`/`Stairs` từ dataset ngoài không được tự động ánh xạ sang trạng thái cửa hoặc hướng cầu thang.
- Mọi thay đổi tên/thứ tự lớp làm tăng major schema version và yêu cầu chuyển đổi toàn bộ label.

### 3.3 Hướng

Theo tâm bbox trong ảnh:

- `left`: vùng bên trái.
- `center`: vùng lối đi trung tâm theo protocol/config.
- `right`: vùng bên phải.

Hướng là thuộc tính hình học của ảnh, không phải chỉ dẫn người dùng phải rẽ.

### 3.4 Hazard label

Controlled vocabulary:

- `hazard_now`: nằm trong vùng di chuyển dự kiến và có nguy cơ va chạm sớm theo bằng chứng cảnh/đo độc lập.
- `potential_hazard`: có thể cản trở nhưng bằng chứng khoảng cách/lối đi chưa đủ.
- `not_hazard`: xuất hiện nhưng không nằm trong lối đi hoặc rõ ràng không gây cản trở hiện tại.
- `unknown`: ảnh không đủ để quyết định.

`hazard_reason` phải nêu bằng chứng ngắn, ví dụ `center_path_measured_near`, `side_object`, `depth_ambiguous`. Lớp `person` hay `chair` tự nó không quyết định hazard.

### 3.5 Distance band

- `near`, `medium`, `far`, `unknown`.
- `distance_evidence`: `measured`, `calibrated_reference`, `scene_reference`, `annotator_estimate`, `none`.
- Chỉ `measured`/`calibrated_reference` mới được lưu giá trị mét trong file đo riêng. `annotator_estimate` không dùng làm ground truth khoảng cách tuyệt đối.
- Ranh giới band phải được khóa sau pilot/calibration và trước test; hiện **TBD**, vì vậy pilot dùng `unknown` trừ khi có đo/reference.

### 3.6 Chất lượng bbox

- `occlusion`: `none`, `partial`, `heavy`.
- `truncation`: `none`, `image_boundary`.
- Không gán bbox cho phản chiếu như vật thể thật; cửa kính/vũng nước cần taxonomy hazard riêng nếu có evidence và được thêm bằng version guide.

### 3.7 Review

- Tất cả `hazard_now`, `unknown` và toàn bộ test phải được reviewer kiểm tra.
- Bất đồng hazard ưu tiên adjudication, không lấy đa số tùy tiện.
- Báo agreement cho nhãn hazard/distance nếu có từ hai annotator; chưa có số liệu thì ghi `chưa chạy`.

## 4. OCR annotation

### 4.1 Vùng chữ

- Một dòng cho mỗi vùng đọc hợp lý; polygon lưu dưới dạng danh sách cặp tọa độ theo chiều kim đồng hồ.
- `orientation`: `0`, `90`, `180`, `270` hoặc `skewed`.
- `layout_type`: `sign`, `label`, `document`, `menu`, `screen`, `other`.

### 4.2 Transcript

- Ghi đúng Unicode tiếng Việt, phân biệt hoa/thường và dấu câu nhìn thấy.
- Không tự sửa chính tả của nguồn; nếu chữ in sai, transcript giữ nguyên.
- Khoảng trắng liên tiếp chuẩn hóa thành một khoảng; xuống dòng ghi `\\n` khi ảnh thể hiện cấu trúc dòng cần đánh giá.
- Ký tự không đọc được dùng `�` tại đúng vị trí; nếu phần lớn vùng không đọc được, đặt `legibility=unreadable`.
- Không mở rộng chữ viết tắt hoặc dịch nội dung.

### 4.3 Legibility

- `clear`: đọc chắc chắn.
- `degraded`: mờ/xiên/chói nhưng vẫn đọc được phần lớn.
- `partial`: chỉ đọc được một phần.
- `unreadable`: không đủ bằng chứng để tạo transcript tin cậy.

### 4.4 Dữ liệu nhạy cảm

- Nếu thấy họ tên, địa chỉ, số điện thoại, email, mã định danh, tài khoản, khuôn mặt hoặc biển số: `contains_sensitive_text=true`, `redaction_required=true` và chuyển quarantine.
- Sau redaction phải tạo asset/hash mới; bản gốc vẫn theo chính sách xóa/giữ riêng đã duyệt.
- Không dùng dữ liệu nhạy cảm làm ví dụ trong luận văn hoặc slide.

### 4.5 Review

- Test transcript phải được reviewer đối chiếu ký tự–ký tự.
- Bất đồng dấu tiếng Việt được adjudicate bằng ảnh gốc; không dùng output OCR làm ground truth.
- CER/WER chỉ tính trên vùng có ground truth phù hợp; báo riêng `unreadable`/abstention.

## 5. VQA và mô tả cảnh

### 5.1 Tạo câu hỏi

Mỗi ảnh 2–3 câu, ưu tiên tiếng Việt tự nhiên và ý định thực tế:

- `scene`: mô tả ngắn điều quan trọng.
- `object_attribute`: vật/thành phần nhìn thấy.
- `text_reading`: câu trả lời phụ thuộc chữ trong ảnh.
- `count`: chỉ khi đối tượng đủ rõ.
- `spatial`: trái/phải/trước ảnh, không suy diễn khoảng cách mét.
- `unanswerable`: cố ý thiếu bằng chứng để đánh giá từ chối.

Không hỏi danh tính người, cảm xúc, chủng tộc, sức khỏe, ý định, địa chỉ chính xác hoặc lời khuyên điều hướng nguy hiểm.

### 5.2 Answerability

- `answerable`: ảnh chứa đủ bằng chứng cho một câu trả lời ngắn, ổn định.
- `partially_answerable`: trả lời được một phần nhưng cần nêu giới hạn.
- `unanswerable`: mờ/che khuất/ngoài khung hoặc yêu cầu suy đoán.
- `unsafe_to_answer`: câu hỏi đòi hỏi hướng dẫn nguy hiểm hay thuộc tính nhạy cảm dù ảnh có chi tiết liên quan.

`unanswerable_reason`: `blur`, `too_small`, `occluded`, `out_of_frame`, `requires_external_knowledge`, `sensitive_inference`, `unsafe_navigation`, `ambiguous`, `other`.

### 5.3 Reference answer

- Một câu ngắn, chỉ nêu điều quan sát được; không thêm kiến thức ngoài ảnh.
- Với `partially_answerable`, trả lời phần chắc chắn rồi nói giới hạn.
- Với `unanswerable`, dùng câu từ chối tự nhiên, ví dụ: `Tôi không xác định được từ ảnh này.`
- Với `unsafe_to_answer`, nêu giới hạn và nhắc không dựa vào hệ thống để điều hướng nguy hiểm khi phù hợp.

### 5.4 Bằng chứng và hallucination trap

- `required_evidence`: vùng/vật/chữ tối thiểu cần nhìn thấy để trả lời.
- `hallucination_traps`: các chi tiết dễ bị bịa, ví dụ màu bị cháy sáng, chữ quá nhỏ, vật bị cắt khỏi khung.
- Khi chấm, một chi tiết không có bằng chứng nhưng không ảnh hưởng an toàn là hallucination thường; chi tiết có thể dẫn tới hành động sai/nguy hiểm là hallucination nghiêm trọng.

### 5.5 Review

- Test phải có reviewer độc lập cho answerability, reference answer và trap.
- Reviewer không được xem phản hồi của prompt đang được đánh giá.
- Nếu hai câu trả lời tham chiếu đều hợp lệ, lưu rubric/chấp nhận biến thể thay vì ép một chuỗi duy nhất.

## 6. Adjudication và versioning

- Bất đồng được ghi: `sample_id`, trường, nhãn A/B, quyết định cuối, lý do, reviewer và guide version.
- Thay đổi định nghĩa nhãn làm tăng minor/major version; sửa typo không đổi nghĩa tăng patch.
- Sau khi test khóa, thay đổi nhãn test chỉ được phép để sửa lỗi annotation có bằng chứng; phải lưu trước/sau và đánh giá lại mọi cấu hình liên quan.

## 7. Ví dụ lỗi cần tránh

- Chia hai frame liên tiếp của cùng video sang development và test.
- Gắn `near` chỉ vì bbox lớn mà không có đo/reference.
- Sửa transcript sai chính tả thành câu đúng.
- Viết reference VQA dài, chứa chi tiết không nhìn thấy.
- Dùng output model để quyết định ground truth.
- Ghi tên người tham gia trong `sample_id`, đường dẫn hoặc notes.
