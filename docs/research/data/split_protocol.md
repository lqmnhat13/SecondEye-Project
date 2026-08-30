# Split protocol - group-safe development/test

Phiên bản: 0.1.0  
Cập nhật: 2026-08-04

## 1. Mục tiêu

Ngăn model/prompt/threshold nhìn thấy nội dung gần giống test trong quá trình development. Đơn vị chia là **group**, không phải ảnh/dòng annotation.

Tỷ lệ mặc định trên asset đã được chấp nhận:

- `development`: 75% theo group, dùng cho pilot, tuning, prompt và lựa chọn cấu hình.
- `test`: 25% theo group, khóa trước thí nghiệm cuối.
- `quarantine`: ngoài tỷ lệ; không dùng cho training, tuning hay báo cáo metric.

Tỷ lệ là target gần đúng vì group có kích thước khác nhau. Ưu tiên không leakage và coverage test hơn việc ép đúng tỷ lệ từng mẫu.

## 2. Định nghĩa group

Một `group_id` phải chứa toàn bộ mẫu có nguy cơ chia sẻ tín hiệu gần trùng:

- Mọi frame từ cùng video, burst hoặc đoạn quay liên tục.
- Ảnh chụp cùng cảnh/bố cục/vật thể từ các góc gần nhau trong một phiên.
- Derivative: crop, resize, redaction, frame trích xuất hoặc ảnh nén lại.
- Cùng asset dùng cho obstacle, OCR và VQA.
- OCR của cùng một document/template hoặc VQA của cùng scene, kể cả câu hỏi khác nhau.

Quy tắc bảo thủ: nếu không chắc hai mẫu có độc lập hay không, đặt cùng group.

## 3. Invariant bắt buộc

1. Một `group_id` chỉ thuộc một split.
2. Một `scene_id` chỉ thuộc một split trên toàn bộ tác vụ.
3. Một `video_id` chỉ thuộc một group và một split.
4. Cùng SHA-256 chỉ thuộc một group/split; derivative khác hash vẫn phải chung group.
5. Split không dựa trên prediction, error hoặc confidence của model.
6. Test không được dùng để sửa prompt, ngưỡng, tiền xử lý hoặc taxonomy.
7. Mẫu `pending`/`withdrawn` luôn ở quarantine.

Validator tự động kiểm tra invariant 1–4 và 7; near-duplicate khác hash cần contact sheet/perceptual review thủ công.

## 4. Trình tự chia

1. Hoàn tất quyền sử dụng, manifest, group và annotation review.
2. Loại quarantine; tạo inventory theo group gồm task, số mẫu và strata.
3. Đặt seed cố định `20260804` và ghi seed trong manifest thí nghiệm.
4. Phân tầng ở cấp group theo các thuộc tính quan trọng, không theo model result.
5. Gán group vào development/test để gần 75/25 và giữ test có coverage hợp lý.
6. Chạy validator, xem bảng phân bố và contact sheet chéo split.
7. Lưu bảng `group_id -> split`, protocol version, seed và manifest hash.

Không chạy `random_split` trên từng dòng. Nếu dùng code tối ưu cân bằng, code/config/seed phải được version hóa và kết quả assignment phải lưu thành artifact.

## 5. Strata cần cân bằng

### Chung

- task, nguồn, thiết bị camera, ánh sáng, blur/chói, portrait/landscape.

### Obstacle

- hazard label, direction, distance band/evidence, class canonical, occlusion.

### OCR

- layout type, legibility, orientation, chữ nhỏ/lớn và dạng nền.

### VQA

- question type, answerability và hallucination-risk category.

Không thu thập thuộc tính nhân khẩu học chỉ để stratify nếu chúng không cần cho RQ và chưa có phê duyệt.

## 6. Fine-tune trong tương lai

MVP không yêu cầu fine-tune. Nếu sau này cần:

- Chỉ tách `train`/`validation` bên trong development ở cấp group.
- Test 25% giữ nguyên, không trở lại pool huấn luyện.
- Mọi pretraining data trùng/near-duplicate với test phải được ghi và loại hoặc thảo luận như contamination.

## 7. Khóa test

Trước thí nghiệm cuối:

1. Xác nhận tất cả test annotation `accepted` và reviewed.
2. Xuất manifest test theo thứ tự ổn định.
3. Tính SHA-256 cho manifest, annotation files và từng asset.
4. Ghi protocol version, config, seed, ngày khóa và người phê duyệt.
5. Đặt quyền read-only nếu khả thi; evaluation script chỉ đọc.

Sau khóa, không xem ground-truth test trong quá trình tuning. Prediction cấp mẫu được lưu riêng để phân tích sau khi cấu hình cuối đã cố định.

## 8. Xử lý lỗi/leakage

- Trước khóa: chuyển **toàn bộ group** về một split hoặc quarantine, rồi tái kiểm tra phân bố.
- Sau khóa nhưng trước chạy cuối: tạo version test mới và ghi lý do.
- Sau khi đã báo cáo kết quả: đánh dấu kết quả bị ảnh hưởng, sửa split/version và chạy lại mọi cấu hình so sánh; không chỉ chạy lại model tốt nhất.
- Không giải quyết leakage bằng cách xóa một frame test nhưng giữ các frame gần giống ở development.

## 9. Báo cáo bắt buộc

Mỗi bảng kết quả phải nêu:

- Dataset/protocol version và manifest hash.
- Số group và mẫu theo split/tác vụ.
- Phân bố strata chính và số quarantine/loại bỏ.
- Kiểm tra leakage đã chạy, kết quả và công cụ.
- Config/model/prompt/seed/commit và thiết bị demo.

## 10. Lệnh audit

```bash
python -m secondeye.data.protocol data/local/sample_manifest.csv \
  --require-rows
```

Exit code `0` nghĩa không có lỗi schema/leakage đã mã hóa; không thay thế review near-duplicate bằng mắt hoặc rà soát quyền/đạo đức.
