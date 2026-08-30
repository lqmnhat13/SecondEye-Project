# Indoor dataset v1.2: semantic re-audit cho hướng phát triển tương lai

Ngày audit: 2026-08-09. Cập nhật phạm vi: 2026-08-30.

Trạng thái: **lưu trữ/staging, không thuộc critical path của MVP pretrained**.
Dự án hiện không fine-tune. Tài liệu này bảo tồn kết quả audit và xác định các
điều kiện chỉ áp dụng nếu sau này dự án quyết định xây dựng custom detector. Việc
review còn pending không được mô tả là blocker của MVP hiện tại.

## Phạm vi và nguyên tắc

- `data/local/indoor_dataset_v1_1` là baseline bất biến; không sửa trực tiếp.
- `data/local/indoor_dataset_v1_2` là staging mới và không được đưa lên Git.
- 15 lớp giữ nguyên theo `indoor_schema_v1.md`.
- Escalator không phải cầu thang bộ và luôn bị loại. Không đoán trạng thái cửa
  hoặc hướng cầu thang khi ảnh không đủ bằng chứng.
- Một ảnh chỉ được nhập khi toàn bộ vật thuộc 15 lớp đã được dense-label, privacy
  trong pixel và metadata đã được kiểm tra, hai reviewer đã ghi danh tính, và
  bất đồng đã được adjudicate.
- Reviewer AI chỉ hỗ trợ sàng lọc; reviewer thứ hai bắt buộc là người để đạt tiêu
  chí nghiên cứu. Prediction của model không được dùng làm ground truth.

## Kết quả re-audit v1.1

Hai lượt AI đã xem toàn bộ 276 ảnh. Validator cũ vẫn đúng về cấu trúc file, nhưng
không phát hiện các lỗi semantic sau:

- 114/276 ảnh thuộc nhóm loại với độ tin cậy cao do outdoor, product/staged,
  privacy, escalator, sai domain/sai lớp hoặc near-duplicate.
- Nếu chỉ áp các loại chắc chắn, còn 162 ảnh. Độ phủ bị tụt nghiêm trọng:
  `trash_bin` còn 1 bbox, `backpack_bag` 2, `doorway_open` 7,
  `door_closed` 5, `stairs_up` 9; vì vậy không thể quảng bá bản lọc này thành
  dataset đủ 15 lớp.
- Phát hiện nhiều cặp bbox chồng gần hoàn toàn: sofa/ghế đơn và
  table_desk/cabinet. Các bbox này phải sửa hoặc loại sau khi xem ảnh gốc.
- Các ảnh ADE20K column và một số Open Images chỉ có nhãn theo lớp nguồn, không
  phải dense annotation cho schema SecondEye; cần relabel toàn ảnh.
- EXIF không cho thấy GPS/Artist/Copyright/UserComment hoặc serial nhạy cảm trong
  các ảnh đã quét. Điều này không loại bỏ rủi ro khuôn mặt, nhãn vận chuyển, thẻ
  tên hoặc địa chỉ xuất hiện trực tiếp trong pixel.

Danh sách ảnh/bbox chi tiết được giữ trong vùng review local, không công khai cùng
dataset.

## Candidate test độc lập

Hai pool Open Images chưa từng xuất hiện trong v1.1 đã được tạo:

| Pool | Candidate | AI image-level accept | Vai trò dự kiến |
| --- | ---: | ---: | --- |
| Open Images validation, ID mới | 210 | 39 | bổ sung development, không ưu tiên test |
| Open Images test split, ID mới | 195 | 40 | candidate test |

Chỉ 79/405 ảnh phù hợp ở mức image-level. Chưa ảnh nào được coi là test hợp lệ vì
vẫn thiếu dense relabel và human reviewer. Không hạ tiêu chí để ép đủ số lượng.
ADE20K training được dùng làm pool bổ sung cho column/cầu thang bộ; class 97
`escalator` bị loại ngay ở bước trích xuất.

Pool ADE20K training đã tạo 76 ảnh (30 cảnh column, 50 cảnh stairs, một số ảnh
chứa cả hai) với 154 bbox candidate. Trên toàn bộ 481 ảnh nguồn mới của ba pool,
audit không thấy exact duplicate hoặc pHash Hamming distance <= 5 với v1.1 hay
giữa các pool; không phát hiện các trường EXIF nhạy cảm đã liệt kê trong protocol.
Đây chỉ là kiểm tra kỹ thuật, không thay thế privacy review trong pixel.

Hai lượt AI review ADE20K lần lượt accept 14/76 và 9/76, đồng thuận accept 8 ảnh.
Như vậy candidate test nghiêm hiện chỉ có tối đa 48 ảnh đồng thuận ở mức
image-level (40 Open Images test-split từ lượt AI độc lập và 8 ADE đồng thuận),
chưa đạt target 100--150 và chưa ảnh nào hoàn tất dense-label/human adjudication.

### AI-assisted operation trên 48 candidate

Workflow AI-assisted lịch sử đã xem lại 48 ảnh và tạo artifact local. Script cũ
không còn được duy trì trong source hiện tại; các số dưới đây chỉ là hồ sơ audit,
không phải một lệnh tái lập còn được hỗ trợ:

- 38 ảnh accepted ở mức image QA để tạo labeling task;
- 3 ảnh rejected do crop/product/staged hoặc thiếu ngữ cảnh test;
- 7 ảnh `needs_human_confirmation`, gồm bốn ảnh có màn hình chưa thể privacy-clear,
  hai cảnh chưa chắc thuộc home/school và một ảnh column mờ/cần vẽ lại bbox;
- 81 bbox dự thảo được kế thừa từ bbox/mask nguồn có kiểm tra và đều mang trạng
  thái `proposed`; không bbox nào được coi là dense ground truth;
- gate đã chạy thử và fail như thiết kế vì còn 271 review nguồn pending, chưa có
  test image đã adjudicate và chưa human/dense review.

AI review không thay thế hai reviewer người trong báo cáo học thuật. Chỉ các hàng
trong `accepted_candidates.csv` được đưa vào CVAT; bảy hàng pending nằm trong
`human_confirmation.csv` để chủ dự án xác nhận ở ảnh gốc 100% zoom.

## Điều kiện nếu mở lại nhánh fine-tuning trong tương lai

Tooling audit v1.2 lịch sử không còn nằm trong source public hiện tại, vì vậy
không giữ lệnh chạy đã hết hiệu lực trong tài liệu này. Nếu mở lại nhánh này,
phải tạo/version hóa gate mới và bảo đảm gate từ chối dữ liệu khi:

- còn decision/adjudication `pending`;
- chưa đánh dấu `dense_15_class_review=complete`;
- domain không phải indoor hoặc privacy chưa `safe/cleared`;
- reviewer thứ hai không có `reviewer_2_type=human`;
- test set chưa nằm trong khoảng 100--150 ảnh.

Chỉ khi có quyết định phạm vi mới, human review hoàn tất và semantic gate mới
pass mới được dùng các lệnh nghiên cứu còn được duy trì:

```bash
secondeye-detection validate --dataset data/local/indoor_dataset_v1_2
secondeye-validate-manifest data/local/indoor_dataset_v1_2/sample_manifest.csv --require-rows
secondeye-detection train --dataset data/local/indoor_dataset_v1_2
```

Không dùng validation/test để chọn confidence hoặc hyperparameter. Báo cáo cuối
phải nêu source split, giấy phép, số ảnh/bbox mỗi lớp, kết quả human review,
validator, mAP/precision/recall theo lớp, latency và failure cases.
