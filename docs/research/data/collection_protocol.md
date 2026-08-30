# Public dataset acquisition protocol - SecondEye MVP

Phiên bản: 0.3.0

Cập nhật: 2026-08-09

Trạng thái: protocol được lưu cho dữ liệu đánh giá và hướng phát triển tương lai;
MVP hiện dùng model pretrained, không fine-tune, không tự chụp và không thu dữ
liệu người tham gia.

## 1. Cổng bắt buộc

Trước khi nhập bất kỳ nguồn dữ liệu công khai nào:

1. GVHD duyệt target, annotation guide và split protocol.
2. Xác định data owner, ngày kết thúc và thời hạn lưu bản local.
3. Kiểm tra trang chính thức, phiên bản, giấy phép pixels/annotation và điều kiện phái sinh.
4. Kiểm tra ảnh có người, dữ liệu nhạy cảm hoặc hạn chế sử dụng dù nguồn được công bố công khai.
5. Kiểm tra vùng lưu trữ local, backup được phép và quyền truy cập tối thiểu.

Dữ liệu người tham gia nằm ngoài phạm vi v1.1. Protocol tham chiếu [Luật Bảo vệ dữ liệu cá nhân 91/2025/QH15](https://vanban.chinhphu.vn/?classid=1&docid=214590&pageid=27160&typegroup=), nhưng không thay thế rà soát pháp lý/đạo đức của trường.

## 2. Nguồn và dữ liệu an toàn

- Chỉ tải từ trang chính thức hoặc kho do tác giả dataset công bố.
- Không dùng ảnh lấy tùy ý từ Google Images, mạng xã hội hoặc website không rõ quyền.
- Không coi giấy phép dataset là bằng chứng tự động rằng mọi ảnh người đều phù hợp với mục đích dự án.
- Quarantine ảnh có khuôn mặt nhận dạng được, địa chỉ, màn hình cá nhân, biển số hoặc tài liệu định danh.
- Không dùng SecondEye làm công cụ duy nhất để tránh vật cản trong demo camera.

## 3. Cấu trúc lưu trữ local

```text
data/local/
  indoor_pilot_v1/                 pilot 80 ảnh đã khóa, không sửa
  indoor_dataset_v1_1/             dataset YOLO làm việc đã gán nhãn/review
  public_cache/      archive/metadata tải từ nguồn chính thức
  quarantine/       file chưa xác minh quyền/license/privacy hoặc chờ redaction
  raw/              asset public được chấp nhận, bất biến
  derived/          frame chọn, redaction, resize; có nguồn gốc tới raw
  annotations/      manifest và nhãn đang làm
  releases/v0.1/    manifest/nhãn đã khóa, không chứa consent record
```

`data/local/` không commit. Chỉ thống kê tổng hợp và tài liệu nguồn đã rà soát mới
được cân nhắc đưa lên GitHub.

## 4. ID và đơn vị nhóm

- Dùng `capture_session_id` như mã batch nhập nguồn public, ví dụ `ses_oi_v7_01`.
- Tạo `scene_id` từ ID nguồn/cảnh; không suy diễn hai ảnh độc lập là cùng scene.
- Tạo `video_id` cho mỗi clip.
- `group_id` bao trùm mọi frame/burst/near-duplicate của cùng cảnh và cùng asset dùng qua nhiều tác vụ.
- Không mã hóa tên người/địa điểm nhà riêng vào ID.

Ví dụ: `ses_oi_v7_01`, `scn_oi_ab12`, `vid_source_0001`, `grp_oi_ab12`.

## 5. Pilot

1. Chọn 10–20 ảnh development từ nguồn public có license rõ ràng.
2. Hoàn thành manifest/hash và gán nhãn theo guide.
3. Chạy validator; kiểm tra thủ công near-duplicate.
4. Hai người hoặc annotator + reviewer thử guide trên cùng subset.
5. Sửa ambiguity và tăng version trước khi nhập batch lớn hơn.

Pilot không được tự động đưa vào test; muốn giữ phải qua review giống tập chính.

## 6. Kế hoạch obstacle: tối thiểu 250 mẫu

### Coverage dự kiến

- Nhiều nguồn/camera công khai nếu metadata sẵn có; không tuyên bố đại diện thiết bị đích.
- Sáng thường, thiếu sáng trong nhà, ngược sáng.
- Vật cản trung tâm/bên; nhiều kích thước và che khuất.
- Lớp COCO phổ biến và `other_obstacle` như thùng, dây, mép vật thấp.
- Có mẫu không hazard và mẫu khó/không xác định để đo false alert/abstention.
- Mỗi lớp trong schema 15 lớp cần ít nhất khoảng 20 bbox nếu sau này mở lại nhánh
  custom detector; con số này không phải yêu cầu của MVP pretrained.
- Đợt public v1.1 đã nhập/relabel đủ 10 lớp từng trống, nhưng semantic re-audit
  cho thấy chưa thể dùng để huấn luyện hoặc công bố metric. Không fine-tune trong
  phạm vi hiện tại; mọi lần mở lại phải tăng version và hoàn tất human review.
- `person` chỉ dùng ảnh public có quyền phù hợp và privacy review; ưu tiên không nhận dạng được hoặc đã làm mờ mặt.

### Video/burst

- Lưu video gốc trong quarantine/raw nếu được phép; chọn frame bằng quy tắc trước khi xem prediction.
- Frame liên tiếp hoặc cùng bố cục luôn chung `group_id`.
- Không tăng số mẫu bằng hàng chục frame gần như giống nhau; ghi frame index và khoảng lấy mẫu.
- Nếu đổi vị trí vật thể nhỏ nhưng cảnh/bối cảnh vẫn giống, giữ cùng group để tránh leakage.

### Khoảng cách

- Không có phép đo độc lập từ dataset public thì `distance_band=unknown`.
- Không suy ra mét hoặc band ground truth chỉ từ kích thước bbox/monocular depth.
- Depth/risk chỉ được đánh giá định tính hoặc trên nguồn có ground truth phù hợp được kiểm tra riêng.

## 7. Kế hoạch OCR: 150–300 ảnh

- Dùng dataset public hoặc tài liệu synthetic do dự án tạo, có provenance và quyền sử dụng rõ ràng.
- Bao phủ chữ có dấu, nhiều cỡ/font, nền, góc xiên, chói và mờ có kiểm soát.
- Tránh hồ sơ thật, hóa đơn có dữ liệu cá nhân, thẻ, địa chỉ nhà, số điện thoại/email cá nhân.
- Chọn/tạo cả `clear`, `degraded`, `partial`, `unreadable` để đánh giá từ chối.
- Không dùng cùng template văn bản ở cả development và test nếu bố cục/nội dung tạo near-duplicate; gom theo document/template group.

## 8. Kế hoạch VQA: 100–200 ảnh

- Có thể tái sử dụng ảnh obstacle/OCR nhưng phải giữ cùng group/split.
- Viết 2–3 câu hỏi trước khi xem phản hồi model; lưu question type.
- Bao gồm answerable, partially answerable, unanswerable và một số tình huống unsafe-to-answer có kiểm soát.
- Câu hỏi phải phù hợp người dùng đứng yên; không yêu cầu quyết định băng đường, xuống cầu thang hoặc khoảng cách chính xác.
- Reference answer và hallucination trap do annotator/reviewer tạo độc lập với model.

## 9. Nhập kho từng asset

1. Đặt file vào `quarantine`; không đổi nội dung âm thầm.
2. Quét thủ công dữ liệu nhạy cảm/quyền sử dụng; redaction nếu được phép.
3. Tính SHA-256, kích thước, thiết bị, nguồn gốc, license và consent status.
4. Tạo ID/group trước khi chia split.
5. Chuyển asset đã được phép sang `raw` bất biến; derivative ghi source hash.
6. Gán nhãn, review và đặt `annotation_status=accepted`.
7. Chỉ sau đó mới đưa vào inventory để chia development/test.

## 10. Kiểm soát chất lượng batch nhập

Sau mỗi batch nguồn:

- Đếm asset nhập/loại/quarantine theo lý do.
- Kiểm tra file hỏng, orientation, hash trùng và metadata thiếu.
- Xem contact sheet để phát hiện burst/near-duplicate và sửa group, chưa chia split.
- Kiểm tra coverage strata; không nhập thêm chỉ để cải thiện kết quả model.
- Ghi log protocol version, source/version/license/checksum, người xử lý mã hóa và sự cố.

## 11. Privacy, gỡ nguồn và sự cố

- Không thu người tham gia trong v1.1; `consent_status=not_applicable` không miễn privacy review cho ảnh public.
- Khi nguồn bị gỡ, đổi license hoặc có yêu cầu hợp lệ: quarantine asset liên quan, xóa khỏi release và tăng dataset version.
- Nếu lộ dữ liệu/sai quyền: dừng thu, cô lập bản sao, ghi sự cố và báo người có trách nhiệm; không tự ý tiếp tục dùng.

## 12. Checklist kết thúc acquisition

- [ ] Đạt target hoặc có lý do dừng được ghi rõ.
- [ ] Không còn mẫu `pending` trong development/test.
- [ ] Mọi asset có source/license/consent/hash/group.
- [ ] Tất cả test annotation đã review.
- [ ] Near-duplicate đã gom group trước split.
- [ ] Chạy validator không có error.
- [ ] Cập nhật data card bằng số thật và lý do loại mẫu.
