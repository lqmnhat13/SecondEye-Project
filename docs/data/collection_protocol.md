# Collection protocol - SecondEye MVP

Phiên bản: 0.1.0  
Cập nhật: 2026-08-04  
Trạng thái: chưa bắt đầu thu tập chính.

## 1. Cổng bắt buộc

Trước bất kỳ phiên thu tập chính nào:

1. GVHD duyệt target, annotation guide và split protocol.
2. Xác định data owner, ngày kết thúc, thời hạn lưu và kênh yêu cầu xóa.
3. Kiểm tra quyền đối với địa điểm, vật thể, tài liệu và dataset công khai.
4. Nếu có người tham gia hoặc ảnh/giọng nói nhận dạng được: có phê duyệt/miễn trừ phù hợp và consent form dễ tiếp cận.
5. Kiểm tra vùng lưu trữ local, backup được phép và quyền truy cập tối thiểu.

Dữ liệu người tham gia vẫn **bị khóa** cho đến khi đủ các cổng trên. Protocol tham chiếu [Luật Bảo vệ dữ liệu cá nhân 91/2025/QH15](https://vanban.chinhphu.vn/?classid=1&docid=214590&pageid=27160&typegroup=), nhưng không thay thế rà soát pháp lý/đạo đức của trường.

## 2. Môi trường an toàn

- Pilot trong nhà, khu vực kiểm soát, không có giao thông hoặc cầu thang nguy hiểm.
- Người chụp đứng yên hoặc có người giám sát; không yêu cầu bịt mắt.
- Không dùng SecondEye làm công cụ duy nhất để tránh vật cản trong phiên thu.
- Có lối dừng/thoát phiên rõ ràng; dừng ngay khi camera, dây cáp hoặc bố trí tạo nguy cơ.
- Tránh ghi hình người ngoài cuộc, màn hình cá nhân, biển số và tài liệu có định danh.

## 3. Cấu trúc lưu trữ local

```text
data/local/
  quarantine/       file chưa xác minh quyền/consent hoặc chờ redaction
  raw/              asset được phép, bất biến
  derived/          frame chọn, redaction, resize; có nguồn gốc tới raw
  annotations/      manifest và nhãn đang làm
  releases/v0.1/    manifest/nhãn đã khóa, không chứa consent record
```

`data/local/` không commit. Consent record và bảng ánh xạ subject key được lưu riêng, quyền truy cập hạn chế, không đặt cạnh ảnh phát hành.

## 4. ID và đơn vị nhóm

- Tạo `capture_session_id` cho một phiên thu liên tục.
- Tạo `scene_id` khi thay địa điểm/bố cục/nhiệm vụ thực tế.
- Tạo `video_id` cho mỗi clip.
- `group_id` bao trùm mọi frame/burst/near-duplicate của cùng cảnh và cùng asset dùng qua nhiều tác vụ.
- Không mã hóa tên người/địa điểm nhà riêng vào ID.

Ví dụ: `ses_20260804_01`, `scn_lab_table_01`, `vid_0001`, `grp_lab_table_01`.

## 5. Pilot

1. Thu 10–20 ảnh development không có người nhận dạng được.
2. Hoàn thành manifest/hash và gán nhãn theo guide.
3. Chạy validator; kiểm tra thủ công near-duplicate.
4. Hai người hoặc annotator + reviewer thử guide trên cùng subset.
5. Sửa ambiguity và tăng version trước khi thu tập chính.

Pilot không được tự động đưa vào test; muốn giữ phải qua review giống tập chính.

## 6. Kế hoạch obstacle: 200–500 mẫu

### Capture strata dự kiến

- Thiết bị camera thực dùng cho demo.
- Sáng thường, thiếu sáng trong nhà, ngược sáng.
- Vật cản trung tâm/bên; nhiều kích thước và che khuất.
- Lớp COCO phổ biến và `other_obstacle` như thùng, dây, mép vật thấp.
- Có mẫu không hazard và mẫu khó/không xác định để đo false alert/abstention.

### Video/burst

- Lưu video gốc trong quarantine/raw nếu được phép; chọn frame bằng quy tắc trước khi xem prediction.
- Frame liên tiếp hoặc cùng bố cục luôn chung `group_id`.
- Không tăng số mẫu bằng hàng chục frame gần như giống nhau; ghi frame index và khoảng lấy mẫu.
- Nếu đổi vị trí vật thể nhỏ nhưng cảnh/bối cảnh vẫn giống, giữ cùng group để tránh leakage.

### Khoảng cách

- Nếu cần ground truth, dùng phép đo/reference độc lập trong môi trường an toàn.
- Ghi phương pháp đo và sai số; không dùng monocular depth output làm ground truth cho chính ablation depth.
- Trước khi khóa band gần/trung bình/xa, chạy calibration/pilot và GVHD duyệt ranh giới.

## 7. Kế hoạch OCR: 150–300 ảnh

- Dùng nội dung dự án sở hữu/được phép: biển tự tạo, nhãn hàng hợp lệ, menu/tờ rơi công khai, tài liệu giả lập.
- Bao phủ chữ có dấu, nhiều cỡ/font, nền, góc xiên, chói và mờ có kiểm soát.
- Tránh hồ sơ thật, hóa đơn có dữ liệu cá nhân, thẻ, địa chỉ nhà, số điện thoại/email cá nhân.
- Chụp cả `clear`, `degraded`, `partial`, `unreadable` để đánh giá từ chối.
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

## 10. Kiểm soát chất lượng phiên thu

Sau mỗi phiên:

- Đếm asset thu/loại/quarantine theo lý do.
- Kiểm tra file hỏng, orientation, hash trùng và metadata thiếu.
- Xem contact sheet để phát hiện burst/near-duplicate và sửa group, chưa chia split.
- Kiểm tra coverage strata; không thu thêm chỉ để cải thiện kết quả model.
- Ghi log protocol version, thiết bị, người vận hành mã hóa và sự cố.

## 11. Đồng thuận, rút lui và sự cố

- Người tham gia được giải thích mục đích, dữ liệu thu, nơi lưu, ai truy cập, thời hạn, rủi ro, công bố và quyền dừng/rút.
- Consent phải có định dạng tiếp cận được; không coi im lặng là đồng thuận.
- Khi rút: đổi trạng thái `withdrawn`, quarantine ngay, ánh xạ và xóa raw/derived/annotation/backups theo kế hoạch đã duyệt, rồi tăng dataset version.
- Nếu lộ dữ liệu/sai quyền: dừng thu, cô lập bản sao, ghi sự cố và báo người có trách nhiệm; không tự ý tiếp tục dùng.

## 12. Checklist kết thúc collection

- [ ] Đạt target hoặc có lý do dừng được ghi rõ.
- [ ] Không còn mẫu `pending` trong development/test.
- [ ] Mọi asset có source/license/consent/hash/group.
- [ ] Tất cả test annotation đã review.
- [ ] Near-duplicate đã gom group trước split.
- [ ] Chạy validator không có error.
- [ ] Cập nhật data card bằng số thật và lý do loại mẫu.

