# Rà soát giấy phép nguồn dữ liệu public cho indoor v1.1

Cập nhật: 2026-08-09
Phạm vi: 10 lớp còn thiếu của schema SecondEye indoor v1.1. Đây là hồ sơ kỹ
thuật phục vụ khóa luận, không phải tư vấn pháp lý.

## Quyết định nguồn

| Nguồn | Thành phần dùng | Lớp SecondEye | Điều kiện quyền sử dụng | Quyết định |
|---|---|---|---|---|
| Open Images V7 validation/test | Pixel có trường license đúng bằng CC BY 2.0; verified bbox/human labels | `person`, ba lớp cửa, hai lớp cầu thang, `backpack_bag`, `box`, `trash_bin` | Mỗi ảnh giữ author, landing URL và license; annotation của Google là CC BY 4.0; relabel phải được ghi là thay đổi | Chấp nhận có điều kiện, chỉ sau visual/privacy review |
| ADE20K Scene Parsing Benchmark | Ảnh validation và mask semantic lớp 43 `column, pillar`; các lớp stair chỉ làm ứng viên review hướng | `column`, bổ sung `stairs_down` | Ảnh chỉ dùng cho nghiên cứu/giáo dục phi thương mại; annotation và software theo BSD-3-Clause | Chấp nhận cho khóa luận; không dùng mặc định cho sản phẩm thương mại |
| Objects365 | Không nhập | Không | Annotation CC BY 4.0 nhưng ảnh thuộc chủ sở hữu Flickr; trang chính thức giới hạn mục đích học thuật và yêu cầu tuân thủ điều khoản Flickr | Loại khỏi v1.1 để tránh chuỗi quyền ảnh không rõ bằng hai nguồn trên |

## Nguồn chính thức đã kiểm tra

### Open Images V7

- Trang tải và tuyên bố giấy phép: <https://storage.googleapis.com/openimages/web/download_v7.html>
- Thống kê/taxonomy: <https://storage.googleapis.com/openimages/web/factsfigures_v7.html>
- Pixel validation/test chính thức được tải từ bucket do trang download của Open
  Images công bố.
- Metadata license từng ảnh lấy từ các file image metadata validation/test chính
  thức tương ứng.
- Bản nhập này chỉ chấp nhận ảnh có license chính xác
  `https://creativecommons.org/licenses/by/2.0/`; ảnh thiếu hoặc khác license bị loại.
- Khi phân phối kết quả dẫn xuất phải giữ attribution, liên kết license và ghi rõ
  việc ánh xạ/relabel nhãn. Pixel và manifest attribution đầy đủ chỉ lưu local.

Ánh xạ nguồn:

| Open Images class | MID | SecondEye |
|---|---|---|
| Person | `/m/01g317` | `person`, sau privacy review |
| Door | `/m/02dgv` | chỉ làm ứng viên; reviewer chọn `doorway_open`, `door_closed`, `glass_door` |
| Stairs | `/m/01lynh` | chỉ làm ứng viên; reviewer chọn `stairs_up` hoặc `stairs_down` |
| Backpack | `/m/01940j` | `backpack_bag` |
| Box | `/m/025dyy` | `box` |
| Waste container | `/m/0bjyj5` | `trash_bin`, sau kiểm tra ngữ nghĩa |

Không tự suy diễn trạng thái cửa hay hướng cầu thang từ nhãn generic. Bbox bị mơ
hồ, cảnh ngoài trời, ảnh sản phẩm tách nền, ảnh hỏng/mờ và ảnh có thông tin riêng
tư bị loại.

### ADE20K

- Trang chính thức: <https://ade20k.csail.mit.edu/>
- Điều khoản: <https://ade20k.csail.mit.edu/terms>
- Archive chính thức: <https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip>
- Repository chính thức: <https://github.com/CSAILVision/ADE20K>
- Dùng class semantic 43 (`column, pillar`) của validation. Mỗi connected
  component đủ lớn được đổi thành bbox YOLO rồi kiểm tra trực quan.
- Các class stair semantic 54, 60, 97 và 122 chỉ tạo ứng viên; reviewer chỉ nhận
  component nhìn rõ hướng đi xuống thành `stairs_down`, không ánh xạ tự động.
- Điều kiện “non-commercial research and educational use only” phải đi theo
  model/dataset dẫn xuất. Nếu dự án chuyển sang sản phẩm thương mại, phải loại
  ADE20K, thay nguồn hoặc xin quyền riêng trước khi tái huấn luyện.

## Kiểm soát lưu trữ và công bố

- Pixel, mask, label YOLO, inventory và attribution từng ảnh nằm dưới
  `data/local/`, được `.gitignore` loại trừ.
- Git chỉ lưu script tái tạo, schema, hướng dẫn và thống kê tổng hợp; không lưu
  toàn bộ dataset hay ảnh riêng tư.
- Bản dataset chỉ được nhập sau khi checksum, license, bbox, privacy và split đều
  qua validator. Không dùng ảnh tải từ kết quả tìm kiếm web hoặc nguồn mirror
  không do tác giả công bố.

## Giới hạn

Giấy phép nội dung không tự động giải quyết quyền riêng tư, tính phù hợp với mục
đích an toàn hoặc sai lệch phân bố. Vì vậy ảnh `person` vẫn cần review riêng;
metric trên public dataset không chứng minh khả năng hoạt động với camera của
người dùng trong nhà/trường học thực tế.

Metadata CC BY 2.0 của Open Images là điều kiện lọc kỹ thuật, không phải xác minh
pháp lý độc lập rằng người tải ban đầu sở hữu mọi quyền. Trang chính thức cũng
không bảo đảm trạng thái giấy phép của từng ảnh; cần giữ attribution và rà soát
lại nếu phân phối dataset/model ra ngoài phạm vi khóa luận. Chưa có reviewer độc
lập thứ hai cho v1.1, vì vậy không mô tả tập này là double-annotated.
