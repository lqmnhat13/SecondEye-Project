# SecondEye

SecondEye là MVP tích hợp AI đa phương thức chạy local trên macOS. Phạm vi hiện
tại chủ đích **không fine-tune**: hệ thống ghép các model pretrained cho
detection, metric depth, OCR, VQA, STT và TTS. Việc hoàn thành MVP và các thí
nghiệm hiện tại không phụ thuộc vào một checkpoint fine-tuned. Fine-tuning chỉ
là hướng phát triển tương lai tùy chọn nếu sau này dự án có dữ liệu đã được rà
soát đầy đủ và một câu hỏi nghiên cứu phù hợp.

> **Cảnh báo an toàn:** Đây là prototype nghiên cứu, không phải thiết bị điều
> hướng đã kiểm định và không thay thế gậy trắng, chó dẫn đường hoặc thiết bị hỗ
> trợ chuyên dụng.

## Runtime hiện tại

```text
Camera Mac/iPhone
    -> YOLO26m COCO (chỉ gắn nhãn)
    -> Depth Anything V2 Metric Indoor Small hoặc depth sensor đã căn chỉnh
    -> mặt sàn RANSAC + hành lang 3D + vật cản không phụ thuộc nhãn
    -> track ID + vận tốc tiếp cận/TTC + confirmation/cooldown/rearm
    -> macOS TTS

Ảnh theo yêu cầu -> Apple Vision OCR (PaddleOCR fallback), visual query/VQA
                   hoặc Grounding DINO tùy chọn -> TTS
Audio đã ghi     -> Whisper STT
```

Detection dùng schema `indoor_coco_baseline_v1` gồm 15 lớp COCO có ánh xạ trực
tiếp:

```text
person, chair, table, sofa, bed,
backpack, handbag, suitcase, bottle, potted_plant,
tv, laptop, toilet, sink, refrigerator
```

15 lớp chỉ là lớp ngữ nghĩa của YOLO, không còn là biên bao phủ an toàn. Một cụm
3D nhô khỏi sàn trong hành lang vẫn được giữ dưới nhãn `unknown_obstacle` khi
YOLO không nhận ra. Có thể bật Grounding DINO cho mô tả cửa, cầu thang, cột, tủ,
hộp hoặc thùng rác; kết quả open-vocabulary không tự trở thành cảnh báo an toàn.

## Cài đặt

Yêu cầu: Python 3.11 trên macOS Apple Silicon. Nên đặt runtime ngoài thư mục
Documents/iCloud để model và dependency không bị macOS offload:

```bash
cd /Users/lenhat/Documents/SecondEye-Project
brew install ffmpeg
./setup_mvp.sh
source ~/Library/Caches/SecondEye/venv/bin/activate
```

Runtime mặc định nằm tại `~/Library/Caches/SecondEye/venv`. Có thể đổi bằng biến
`SECONDEYE_RUNTIME_DIR`. `setup_mvp.sh` cài detection, depth/VQA/STT, OCR và test,
biên dịch helper Apple Vision OCR, sau đó chạy `doctor` bằng import thật.

Model pretrained được tải vào cache ở lần chạy đầu. Sau khi đã tải đủ model,
các module local có thể chạy không cần gửi ảnh lên API.

## Kiểm tra môi trường

```bash
secondeye doctor
pytest
```

Mặc định `doctor` chỉ kiểm tra dependency, không tải model.

Kiểm tra OCR end-to-end bằng đúng runtime đang dùng:

```bash
secondeye doctor --ocr-smoke-image data/samples/ultralytics_bus.jpg
```

Lệnh này chỉ báo engine, số dòng và latency; không in transcript của ảnh.

## Chạy MVP tích hợp

Lệnh dưới đây mở một runtime duy nhất gồm cảnh báo vật cản, OCR, mô tả cảnh,
VQA, push-to-talk, TTS ưu tiên và session log:

```bash
./run_mvp.sh --camera 0
```

Với iPhone Continuity Camera:

```bash
./run_mvp.sh --camera 1 --microphone auto
```

Camera phải hướng ra môi trường di chuyển. Trên MacBook, `--camera 0` thường là
FaceTime HD Camera hướng vào người dùng nên detector sẽ nhận chính người dùng là
`person`. Khi dùng Continuity Camera, đặt iPhone hướng ra phía trước và chọn đúng
camera index.

`--microphone auto` là mặc định và tự ưu tiên microphone tích hợp theo tên thiết
bị, nên không phụ thuộc index AVFoundation. Khi cần, vẫn có thể truyền index hoặc
tên chính xác, ví dụ `--microphone 2` hoặc
`--microphone "MacBook Pro Microphone"`.

Điều khiển trong cửa sổ camera:

- `o`: đọc chữ từ burst frame gần nhất và chỉ phát transcript đủ đồng thuận.
- `s`: mô tả ngắn dựa trên các vật thể detector thực sự nhìn thấy.
- `v`: hỏi câu cấu hình bởi `--question`.
- `m`: thu lệnh giọng nói trong 4 giây; hỗ trợ đọc chữ, mô tả, hỏi ảnh, dừng và lặp lại.
- `r`: lặp lại phản hồi gần nhất; `x`: dừng âm thanh; `q`/`Esc`: thoát.

OCR, VQA và Whisper được lazy-load ở lần dùng đầu để camera cảnh báo khởi động
trước. Tác vụ ngữ nghĩa chạy ở worker riêng nên không chặn capture/UI hoặc worker
an toàn. Log JSONL được lưu mặc định trong `logs/` và có thể đổi bằng `--log`.

## Chạy camera Mac hoặc iPhone

Camera Mac thường là `0`:

```bash
secondeye camera --camera 0 --no-tts
```

Sau khi bật Continuity Camera, iPhone thường là `1`; chỉ số thực tế phụ thuộc
thiết bị đang kết nối:

```bash
secondeye camera --camera 1 --depth
```

Nhấn `q` hoặc `Esc` để thoát, `x` để dừng âm thanh. Khi không bật `--depth`,
detection chỉ hiển thị ứng viên và không phát cảnh báo “ở gần”.

Camera runtime dùng capture thread và latest-frame buffer riêng: UI không chờ
YOLO/depth, frame cũ tự bị bỏ và kết quả quá hạn không được vẽ/cảnh báo. Depth
chỉ được xác nhận cảnh báo khi được tính từ đúng frame đang detection; kết quả
depth cũ không được chiếu lên bbox mới. Mặc định camera/display là 30 FPS,
detection tối đa 12 Hz và depth tối đa 3 Hz:

```bash
secondeye camera --camera 1 --depth \
  --width 1280 --height 720 \
  --camera-fps 30 --display-fps 30 \
  --detection-fps 12 --depth-fps 3 \
  --max-depth-age 0.5 --max-result-age 0.75 \
  --confirmation-frames 2 --overlay-max-age 0.75 \
  --voice Linh --speech-rate 165
```

Kiểm tra riêng giọng Việt trước khi mở camera:

```bash
secondeye speech-test \
  --voice Linh --speech-rate 165 \
  --text "Cẩn thận, ghế phía trước."
```

## Chạy các module trên một ảnh

```bash
# Detection pretrained
secondeye image --source /duong/dan/anh.jpg --no-tts

# Detection + metric depth + geometry
secondeye image --source /duong/dan/anh.jpg --depth --no-tts

# OCR tiếng Việt
secondeye image --source /duong/dan/anh.jpg --ocr

# VQA local; câu trả lời không được dùng làm chỉ dẫn điều hướng
secondeye image \
  --source /duong/dan/anh.jpg \
  --question "What objects are in front of me?" \
  --open-vocabulary

# Lưu unified JSON log
secondeye image \
  --source /duong/dan/anh.jpg \
  --depth --ocr --no-tts \
  --output results/session.json
```

STT một file âm thanh đã ghi:

```bash
secondeye transcribe --audio /duong/dan/lenh.wav
```

CLI detection cũ vẫn tồn tại cho smoke test và nghiên cứu:

```bash
python scripts/fetch_smoke_asset.py
secondeye-detection demo --source data/samples/ultralytics_bus.jpg
secondeye-detection camera-demo --camera 0
```

## Quy tắc risk hiện tại

- Bbox và depth tương đối **không bao giờ** là bằng chứng phát cảnh báo.
- Depth metric phải cùng frame RGB; kết quả quá `--max-result-age` bị loại.
- Pipeline dựng point cloud, fit mặt sàn, giới hạn hành lang di chuyển và tìm
  mọi cụm nhô khỏi sàn, kể cả khi không có nhãn YOLO.
- Khoảng cách dùng mét: mặc định emergency ≤ 0,8 m, near ≤ 1,8 m và medium ≤ 3 m.
- Tracker tạo `track_id`, làm mượt vận tốc tiếp cận và tính TTC. TTC ≤ 1,5 s
  được nâng thành emergency.
- Near cần hai quan sát metric liên tiếp; emergency có thể phát ngay. Cooldown
  chỉ chặn lặp âm thanh, không làm trạng thái vật cản trở về `IDLE`.
- Nhiều vật cản cùng lúc được gộp thành một câu, không bỏ mất cảnh báo thứ hai.
- TTS mặc định dùng giọng `Linh` (`vi_VN`) ở tốc độ 165 từ/phút.
- Câu trả lời VQA ngắn được dịch bằng từ vựng và mẫu câu thị giác có kiểm soát.
  Câu ngoài miền hỗ trợ bị từ chối thay vì phát một bản dịch chưa kiểm chứng;
  Marian chỉ còn là fallback thử nghiệm và mặc định bị tắt.
- Overlay camera dùng font Unicode qua Pillow nên hiển thị đầy đủ dấu tiếng Việt.
- Trong camera MVP, YOLO/depth dùng Apple MPS; BLIP và Whisper được cố định trên
  CPU để tránh Metal command-buffer crash giữa các worker thread.

## Cấu trúc source public

```text
configs/pretrained_indoor.toml   schema và runtime config
src/secondeye/detection/         YOLO26 COCO adapter và risk candidate
src/secondeye/detection/geometry.py  mặt sàn và vật cản class-agnostic
src/secondeye/multimodal/        metric depth/provider, OCR, VQA, STT và TTS
src/secondeye/system/            tracking, state machine, freshness và unified CLI
src/secondeye/evaluation/        metric safety theo sự kiện
tests/                           unit tests không cần tải model
scripts/fetch_smoke_asset.py     tải ảnh smoke test có thể tái tạo
```

Tài liệu local được phân loại và dẫn đường tại
[`docs/README.md`](docs/README.md). Nhóm `docs/current/` và `docs/guides/` mô tả
phạm vi đang dùng; tài liệu dữ liệu/taxonomy an toàn lịch sử nằm dưới
`docs/research/data/` và không phải dependency của MVP pretrained.

Hướng dẫn vận hành chi tiết, đầy đủ tham số, phím điều khiển, output và xử lý lỗi:
[`docs/guides/complete-usage-guide.md`](docs/guides/complete-usage-guide.md).

GitHub có thể chứa source code, test, runtime config và tài liệu Markdown công
khai. Dataset, ảnh/model weights, log, artifact, tài liệu trong `docs/private/`,
DOCX/PDF/XLSX và kết quả chạy được giữ local và bị `.gitignore` loại khỏi commit.

## Giới hạn đã biết

- Nhãn YOLO/Grounding DINO vẫn có thể sai; an toàn dựa vào geometry metric nhưng
  geometry cũng có thể thất bại khi sàn không thấy rõ hoặc depth nhiễu.
- Chín threshold mới ngoài sáu lớp benchmark cũ đang là giá trị provisional 0.35.
- Metric monocular depth có sai số scale; depth sensor/LiDAR đã căn chỉnh được ưu
  tiên. FOV ước lượng phải thay bằng intrinsics thật khi tích hợp camera riêng.
- Chưa có ground-truth safety test set đủ độc lập, nên chưa được dùng như thiết
  bị điều hướng. Chạy `secondeye-evaluate-safety` trước mọi pilot có người dùng.
- Apple Vision/PaddleOCR, BLIP và Whisper vẫn cần benchmark trên dữ liệu thực tế.
- Camera demo cần kiểm thử trực tiếp vì quyền camera và chỉ số thiết bị phụ thuộc macOS.
- Fine-tuning không thuộc phạm vi phiên bản hiện tại. Nếu được thực hiện trong
  tương lai, đó sẽ là một nhánh nghiên cứu riêng với dataset/protocol mới được
  duyệt; không mặc định là bước kế tiếp của MVP.

Ultralytics được sử dụng theo điều kiện giấy phép tương ứng; cần rà giấy phép của
từng model và dependency trước khi phát hành hoặc thương mại hóa.
