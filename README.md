# SecondEye

SecondEye là MVP tích hợp AI đa phương thức chạy local trên macOS. Phạm vi hiện
tại chủ đích **không fine-tune**: hệ thống ghép các model pretrained cho
detection, relative depth, OCR, VQA, STT và TTS. Việc hoàn thành MVP và các thí
nghiệm hiện tại không phụ thuộc vào một checkpoint fine-tuned. Fine-tuning chỉ
là hướng phát triển tương lai tùy chọn nếu sau này dự án có dữ liệu đã được rà
soát đầy đủ và một câu hỏi nghiên cứu phù hợp.

> **Cảnh báo an toàn:** Đây là prototype nghiên cứu, không phải thiết bị điều
> hướng đã kiểm định và không thay thế gậy trắng, chó dẫn đường hoặc thiết bị hỗ
> trợ chuyên dụng.

## Runtime hiện tại

```text
Camera Mac/iPhone
    -> YOLO26m COCO detection
    -> Depth Anything V2 Small (tùy chọn)
    -> risk fusion + cooldown
    -> macOS TTS

Ảnh theo yêu cầu -> Apple Vision OCR (PaddleOCR fallback) hoặc visual query/VQA -> TTS
Audio đã ghi     -> Whisper STT
```

Detection dùng schema `indoor_coco_baseline_v1` gồm 15 lớp COCO có ánh xạ trực
tiếp:

```text
person, chair, table, sofa, bed,
backpack, handbag, suitcase, bottle, potted_plant,
tv, laptop, toilet, sink, refrigerator
```

Schema này cố ý không tuyên bố hỗ trợ cửa, cầu thang, cột, tủ, hộp hoặc thùng
rác. Taxonomy an toàn đặc thù và dữ liệu lịch sử được bảo tồn như tài sản nghiên
cứu cho một nhánh mở rộng tương lai; chúng không phải đầu vào bắt buộc của MVP.

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
  --max-depth-age 0.5 --overlay-max-age 0.75 \
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

# Detection + relative depth
secondeye image --source /duong/dan/anh.jpg --depth --no-tts

# OCR tiếng Việt
secondeye image --source /duong/dan/anh.jpg --ocr

# VQA local; câu trả lời không được dùng làm chỉ dẫn điều hướng
secondeye image \
  --source /duong/dan/anh.jpg \
  --question "What objects are in front of me?"

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

- Chỉ các lớp trong `risk.candidate_classes` mới là ứng viên vật cản.
- Bbox phải nằm trong vùng di chuyển trung tâm.
- Proximity fusion phải xác nhận band `near` trước khi phát cảnh báo gần.
- Fusion kết hợp relative depth với tỷ lệ diện tích/chiều cao bbox trong ảnh;
  bbox rất lớn được coi là gần, bbox rất nhỏ được coi là xa.
- `near/medium/far` vẫn là band heuristic, không phải khoảng cách mét.
- Cooldown ngăn cùng một cảnh báo bị đọc liên tục.
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
src/secondeye/multimodal/        depth, OCR, VQA, STT và TTS adapters
src/secondeye/system/            state machine, orchestrator và unified CLI
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

- 15 lớp mới là schema integration dễ hơn, không phải taxonomy an toàn hoàn chỉnh.
- Chín threshold mới ngoài sáu lớp benchmark cũ đang là giá trị provisional 0.35.
- Relative monocular depth không cung cấp khoảng cách tuyệt đối.
- Apple Vision/PaddleOCR, BLIP và Whisper vẫn cần benchmark trên dữ liệu thực tế.
- Camera demo cần kiểm thử trực tiếp vì quyền camera và chỉ số thiết bị phụ thuộc macOS.
- Fine-tuning không thuộc phạm vi phiên bản hiện tại. Nếu được thực hiện trong
  tương lai, đó sẽ là một nhánh nghiên cứu riêng với dataset/protocol mới được
  duyệt; không mặc định là bước kế tiếp của MVP.

Ultralytics được sử dụng theo điều kiện giấy phép tương ứng; cần rà giấy phép của
từng model và dependency trước khi phát hành hoặc thương mại hóa.
