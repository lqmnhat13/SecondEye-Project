# Hướng dẫn sử dụng đầy đủ SecondEye MVP

Cập nhật: 2026-08-30. Phiên bản mã nguồn: `0.3.0`.

Tài liệu này dành cho người lần đầu sử dụng hoặc bảo trì SecondEye. Nội dung mô
tả đúng pipeline pretrained hiện tại, từ cài đặt đến vận hành, đọc kết quả, xử
lý lỗi và hiểu cấu trúc source. Các lệnh train/custom checkpoint được đặt riêng
ở cuối tài liệu vì chúng là hạ tầng nghiên cứu tương lai, không thuộc quy trình
MVP hiện hành.

> **Cảnh báo an toàn:** SecondEye là prototype nghiên cứu. Không dùng hệ thống
> thay cho gậy trắng, chó dẫn đường, đánh giá của con người hoặc thiết bị hỗ trợ
> đã được kiểm định. Không dựa vào kết quả để băng qua đường, đi cầu thang hoặc
> xử lý tình huống có nguy cơ gây thương tích.

## Mục lục

1. [SecondEye làm được gì](#1-secondeye-làm-được-gì)
2. [Kiến trúc và luồng xử lý](#2-kiến-trúc-và-luồng-xử-lý)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Yêu cầu hệ thống](#4-yêu-cầu-hệ-thống)
5. [Cài đặt lần đầu](#5-cài-đặt-lần-đầu)
6. [Kiểm tra trước khi chạy](#6-kiểm-tra-trước-khi-chạy)
7. [Chạy demo đầy đủ](#7-chạy-demo-đầy-đủ)
8. [Chạy camera detection riêng](#8-chạy-camera-detection-riêng)
9. [Xử lý một ảnh](#9-xử-lý-một-ảnh)
10. [Nhận dạng giọng nói và thử TTS](#10-nhận-dạng-giọng-nói-và-thử-tts)
11. [Cấu hình detection và risk](#11-cấu-hình-detection-và-risk)
12. [Định dạng kết quả và session log](#12-định-dạng-kết-quả-và-session-log)
13. [Lệnh detection/dataset cho nghiên cứu tương lai](#13-lệnh-detectiondataset-cho-nghiên-cứu-tương-lai)
14. [Quy trình kiểm thử và phát triển](#14-quy-trình-kiểm-thử-và-phát-triển)
15. [Xử lý lỗi thường gặp](#15-xử-lý-lỗi-thường-gặp)
16. [Giới hạn cần hiểu đúng](#16-giới-hạn-cần-hiểu-đúng)

## 1. SecondEye làm được gì

Pipeline hiện tại chạy local trên macOS và ghép các model pretrained. Không có
bước fine-tuning bắt buộc.

| Chức năng | Thành phần | Cách sử dụng |
|---|---|---|
| Nhận diện vật thể | YOLO26m pretrained COCO | Tự chạy trong `image`, `camera` và `demo` |
| Ước lượng độ sâu tương đối | Depth Anything V2 Small | Bật bằng `--depth`; `demo` bật sẵn |
| Cảnh báo vật cản gần | Detection + depth + risk fusion | Chỉ phát khi đủ điều kiện xác nhận |
| Đọc chữ trong ảnh | Apple Vision `vi-VN`; PaddleOCR `vi` fallback | `image --ocr` hoặc phím `o` trong demo |
| Mô tả cảnh | Tổng hợp từ detection | Phím `s` trong demo |
| Hỏi đáp ảnh | Detection-grounded query + BLIP VQA | `image --question` hoặc phím `v` |
| Dịch câu trả lời VQA | Bộ dịch miền thị giác có kiểm soát | Tự chạy khi câu trả lời tiếng Anh chưa có ánh xạ nhanh |
| Nhận dạng giọng nói | Whisper Small | `transcribe` hoặc phím `m` trong demo |
| Phát giọng nói | macOS `say`, giọng Linh | Tự chạy trừ khi dùng `--no-tts` |
| Ghi phiên chạy | JSONL thread-safe | Tự chạy trong `demo` |

Detection chỉ công bố 15 lớp có ánh xạ trực tiếp:

```text
person, chair, table, sofa, bed,
backpack, handbag, suitcase, bottle, potted_plant,
tv, laptop, toilet, sink, refrigerator
```

Hệ thống không tuyên bố nhận diện cửa, cầu thang, cột, tủ, hộp hoặc thùng rác.
Một vật nằm ngoài 15 lớp trên có thể được YOLO gốc nhận ra nhưng adapter sẽ loại
bỏ khỏi kết quả SecondEye.

## 2. Kiến trúc và luồng xử lý

### 2.1 Luồng vision liên tục

```text
Camera Mac/iPhone
    -> capture thread
    -> latest-frame buffer (chỉ giữ frame mới nhất)
    -> vision worker
       -> YOLO26m detection
       -> Depth Anything (nếu bật và đến chu kỳ depth)
       -> gắn near/medium/far vào bbox
       -> risk fusion + xác nhận nhiều frame + cooldown
    -> overlay cửa sổ OpenCV
    -> audio priority queue
    -> macOS TTS
```

Capture, inference và UI không chạy trong cùng một vòng chặn. Nếu model xử lý
chậm hơn camera, frame cũ bị thay bằng frame mới thay vì tạo hàng đợi dài. Một
kết quả quá cũ hơn `--overlay-max-age` không được vẽ. Chỉ depth sinh từ đúng
`frame_id` đang detection và không quá `--max-depth-age` mới được xác nhận vật
cản gần. Depth của frame trước không được tái sử dụng trên bbox của frame mới.

### 2.2 Luồng tác vụ theo yêu cầu

```text
Frame camera thô (không chứa bbox/chữ overlay)
    + phím o -> kiểm tra chất lượng ảnh -> Apple Vision/PaddleOCR -> TTS
    + phím s -> detection hiện tại -> mô tả cảnh tiếng Việt -> TTS
    + phím v -> câu hỏi object phía trước -> detection grounded -> TTS
    + phím m -> microphone -> Whisper STT -> phân loại ý định
               -> OCR / mô tả / câu hỏi grounded / BLIP / dừng / lặp lại
```

Frame dùng cho OCR/VQA được sao chép trước khi vẽ bbox, trạng thái và danh sách
phím. Vì vậy OCR không đọc lại chữ do chính giao diện SecondEye tạo ra.

Câu hỏi số lượng và đồ vật phía trước được trả lời từ detection có confidence từ
`0.45`, không để BLIP đoán tên vật. Câu tiếng Việt về màu sắc, hành động và trang
phục được chuyển bằng mẫu xác định sang tiếng Anh trước khi gọi BLIP. Câu tiếng
Việt ngoài các dạng hỗ trợ sẽ `abstain` kèm hướng dẫn, không gửi nguyên văn cho
model tiếng Anh.

OCR, VQA và Whisper được lazy-load ở lần sử dụng đầu. Vì vậy lần nhấn phím đầu
có thể chậm hơn các lần sau. Semantic worker chỉ nhận một tác vụ chờ; nếu đang
bận, yêu cầu mới được từ chối thay vì làm nghẽn vision worker.

### 2.3 Điều kiện phát cảnh báo vật cản

Một detection chỉ trở thành cảnh báo khi đồng thời thỏa mãn:

1. Confidence đạt threshold riêng của lớp.
2. Lớp nằm trong `risk.candidate_classes`.
3. Tâm bbox nằm trong 40% vùng giữa ảnh.
4. Depth còn đủ mới và gắn band `near`.
5. Cùng khóa `label + direction` xuất hiện trong 2 frame xác nhận.
6. Cảnh báo đó không nằm trong cooldown 4 giây.

Nếu depth tắt hoặc quá cũ, detection vẫn hiển thị nhưng không được phát thành
câu “ở gần”. `near/medium/far` là độ sâu tương đối trong từng frame, không phải
khoảng cách theo mét.

### 2.4 Thứ tự ưu tiên âm thanh

```text
STOP > OBSTACLE > ERROR > SEMANTIC > INFO
```

Cảnh báo vật cản có thể ngắt câu OCR/VQA đang đọc. `r` lặp lại câu gần nhất và
`x` xóa hàng đợi, dừng câu đang phát.

### 2.5 Dịch câu trả lời VQA

Câu trả lời phổ biến như màu sắc, số lượng, vật thể, vị trí và hành động được
dịch bằng từ vựng/mẫu câu thị giác có kiểm soát. Chỉ kết quả có
`quality_assured: true` mới được phép đi tới TTS. Câu nằm ngoài miền hỗ trợ bị
từ chối với trạng thái `abstained`; hệ thống không đọc một bản dịch đoán mò.
Marian tồn tại như fallback thử nghiệm nhưng mặc định bị tắt.

## 3. Cấu trúc thư mục

```text
SecondEye-Project/
├── README.md                         giới thiệu và quick start
├── pyproject.toml                    package, dependency extras, CLI entrypoint
├── setup_mvp.sh                      tạo/cập nhật runtime đầy đủ
├── run_mvp.sh                        launcher cho lệnh `secondeye demo`
├── yolo26m.pt                        weights local; bị Git bỏ qua
├── configs/
│   └── pretrained_indoor.toml        schema, mapping, threshold và risk config
├── src/secondeye/
│   ├── accelerator.py                khóa dùng chung cho Apple MPS
│   ├── data/
│   │   └── protocol.py               audit manifest CSV nghiên cứu
│   ├── detection/
│   │   ├── config.py                 đọc và kiểm tra TOML
│   │   ├── model.py                  YOLO adapter và output schema
│   │   ├── risk.py                   vùng trái/giữa/phải và obstacle candidate
│   │   ├── pipeline.py               CLI detection/custom checkpoint cũ
│   │   └── dataset.py                prepare/validate dataset
│   ├── multimodal/
│   │   ├── depth.py                  relative depth và depth band
│   │   ├── ocr.py                    Apple Vision + PaddleOCR fallback
│   │   ├── apple_vision_ocr.m         helper OCR native macOS
│   │   ├── questions.py              định tuyến câu hỏi Việt/Anh
│   │   ├── quality.py                quality gate cho OCR/VQA
│   │   ├── speech.py                 microphone, Whisper và macOS TTS
│   │   ├── translation.py            dịch VQA fail-safe
│   │   └── vqa.py                    BLIP VQA adapter
│   └── system/
│       ├── cli.py                    CLI `secondeye`
│       ├── camera.py                 capture và async vision runtime
│       ├── demo.py                   UI demo, phím và semantic worker
│       ├── pipeline.py               ghép detection/depth/OCR/VQA/TTS
│       ├── orchestrator.py           state, priority, confirmation, cooldown
│       ├── audio.py                  hàng đợi TTS ưu tiên
│       ├── overlay.py                vẽ bbox và chữ Unicode
│       └── session.py                ghi JSONL
├── tests/unit/                        unit/integration regression tests
├── docs/                              tài liệu public
├── data/                              dữ liệu local, bị Git bỏ qua
├── logs/                              session log, bị Git bỏ qua
└── results/                           JSON/output chạy máy, bị Git bỏ qua
```

## 4. Yêu cầu hệ thống

Môi trường đã được thiết kế và kiểm tra cho:

- macOS trên Apple Silicon;
- Python 3.11; package khai báo hỗ trợ `>=3.11,<3.13` nhưng script cài đặt mặc
  định gọi `python3.11`;
- Homebrew và FFmpeg;
- quyền Camera và Microphone cho Terminal/Python;
- giọng tiếng Việt `Linh` của macOS nếu muốn dùng TTS.

Kiểm tra công cụ:

```bash
python3.11 --version
ffmpeg -version
say -v '?' | grep '^Linh '
```

Nếu chưa có FFmpeg hoặc Python 3.11:

```bash
brew install python@3.11 ffmpeg
```

Nếu chưa có giọng Linh, mở:

```text
System Settings -> Accessibility -> Spoken Content -> System Voice
```

Tải một giọng tiếng Việt có tên `Linh`, sau đó chạy lại `speech-test` ở phần 10.

## 5. Cài đặt lần đầu

### 5.1 Cài runtime mặc định

```bash
cd /Users/lenhat/Documents/SecondEye-Project
./setup_mvp.sh
```

Script thực hiện lần lượt:

1. tạo virtual environment tại `~/Library/Caches/SecondEye/venv`;
2. cập nhật `pip`, `setuptools`, `wheel`;
3. cài package cùng extras `dev,detection,multimodal,ocr`;
4. chạy `secondeye doctor`.

Kích hoạt runtime để dùng các entrypoint trực tiếp:

```bash
source ~/Library/Caches/SecondEye/venv/bin/activate
```

Sau khi activate, ba lệnh sau khả dụng:

```text
secondeye                    CLI MVP chính
secondeye-detection          CLI detection/nghiên cứu tương thích ngược
secondeye-validate-manifest  audit manifest CSV
```

`run_mvp.sh` tự gọi executable trong runtime nên không bắt buộc activate.

### 5.2 Chọn vị trí runtime khác

Dùng cùng một `SECONDEYE_RUNTIME_DIR` khi cài và chạy:

```bash
SECONDEYE_RUNTIME_DIR=/duong/dan/runtime ./setup_mvp.sh
SECONDEYE_RUNTIME_DIR=/duong/dan/runtime ./run_mvp.sh --camera 0
```

Có thể chọn Python dùng để tạo venv:

```bash
SECONDEYE_PYTHON=/opt/homebrew/bin/python3.11 ./setup_mvp.sh
```

### 5.3 Model và cache

Lần chạy đầu có thể tải model và mất nhiều thời gian. Các vị trí thường dùng:

```text
yolo26m.pt                         weights YOLO trong project
~/.cache/huggingface/hub/         Depth Anything, BLIP, Whisper, Marian
~/.paddlex/official_models/       các model PaddleOCR
~/Library/Caches/SecondEye/venv/  Python runtime
~/Library/Caches/SecondEye/venv/bin/secondeye-vision-ocr  OCR chính trên macOS
```

Chỉ bật chế độ offline sau khi tất cả model cần dùng đã có trong cache:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
./run_mvp.sh --camera 0
```

Nếu model chưa cache, chế độ offline sẽ làm module tương ứng không tải được.

### 5.4 Cấp quyền macOS

Khi macOS hỏi quyền, cho phép ứng dụng terminal hoặc Python đang chạy truy cập:

```text
System Settings -> Privacy & Security -> Camera
System Settings -> Privacy & Security -> Microphone
```

Sau khi đổi quyền, đóng hoàn toàn tiến trình Python/cửa sổ terminal đang chạy và
mở lại trước khi thử lại.

## 6. Kiểm tra trước khi chạy

### 6.1 Doctor

```bash
source ~/Library/Caches/SecondEye/venv/bin/activate
secondeye doctor
```

Kết quả đạt phải có:

```json
{
  "success": true,
  "modules": {
    "detection": true,
    "depth_vqa_stt": true,
    "translation_en_vi": true,
    "ocr": true,
    "tts_macos": true,
    "tts_voice_linh_vi_vn": true
  }
}
```

`doctor` chỉ import dependency và kiểm tra giọng TTS; nó không tải hoặc chạy
inference model. Luôn đọc trường JSON `success`, không chỉ dựa vào việc command
đã in được output.

### 6.2 Unit/integration test

```bash
cd /Users/lenhat/Documents/SecondEye-Project
source ~/Library/Caches/SecondEye/venv/bin/activate
pytest -q
```

Chế độ nghiêm ngặt, biến warning thành lỗi:

```bash
pytest -q -W error -p no:cacheprovider
```

Test dùng `src/` theo cấu hình trong `pyproject.toml`; phần lớn không tải model.
Test pass chứng minh contract và orchestration, không thay thế benchmark chất
lượng model trên dữ liệu thực tế.

### 6.3 Kiểm tra CLI

```bash
secondeye --help
secondeye demo --help
secondeye image --help
```

Nếu dùng config riêng, `--config` là option cấp cao và phải đứng trước
subcommand:

```bash
secondeye --config /duong/dan/config.toml image --source /duong/dan/anh.jpg
```

## 7. Chạy demo đầy đủ

### 7.1 Lệnh khuyến nghị

Camera Mac tích hợp thường là index `0`:

```bash
cd /Users/lenhat/Documents/SecondEye-Project
./run_mvp.sh --camera 0
```

`run_mvp.sh` tương đương với:

```bash
~/Library/Caches/SecondEye/venv/bin/secondeye demo --camera 0
```

Demo mặc định bật depth, OCR lazy, VQA lazy, microphone auto, TTS ưu tiên và
session log.

### 7.2 Ví dụ cấu hình đầy đủ

```bash
./run_mvp.sh \
  --camera 0 \
  --depth \
  --width 1280 \
  --height 720 \
  --camera-fps 30 \
  --display-fps 30 \
  --detection-fps 12 \
  --depth-fps 3 \
  --max-depth-age 0.5 \
  --overlay-max-age 1.5 \
  --question "What objects are directly in front of me?" \
  --microphone auto \
  --listen-seconds 4 \
  --voice Linh \
  --speech-rate 165 \
  --log logs/my_demo.jsonl
```

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `--camera` | `0` | Index camera OpenCV |
| `--depth` / `--no-depth` | bật | Bật/tắt Depth Anything |
| `--width` | `1280` | Chiều rộng camera yêu cầu |
| `--height` | `720` | Chiều cao camera yêu cầu |
| `--camera-fps` | `30` | FPS yêu cầu từ camera |
| `--display-fps` | `30` | Tần suất cập nhật cửa sổ |
| `--detection-fps` | `12` | Giới hạn tần suất detection; không bảo đảm máy đạt đúng 12 Hz |
| `--depth-fps` | `3` | Tần suất mục tiêu của depth |
| `--max-depth-age` | `0.50` giây | Depth cùng frame nhưng cũ hơn giá trị này không được fusion |
| `--overlay-max-age` | `1.50` giây | Kết quả cũ hơn giá trị này không được vẽ |
| `--depth-medium-threshold` | `0.3333` | Ngưỡng tương đối bắt đầu band `medium`; phải hiệu chuẩn trên validation set |
| `--depth-near-threshold` | `0.6667` | Ngưỡng tương đối bắt đầu band `near`; không phải mét |
| `--depth-max-iqr` | `0.35` | Độ phân tán tối đa trong lõi bbox; vượt ngưỡng trả `unknown` |
| `--question` | câu hỏi mặc định | Câu hỏi ảnh dùng khi nhấn `v`; mặc định trả lời grounded từ detection |
| `--microphone` | `auto` | `auto`, index hoặc tên AVFoundation |
| `--listen-seconds` | `4` | Thời lượng thu mỗi lần nhấn `m` |
| `--max-seconds` | không giới hạn | Tự thoát sau N giây, hữu ích cho smoke test |
| `--log` | tự tạo trong `logs/` | File JSONL của phiên |
| `--no-tts` | tắt | Không phát âm thanh |
| `--voice` | `Linh` | Giọng macOS |
| `--speech-rate` | `165` | Tốc độ đọc từ/phút |

`--camera-fps`, `--detection-fps` và `--depth-fps` là mục tiêu/giới hạn. Tốc độ
thực tế phụ thuộc phần cứng, model, độ phân giải và việc MPS có khả dụng hay
không.

### 7.3 Phím điều khiển

| Phím | Chức năng | Điều gì xảy ra |
|---|---|---|
| `o` | OCR | Chụp frame hiện tại, kiểm tra chất lượng, đọc chữ và phát TTS |
| `s` | Mô tả cảnh | Gom nhóm label/hướng từ detection hiện tại và nói bằng tiếng Việt |
| `v` | Hỏi ảnh | Câu hỏi grounded dùng detection; dạng màu/hành động/trang phục mới gọi BLIP |
| `m` | Push-to-talk | Thu microphone, chạy Whisper và định tuyến ý định |
| `r` | Lặp lại | Đưa câu TTS gần nhất vào hàng đợi semantic |
| `x` | Dừng | Dừng âm thanh hiện tại và xóa hàng đợi |
| `q` hoặc `Esc` | Thoát | Đóng worker, camera, audio và ghi `session_ended` |

Nếu semantic worker đang bận chạy OCR/VQA/Whisper, lần nhấn `o`, `s`, `v` hoặc
`m` tiếp theo có thể bị từ chối. Thanh trạng thái sẽ hiển thị “Tác vụ ngữ nghĩa
đang chạy”.

### 7.4 Câu lệnh giọng nói

Sau khi nhấn `m`, hãy nói trong khoảng thời gian `--listen-seconds`. Hệ thống bỏ
dấu để dò các cụm ý định sau:

| Câu gợi ý | Ý định |
|---|---|
| “dừng”, “im lặng”, “stop” | Dừng audio |
| “lặp lại” | Lặp câu gần nhất |
| “đọc chữ”, “đọc văn bản”, “đọc cho tôi” | OCR frame đã chụp |
| “mô tả”, “xung quanh”, “khung cảnh” | Mô tả cảnh |
| Câu khác | Dùng transcript làm câu hỏi VQA |

Whisper có thể nhận sai trong môi trường ồn. Nếu RMS dưới `0.025`, STT trả
`audio_too_quiet` và không tải model. Các câu hallucination phổ biến kiểu
“hãy subscribe/đăng ký kênh” ở mức âm thanh rất thấp cũng bị loại.

### 7.5 Chọn camera và microphone

Liệt kê thiết bị AVFoundation:

```bash
ffmpeg -hide_banner -f avfoundation -list_devices true -i ""
```

FFmpeg thường kết thúc bằng lỗi “Error opening input” vì lệnh chỉ dùng để liệt
kê; danh sách thiết bị phía trên mới là phần cần đọc.

Ví dụ camera iPhone Continuity Camera:

```bash
./run_mvp.sh --camera 1 --microphone auto
```

Microphone `auto` không lưu một index cố định. Nó dò danh sách ở thời điểm thu
và ưu tiên microphone tích hợp theo tên. Có thể chỉ định thủ công:

```bash
./run_mvp.sh --camera 1 --microphone 2
./run_mvp.sh --camera 1 --microphone "MacBook Pro Microphone"
```

Tên cần khớp chính xác với danh sách AVFoundation. Dùng tên dễ ổn định hơn index
khi cắm/rút AirPods, iPhone hoặc thiết bị âm thanh ảo.

### 7.6 Chạy smoke demo tự thoát

```bash
./run_mvp.sh --camera 0 --no-depth --no-tts --max-seconds 5 \
  --log /tmp/secondeye_smoke.jsonl
```

Kiểm tra dòng cuối:

```bash
tail -n 1 /tmp/secondeye_smoke.jsonl
```

Dòng cuối bình thường có `"event":"session_ended"`.

## 8. Chạy camera detection riêng

Lệnh `camera` chỉ chạy detection, depth tùy chọn, overlay và TTS cảnh báo. Nó
không bật OCR, scene, VQA, microphone hoặc session log.

Không depth, không TTS:

```bash
secondeye camera --camera 0 --no-tts
```

Có depth và TTS:

```bash
secondeye camera --camera 0 --depth
```

Cấu hình đầy đủ:

```bash
secondeye camera \
  --camera 0 \
  --depth \
  --width 1280 --height 720 \
  --camera-fps 30 --display-fps 30 \
  --detection-fps 12 --depth-fps 3 \
  --max-depth-age 0.5 \
  --overlay-max-age 0.75 \
  --voice Linh --speech-rate 165
```

Khác với `demo`, `camera` mặc định không bật depth. Khi bật, ngưỡng tuổi depth
mặc định là 0,5 giây, nghiêm hơn demo. Nếu máy fallback CPU và depth chậm, kết
quả depth có thể bị loại; hãy quan sát trạng thái depth trên overlay trước khi
kết luận cảnh báo đang dùng depth.

Phím dùng được trong cửa sổ `camera`:

- `q` hoặc `Esc`: thoát;
- `x`: dừng âm thanh.

Lệnh `camera` không có `--max-seconds`; dùng `demo --max-seconds` cho smoke test
tự động.

## 9. Xử lý một ảnh

Lệnh `image` luôn chạy detection trước, sau đó chạy thêm module được yêu cầu.
Ảnh phải là định dạng OpenCV đọc được như JPEG hoặc PNG.

### 9.1 Detection

```bash
secondeye image \
  --source /duong/dan/anh.jpg \
  --no-tts
```

### 9.2 Detection và depth

```bash
secondeye image \
  --source /duong/dan/anh.jpg \
  --depth \
  --no-tts
```

Mỗi bbox có thể nhận thêm `relative_depth`, `depth_zone`, `depth_confidence`,
`depth_iqr`, `depth_reason` và `depth_sample_xyxy`. Không diễn giải
`relative_depth` thành mét. `depth_zone=unknown` là abstention chủ động khi vùng
lấy mẫu thiếu dữ liệu hoặc chứa nhiều lớp độ sâu. `depth_confidence = 1 - IQR`
chỉ là độ nhất quán không gian heuristic, không phải xác suất đúng đã calibration.

### 9.3 OCR

```bash
secondeye image \
  --source /duong/dan/van_ban.jpg \
  --ocr
```

Tắt đọc thành tiếng khi chỉ cần JSON:

```bash
secondeye image --source /duong/dan/van_ban.jpg --ocr --no-tts
```

Trước OCR, quality gate từ chối ảnh có cạnh ngắn dưới 240 px, quá tối, quá sáng
ít tương phản hoặc quá mờ. Kết quả bị từ chối có `abstained: true` và hướng dẫn
tiếng Việt thay vì transcript đoán mò.

### 9.4 VQA

```bash
secondeye image \
  --source /duong/dan/anh.jpg \
  --question "How many people are visible?" \
  --no-tts
```

Có thể hỏi tiếng Việt theo các mẫu: “Có bao nhiêu người?”, “Có gì phía trước?”,
“Chiếc ghế màu gì?”, “Người này đang làm gì?” hoặc “Người này mặc gì?”. Số lượng
và đồ vật phía trước dùng detection; màu sắc, hành động và trang phục được đổi
sang câu tiếng Anh có kiểm soát rồi mới gọi BLIP. Câu ngoài miền hỗ trợ sẽ bị từ
chối rõ ràng thay vì gửi tiếng Việt trực tiếp cho BLIP.

Không hỏi VQA để quyết định hướng đi hoặc an toàn giao thông. Một số cụm yêu cầu
điều hướng như “safe to cross” hoặc “which way should I go” bị chặn trước khi
gọi model.

Câu trả lời tiếng Anh phổ biến được ánh xạ/dịch an toàn sang
`spoken_answer_vi`. Nếu ngoài miền dịch có kiểm soát, kết quả có
`localization_abstained: true` và không được phát như một bản dịch đáng tin.

### 9.5 Chạy nhiều module và lưu JSON

```bash
mkdir -p results
secondeye image \
  --source /duong/dan/anh.jpg \
  --depth \
  --ocr \
  --question "What objects are visible?" \
  --no-tts \
  --output results/image_full.json
```

CLI vừa in JSON ra terminal vừa ghi file khi có `--output`.

### 9.6 Ảnh smoke test của project

Nếu chưa có ảnh mẫu:

```bash
python scripts/fetch_smoke_asset.py
```

Sau đó:

```bash
secondeye image \
  --source data/samples/ultralytics_bus.jpg \
  --depth \
  --question "How many people are visible?" \
  --no-tts
```

Script tải asset nên cần mạng ở lần chạy đó. Thư mục `data/` không được commit.

## 10. Nhận dạng giọng nói và thử TTS

### 10.1 STT một file WAV

```bash
secondeye transcribe --audio /duong/dan/lenh.wav
```

Whisper được gọi với ngôn ngữ `vi`. WAV cho phép hệ thống đo duration/RMS trước
inference. Với file không phải WAV, quality metadata có thể là `null` nhưng
pipeline vẫn thử gọi Whisper nếu định dạng được backend hỗ trợ.

Ví dụ output rút gọn:

```json
{
  "module": "stt",
  "success": true,
  "transcript": "hãy mô tả khung cảnh",
  "abstained": false,
  "audio_quality": {
    "duration_seconds": 3.1,
    "rms": 0.08
  },
  "latency_ms": 8200.0
}
```

### 10.2 Thử giọng TTS

```bash
secondeye speech-test
```

Tùy chỉnh:

```bash
secondeye speech-test \
  --voice Linh \
  --speech-rate 165 \
  --text "Cảnh báo, có ghế ở gần phía trước."
```

`speech-rate` phải dương. TTS chạy bằng lệnh macOS `say`, không gửi nội dung tới
API bên ngoài.

## 11. Cấu hình detection và risk

File mặc định: `configs/pretrained_indoor.toml`.

Chạy với một bản config khác:

```bash
secondeye --config /duong/dan/pretrained_indoor_custom.toml image \
  --source /duong/dan/anh.jpg --no-tts
```

Không sửa trực tiếp file chuẩn nếu cần so sánh thí nghiệm; hãy sao chép thành
file mới để giữ khả năng tái tạo.

### 11.1 `[model]`

| Trường | Hiện tại | Ý nghĩa |
|---|---:|---|
| `base_weights` | `yolo26m.pt` | Weights pretrained |
| `image_size` | `640` | Kích thước inference YOLO |
| `confidence_threshold` | `0.35` | Threshold chung cho nhánh custom cũ |
| `iou_threshold` | `0.50` | NMS IoU |
| `device` | `auto` | CUDA, MPS rồi CPU |

### 11.2 Mapping COCO

| Label COCO | Label SecondEye |
|---|---|
| `person` | `person` |
| `chair` | `chair` |
| `dining table` | `table` |
| `couch` | `sofa` |
| `bed` | `bed` |
| `backpack` | `backpack` |
| `handbag` | `handbag` |
| `suitcase` | `suitcase` |
| `bottle` | `bottle` |
| `potted plant` | `potted_plant` |
| `tv` | `tv` |
| `laptop` | `laptop` |
| `toilet` | `toilet` |
| `sink` | `sink` |
| `refrigerator` | `refrigerator` |

Threshold sáu lớp đầu kế thừa development benchmark cũ; chín lớp còn lại dùng
giá trị provisional `0.35`. Không mô tả các giá trị provisional như accuracy đã
được calibration.

### 11.3 `[risk]`

```toml
central_zone_fraction = 0.40
candidate_classes = [
  "person", "chair", "table", "sofa", "bed",
  "backpack", "handbag", "suitcase", "bottle", "potted_plant",
]
```

`tv`, `laptop`, `toilet`, `sink`, `refrigerator` vẫn có thể hiển thị nhưng không
nằm trong candidate set mặc định.

### 11.4 Depth band

Depth map được chuẩn hóa theo percentile 2–98% của chính frame đó. Depth map
phẳng hoặc không hữu hạn bị đánh dấu `usable=false`. Với mỗi detection, pipeline
thu bbox vào 20% mỗi cạnh ngang, 15% phía trên và 10% phía dưới để giảm nền, rồi
lấy median trong lõi. Nếu IQR của lõi lớn hơn `--depth-max-iqr`, hệ thống trả
`unknown` thay vì ép một band.

Các ngưỡng mặc định vẫn là:

```text
relative depth < 1/3       -> far
1/3 <= relative depth < 2/3 -> medium
relative depth >= 2/3      -> near
```

Giá trị lớn hơn nghĩa là tương đối gần hơn trong frame hiện tại. Không so sánh
trực tiếp giá trị giữa hai camera, hai cảnh hoặc hai thời điểm như một thước đo
vật lý. Có thể thay ngưỡng qua CLI, nhưng chỉ nên làm sau khi chọn trên validation
set khóa; việc thay ngưỡng không biến relative depth thành metric depth.

## 12. Định dạng kết quả và session log

### 12.1 JSON của lệnh `image`

Payload cấp cao có các khóa tùy module:

```json
{
  "frame": {
    "state": "IDLE",
    "detection": {
      "schema_name": "indoor_coco_baseline_v1",
      "detections": []
    },
    "depth": null,
    "alerts": [],
    "latency_ms": 123.4
  },
  "ocr": null,
  "vqa": null
}
```

Một detection có dạng:

```json
{
  "class_id": 0,
  "source_class_id": 0,
  "label": "person",
  "source_label": "person",
  "confidence": 0.95,
  "bbox_xyxy": [49.5, 399.2, 247.0, 903.3],
  "direction": "left",
  "obstacle_candidate": false,
  "candidate_reason": "outside_central_travel_zone",
  "depth_zone": "medium",
  "relative_depth": 0.56,
  "depth_confidence": 0.9685,
  "depth_iqr": 0.0315,
  "depth_reason": "relative_bbox_core",
  "depth_sample_xyxy": [89, 475, 208, 853]
}
```

Các trường `limitations` là một phần của contract; ứng dụng đọc JSON không nên
bỏ qua chúng khi hiển thị kết quả cho người dùng.

### 12.2 Session JSONL của demo

Nếu không truyền `--log`, file có tên tương tự:

```text
logs/session_20260830T041234Z_a1b2c3d4e5f6.jsonl
```

Mỗi dòng là một JSON độc lập:

```json
{
  "timestamp_utc": "2026-08-30T04:12:34.123456+00:00",
  "session_id": "a1b2c3d4e5f6",
  "event": "vision",
  "success": true,
  "payload": {}
}
```

Các event thường gặp:

| Event | Ý nghĩa |
|---|---|
| `session_started` | Bắt đầu demo, lưu camera/depth/control |
| `vision` | Một kết quả vision mới |
| `command` | Người dùng nhấn phím semantic; có `accepted` |
| `semantic_started` | Worker bắt đầu OCR/scene/VQA/microphone |
| `ocr`, `scene`, `vqa`, `microphone` | Kết quả semantic |
| `semantic_error` | Tác vụ gặp lỗi |
| `audio_repeat`, `audio_stopped` | Điều khiển audio |
| `session_ended` | Shutdown bình thường |

Đọc nhanh log:

```bash
wc -l logs/session_*.jsonl
tail -n 5 logs/session_*.jsonl
```

Đọc bằng Python:

```bash
python -c "import json, pathlib; p=pathlib.Path('logs/my_demo.jsonl'); print([json.loads(x)['event'] for x in p.read_text().splitlines()])"
```

`logs/` và JSON trong `results/` bị `.gitignore` loại khỏi Git vì có thể lớn,
thay đổi theo máy hoặc chứa dữ liệu nhạy cảm.

## 13. Lệnh detection/dataset cho nghiên cứu tương lai

Phần này không thuộc cách chạy MVP bình thường. Chỉ dùng nếu dự án chính thức mở
nhánh dataset/custom checkpoint với protocol và human review mới.

### 13.1 Smoke detection pretrained

```bash
secondeye-detection demo \
  --source data/samples/ultralytics_bus.jpg \
  --output-json results/detection.json \
  --output-image results/detection.jpg
```

Camera pretrained detection cũ:

```bash
secondeye-detection camera-demo --camera 0
```

### 13.2 Chuẩn bị và validate dataset

Giải nén ZIP an toàn:

```bash
secondeye-detection prepare \
  --archive /duong/dan/dataset.zip \
  --destination /duong/dan/dataset
```

Validate cấu trúc dataset:

```bash
secondeye-detection validate \
  --dataset /duong/dan/dataset \
  --output results/dataset_validation.json
```

### 13.3 Audit manifest nghiên cứu

```bash
secondeye-validate-manifest /duong/dan/manifest.csv --require-rows
```

Với vocabulary/split config riêng:

```bash
secondeye-validate-manifest /duong/dan/manifest.csv \
  --config /duong/dan/data_protocol.toml \
  --require-rows
```

Validator kiểm tra schema cột, ID, đường dẫn tương đối, license/consent, SHA-256,
privacy flag và leakage giữa group/scene/video/hash. Exit code `1` nghĩa là có
error audit; `2` nghĩa là không tải/parse được input.

### 13.4 Train, evaluate và export custom checkpoint

Không chạy các lệnh sau cho MVP pretrained hiện tại:

```bash
secondeye-detection train \
  --dataset /duong/dan/dataset \
  --name experiment_name

secondeye-detection evaluate \
  --model /duong/dan/best.pt \
  --dataset /duong/dan/dataset \
  --split test \
  --output results/evaluation.json

secondeye-detection export \
  --model /duong/dan/best.pt \
  --output-dir artifacts/export

secondeye-detection predict \
  --model /duong/dan/best.pt \
  --source /duong/dan/anh.jpg \
  --output-json results/predict.json \
  --output-image results/predict.jpg

secondeye-detection camera \
  --model /duong/dan/best.pt \
  --camera 0
```

`train` mặc định tiếp tục export sau huấn luyện; `--no-export` tắt bước đó. Một
custom checkpoint phải có class schema khớp hoàn toàn với config, nếu không
runtime chủ động báo lỗi thay vì dùng nhầm label.

Trước khi mở nhánh fine-tuning cần có taxonomy/version riêng, dữ liệu đã rà
license/privacy/dense-label, split độc lập, manifest/hash, metric protocol và lý
do nghiên cứu rõ ràng. Dataset lịch sử không phải dependency của MVP.

## 14. Quy trình kiểm thử và phát triển

### 14.1 Chạy source hiện tại

`run_mvp.sh` gọi package đã cài trong cache runtime, không tự động đọc file mới
trong `src/`. Sau khi sửa source, cập nhật runtime:

```bash
~/Library/Caches/SecondEye/venv/bin/python \
  -m pip install --no-deps --no-build-isolation .
```

Hoặc chạy module trực tiếp từ source để kiểm tra nhanh:

```bash
PYTHONPATH=src ~/Library/Caches/SecondEye/venv/bin/python \
  -m secondeye.system.cli doctor
```

Để cài lại toàn bộ dependency, dùng `./setup_mvp.sh`.

### 14.2 Bộ kiểm tra khuyến nghị

```bash
pytest -q -W error -p no:cacheprovider
python -m compileall -q src tests
python -m pip check
git diff --check
```

Nếu đã cài Ruff:

```bash
ruff check src tests
```

Build wheel không tải dependency mới:

```bash
python -m pip wheel . \
  --no-deps \
  --no-build-isolation \
  --wheel-dir /tmp/secondeye-wheel
```

### 14.3 Phân biệt ba mức kiểm tra

| Mức | Chứng minh được | Không chứng minh được |
|---|---|---|
| `doctor` | Dependency import được, có TTS/voice | Model chạy đúng, accuracy |
| Unit/integration test | Contract, validation, state, queue, fail-safe | Chất lượng model thực tế |
| Smoke/inference thật | Model và phần cứng chạy trên mẫu cụ thể | Độ tin cậy trên toàn bộ môi trường |

Không đánh dấu một chức năng “ổn định” chỉ vì `doctor` hoặc unit test pass.

## 15. Xử lý lỗi thường gặp

### 15.1 `run_mvp.sh` báo chưa có runtime

```text
Chưa có runtime. Chạy trước: .../setup_mvp.sh
```

Khắc phục:

```bash
./setup_mvp.sh
```

Nếu từng cài ở runtime tùy chỉnh, phải truyền lại đúng
`SECONDEYE_RUNTIME_DIR`.

### 15.2 `secondeye: command not found`

Activate runtime:

```bash
source ~/Library/Caches/SecondEye/venv/bin/activate
```

Hoặc dùng đường dẫn đầy đủ:

```bash
~/Library/Caches/SecondEye/venv/bin/secondeye doctor
```

### 15.3 Doctor báo thiếu dependency

Chạy lại cài đặt:

```bash
./setup_mvp.sh
```

Không trộn `.venv` trong repo với runtime cache. Kiểm tra Python thực tế:

```bash
which python
python -c "import sys, secondeye; print(sys.executable); print(secondeye.__file__)"
```

### 15.4 Không mở được camera

1. Kiểm tra quyền Camera trong macOS.
2. Liệt kê thiết bị bằng FFmpeg.
3. Thử lần lượt `--camera 0`, `--camera 1`.
4. Với Continuity Camera, mở khóa iPhone, bật Wi-Fi/Bluetooth và đặt iPhone gần
   Mac.
5. Đóng FaceTime/Teams/ứng dụng khác đang giữ camera.

```bash
secondeye camera --camera 0 --no-tts
secondeye camera --camera 1 --no-tts
```

### 15.5 Camera mở nhưng không có frame

Continuity Camera có thể mở device nhưng chưa trả frame. Mở khóa iPhone và thử
camera Mac tích hợp trước. Nếu cần smoke test có thời hạn, dùng
`demo --max-seconds`; lưu ý bộ đếm chỉ có ý nghĩa sau khi camera trả được frame.

### 15.6 Microphone thu im lặng hoặc sai thiết bị

Dùng mặc định:

```bash
./run_mvp.sh --microphone auto
```

Nếu vẫn sai, liệt kê thiết bị rồi chọn bằng tên:

```bash
ffmpeg -hide_banner -f avfoundation -list_devices true -i ""
./run_mvp.sh --microphone "MacBook Pro Microphone"
```

Kiểm tra quyền Microphone. Không hard-code index trong script cá nhân vì index
có thể đổi sau khi kết nối AirPods/iPhone/thiết bị ảo.

### 15.7 Không có giọng Linh

```bash
say -v '?' | grep '^Linh '
```

Nếu không có output, tải giọng trong Spoken Content hoặc tạm chạy `--no-tts`.

### 15.8 Model cố truy cập mạng dù đã cache

Sau khi xác nhận cache đầy đủ:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
./run_mvp.sh --camera 0
```

Không dùng các biến này ở lần tải model đầu.

### 15.9 OCR chậm hoặc đọc sai

- Chạy lại `./setup_mvp.sh` nếu thiếu
  `~/Library/Caches/SecondEye/venv/bin/secondeye-vision-ocr`.
- Trên macOS, JSON nên có `engine: "Apple Vision"`. Nếu helper lỗi, hệ thống mới
  lazy-load PaddleOCR và ghi `fallback_from`/`fallback_error`.
- Lần đầu dùng fallback có thể tải/khởi tạo model PaddleOCR.
- Giữ ảnh đủ sáng, nét và cạnh ngắn ít nhất 240 px.
- Đưa văn bản gần camera hơn.
- Dùng `--no-tts` khi cần đo latency riêng.
- Không coi confidence cao là bảo đảm transcript đúng.

### 15.10 VQA không dịch câu trả lời

Đây có thể là hành vi fail-safe. Nếu câu tiếng Anh nằm ngoài từ vựng/mẫu câu thị
giác, pipeline trả `localization_abstained: true` thay vì đọc bản dịch chưa kiểm
chứng. Hãy đặt câu hỏi ngắn và cụ thể; không bật Marian fallback cho luồng an
toàn nếu chưa có đánh giá độc lập.

### 15.11 Có detection nhưng không có cảnh báo

Kiểm tra lần lượt:

- depth đã bật chưa;
- `depth_zone` có phải `near` không;
- `obstacle_candidate` có phải `true` không;
- bbox có nằm vùng giữa không;
- lớp có nằm trong `risk.candidate_classes` không;
- detection đã xuất hiện đủ 2 frame chưa;
- cảnh báo có đang trong cooldown 4 giây không;
- depth có bị loại vì `depth_age_ms` quá lớn không.
- `depth_synchronized` có phải `true` không;
- `depth_reason` có phải `ambiguous_bbox_depth` hoặc
  `insufficient_valid_depth` không.

Việc không có cảnh báo không có nghĩa đường đi an toàn.

### 15.12 MPS lỗi hoặc hiệu năng không ổn định

Config `device = "auto"` ưu tiên MPS rồi mới CPU. Semantic model trong demo dùng
CPU để tránh xung đột Metal. Nếu MPS không khả dụng, pipeline có thể chậm đáng
kể; giảm độ phân giải/tần suất mục tiêu hoặc tắt depth để chẩn đoán:

```bash
./run_mvp.sh --camera 0 --no-depth --no-tts \
  --width 640 --height 480 --detection-fps 6
```

## 16. Giới hạn cần hiểu đúng

- Đây là integration baseline pretrained, không phải model đã fine-tune cho môi
  trường của người khiếm thị.
- Detection chỉ hỗ trợ 15 lớp được ánh xạ; candidate cảnh báo chỉ gồm 10 lớp.
- Depth monocular là tương đối, không đo mét và có thể thay đổi giữa các frame.
- Cảnh báo chỉ dựa trên label + hướng, chưa có object tracking ID đầy đủ.
- Apple Vision/PaddleOCR, BLIP và Whisper vẫn có thể sai dù `success: true`.
- Confidence VQA là token-generation score chưa calibration, không phải xác suất
  câu trả lời đúng.
- Bộ dịch chỉ bảo đảm trong miền câu trả lời thị giác được hỗ trợ; ngoài miền sẽ
  abstain.
- Camera/microphone index và quyền truy cập phụ thuộc máy/macOS.
- Tốc độ cấu hình là mục tiêu; tốc độ thực tế phụ thuộc MPS/CPU và model đang
  chạy.
- Không có kết quả phát hiện/cảnh báo không chứng minh cảnh an toàn.
- Fine-tuning chỉ là hướng tương lai tùy chọn, không phải bước còn thiếu bắt buộc
  của MVP hiện tại.

Tài liệu chuyên sâu về detection nằm tại
[`local-yolo26-runtime.md`](local-yolo26-runtime.md). Phạm vi dự án và tiêu chí
MVP nằm tại [`../current/project-charter.md`](../current/project-charter.md).
