# SecondEye

SecondEye là baseline tích hợp AI đa phương thức chạy local trên macOS. Phiên bản
hiện tại **không fine-tune**: hệ thống ghép các model pretrained cho detection,
relative depth, OCR, VQA, STT và TTS để hoàn thiện kiến trúc end-to-end trước.

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

Ảnh theo yêu cầu -> PaddleOCR hoặc BLIP VQA -> TTS
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
rác. Các lớp an toàn đặc thù được bảo tồn local cho giai đoạn fine-tuning sau.

## Cài đặt

Yêu cầu: Python 3.11 trên macOS Apple Silicon.

```bash
cd /Users/lenhat/Documents/SecondEye-Project
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Detection, camera và test
python -m pip install -e ".[dev,detection]"

# Depth, VQA và Whisper STT
python -m pip install -e ".[multimodal]"

# OCR cài riêng vì PaddlePaddle là dependency lớn
python -m pip install -e ".[ocr]"
```

Model pretrained được tải vào cache ở lần chạy đầu. Sau khi đã tải đủ model,
các module local có thể chạy không cần gửi ảnh lên API.

## Kiểm tra môi trường

```bash
secondeye doctor
pytest
```

`doctor` chỉ kiểm tra dependency, không tải model.

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
- Depth phải xác nhận band `near` trước khi phát cảnh báo gần.
- `near/medium/far` là độ sâu tương đối theo frame, không phải mét.
- Cooldown ngăn cùng một cảnh báo bị đọc liên tục.
- VQA confidence thấp phải abstain thay vì cố trả lời.

## Cấu trúc source public

```text
configs/pretrained_indoor.toml   schema và runtime config
src/secondeye/detection/         YOLO26 COCO adapter và risk candidate
src/secondeye/multimodal/        depth, OCR, VQA, STT và TTS adapters
src/secondeye/system/            state machine, orchestrator và unified CLI
tests/                           unit tests không cần tải model
scripts/fetch_smoke_asset.py     tải ảnh smoke test có thể tái tạo
```

GitHub chỉ chứa source code, test, runtime config và README. Dataset, ảnh, model
weights, log, artifact, báo cáo, DOCX/PDF và kết quả chạy được giữ local và bị
`.gitignore` loại khỏi commit.

## Giới hạn đã biết

- 15 lớp mới là schema integration dễ hơn, không phải taxonomy an toàn hoàn chỉnh.
- Chín threshold mới ngoài sáu lớp benchmark cũ đang là giá trị provisional 0.35.
- Relative monocular depth không cung cấp khoảng cách tuyệt đối.
- PaddleOCR, BLIP và Whisper cần benchmark riêng trên dữ liệu tiếng Việt/thực tế.
- Camera demo cần kiểm thử trực tiếp vì quyền camera và chỉ số thiết bị phụ thuộc macOS.
- Fine-tuning và đánh giá test độc lập được hoãn đến sau khi integration ổn định.

Ultralytics được sử dụng theo điều kiện giấy phép tương ứng; cần rà giấy phép của
từng model và dependency trước khi phát hành hoặc thương mại hóa.
