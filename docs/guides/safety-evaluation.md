# Đánh giá safety không fine-tuning

Cập nhật: 2026-09-03.

Tài liệu này mô tả cách kiểm tra luồng metric-depth/geometry/tracking trước khi
demo hoặc pilot. Unit test không chứng minh hệ thống an toàn trong môi trường
thật; cần video/scenario độc lập, ground truth do người đánh giá gán và replay
cùng một cấu hình đã khóa.

## Dữ liệu đầu vào

`secondeye-evaluate-safety` nhận JSONL, mỗi dòng là một thời điểm quan sát:

```json
{"timestamp_s":0.0,"hazard_present":false,"alert":false,"source_age_ms":120}
{"timestamp_s":1.0,"hazard_present":true,"hazard_id":"chair_01","critical":true,"alert":false,"source_age_ms":180}
{"timestamp_s":1.35,"hazard_present":true,"hazard_id":"chair_01","critical":true,"alert":true,"source_age_ms":210}
```

Trường bắt buộc:

- `timestamp_s`: thời gian tăng dần trong scenario;
- `hazard_present`: ground truth cho biết vật cản có trong corridor hay không;
- `alert`: runtime có phát cảnh báo ở thời điểm đó hay không;
- `hazard_id`: bắt buộc trên mọi dòng `hazard_present=true`, ổn định trong suốt
  một lần xuất hiện của vật cản.

Trường tùy chọn:

- `critical`: tình huống có nguy cơ va chạm nghiêm trọng;
- `source_age_ms`: tuổi frame tại thời điểm cảnh báo, dùng đếm stale alert.

Một hazard kéo dài nhiều frame chỉ được tính là **một sự kiện**, tránh làm recall
đẹp giả tạo do đếm từng frame.

## Chạy báo cáo

```bash
secondeye-evaluate-safety \
  --input data/local/safety/scenario_frames.jsonl \
  --max-source-age-ms 750 \
  --output results/safety_metrics.json
```

Output gồm:

- `hazard_event_recall` và `critical_event_recall`;
- `false_alerts_per_minute`;
- `stale_alerts`;
- `alert_latency_p50_ms`, `p95_ms`, `p99_ms`.

Nếu không có sự kiện hoặc thời lượng bằng 0, metric tương ứng là `null`, không
được thay bằng 0 hoặc 1.

## Protocol tối thiểu

1. Khóa commit, config và hash model trước khi quay/chạy replay.
2. Tách cảnh thành indoor sáng/tối, sàn có texture/bóng, vật thấp, vật trong
   suốt, người di chuyển, che khuất, camera rung và corridor trống.
3. Giữ scenario test độc lập với mọi lần chọn threshold.
4. Hai người đánh giá độc lập `hazard_present`, `critical`, thời điểm bắt đầu/kết
   thúc; giải quyết bất đồng trước khi tính metric.
5. Báo cáo cả lỗi fit sàn (`geometry.usable=false`) như miss/abstention tùy
   protocol đã đăng ký trước.
6. Phân tích từng false negative, stale alert và false emergency, không chỉ xem
   số trung bình.

Project chưa đặt ngưỡng đạt mang tính chứng nhận. Trước pilot có người dùng, hội
đồng/nhóm nghiên cứu phải phê duyệt tiêu chí cho recall tình huống critical,
false alerts/phút, P95/P99 latency và thời lượng mất depth; không chọn tiêu chí
sau khi đã xem kết quả test.

## Tích hợp depth sensor/LiDAR

`AlignedMetricDepthFrame` yêu cầu depth H×W đã đăng ký vào RGB, cùng timestamp và
intrinsics `fx`, `fy`, `cx`, `cy`. `SynchronizedDepthCapture` đưa cặp RGB/depth
vào runtime mà không tái sử dụng depth của frame khác. Adapter ARKit/AVFoundation
phải:

- chuyển depth sang mét và giữ `NaN` cho pixel không hợp lệ;
- resample/register depth đúng hệ tọa độ của ảnh RGB;
- dùng timestamp monotonic cùng clock domain;
- truyền intrinsics sau khi scale theo resolution;
- ưu tiên pixel confidence cao khi nguồn sensor cung cấp confidence.

Không tự nội suy một depth map cũ vào frame RGB mới để vượt freshness gate.
