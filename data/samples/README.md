# Smoke-test assets

`scripts/fetch_smoke_asset.py` tải ảnh `bus.jpg` từ kho asset chính thức của Ultralytics và lưu thành `ultralytics_bus.jpg`.

- Nguồn: https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg
- Mục đích: kiểm tra đường ống inference, log và latency sơ bộ.
- Không dùng để đo accuracy, chọn ngưỡng hoặc báo cáo kết quả nghiên cứu.
- Ảnh nhị phân không được commit; script giúp tải lại có kiểm tra định dạng JPEG.

