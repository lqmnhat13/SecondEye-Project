# Mẫu manifest và annotation

Các CSV ở đây chỉ chứa header; số mẫu thực tế hiện là **0**. Sao chép mẫu sang vùng dữ liệu local trước khi nhập nhãn. Không commit ảnh, âm thanh, dữ liệu cá nhân hoặc consent record.

- `sample_manifest.csv`: một dòng cho mỗi cặp mẫu–tác vụ; là nguồn chuẩn cho split.
- `obstacle_annotations.csv`: một dòng cho mỗi vật thể được gán nhãn.
- `ocr_annotations.csv`: một dòng cho mỗi vùng chữ.
- `vqa_annotations.csv`: một dòng cho mỗi câu hỏi; mỗi ảnh dự kiến có 2–3 câu.

Quy ước ID không chứa tên, email, số điện thoại hoặc thông tin nhận dạng:

- `sample_id`: `obs_...`, `ocr_...` hoặc `vqa_...` khớp với tác vụ.
- `group_id`: bắt đầu bằng `grp_`; gom toàn bộ near-duplicate, burst, video và cùng cảnh thực.
- `capture_session_id`: bắt đầu bằng `ses_`.
- `scene_id`: bắt đầu bằng `scn_`.
- `video_id`: để trống cho ảnh độc lập hoặc bắt đầu bằng `vid_`.

Kiểm tra manifest:

```bash
python -m secondeye.data.protocol data/local/sample_manifest.csv \
  --config configs/data_protocol.toml --require-rows
```

Chi tiết nhãn và quy trình review nằm trong `docs/data/annotation_guide.md`.

