"""Unicode-capable OpenCV overlays rendered through Pillow."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypeAlias

from .localization import localize_depth_zone, localize_label


TextOverlay: TypeAlias = tuple[str, tuple[int, int], tuple[int, int, int], int]

_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


class UnicodeTextRenderer:
    """Draw UTF-8 text onto a BGR NumPy image while preserving the input array."""

    def __init__(self, font_path: str | Path | None = None) -> None:
        self.font_path = self._resolve_font(font_path)
        self._fonts: dict[int, Any] = {}

    @staticmethod
    def _resolve_font(font_path: str | Path | None) -> str:
        configured = font_path or os.environ.get("SECONDEYE_FONT")
        if configured:
            resolved = Path(configured).expanduser()
            if not resolved.is_file():
                raise FileNotFoundError(f"Không tìm thấy font Unicode: {resolved}")
            return str(resolved)
        for candidate in _FONT_CANDIDATES:
            if Path(candidate).is_file():
                return candidate
        try:
            from PIL import ImageFont

            ImageFont.truetype("DejaVuSans.ttf", size=20)
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Không tìm thấy font hỗ trợ tiếng Việt. Đặt biến SECONDEYE_FONT "
                "tới một file TTF/OTF Unicode."
            ) from exc
        return "DejaVuSans.ttf"

    def _font(self, size: int) -> Any:
        if size not in self._fonts:
            try:
                from PIL import ImageFont
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise RuntimeError(
                    'Thiếu Pillow. Chạy: python -m pip install ".[detection]"'
                ) from exc
            self._fonts[size] = ImageFont.truetype(self.font_path, size=size)
        return self._fonts[size]

    def draw_bgr(self, image: Any, overlays: list[TextOverlay]) -> Any:
        if not overlays:
            return image
        try:
            import numpy as np
            from PIL import Image, ImageDraw
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("Thiếu NumPy/Pillow để vẽ chữ tiếng Việt") from exc
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Overlay cần ảnh OpenCV BGR ba kênh")
        canvas = Image.fromarray(image[:, :, ::-1])
        draw = ImageDraw.Draw(canvas)
        for value, position, bgr, size in overlays:
            rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
            draw.text(
                position,
                value,
                font=self._font(size),
                fill=rgb,
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )
        image[...] = np.asarray(canvas)[:, :, ::-1]
        return image


def draw_detection_overlays(
    cv2: Any,
    image: Any,
    detections: list[dict[str, Any]],
) -> list[TextOverlay]:
    """Draw detection boxes and return localized text queued for Pillow."""
    overlays: list[TextOverlay] = []
    for detection in detections:
        x1, y1, x2, y2 = (int(value) for value in detection["bbox_xyxy"])
        safety_evaluable = bool(detection.get("safety_evaluable"))
        zone = detection.get("depth_zone")
        if safety_evaluable and zone == "emergency":
            color = (255, 0, 255)
        elif safety_evaluable and zone == "near":
            color = (0, 0, 255)
        elif detection.get("distance_m") is not None:
            color = (0, 200, 255)
        else:
            color = (0, 200, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        caption = (
            f"{localize_label(str(detection['label']))} "
            f"{float(detection['confidence']):.2f}"
        )
        if detection.get("depth_zone"):
            caption += " " + localize_depth_zone(str(detection["depth_zone"]))
        if detection.get("distance_m") is not None:
            caption += f" {float(detection['distance_m']):.1f}m"
        if detection.get("track_id") is not None:
            caption += f" #{int(detection['track_id'])}"
        overlays.append((caption, (x1, max(2, y1 - 24)), color, 18))
    return overlays
