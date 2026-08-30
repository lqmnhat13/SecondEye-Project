"""Image quality gates shared by on-demand OCR and VQA."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ImageQuality:
    acceptable: bool
    reason: str
    guidance_vi: str
    brightness: float
    contrast: float
    sharpness: float
    width: int
    height: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def assess_image_quality(image: Any) -> ImageQuality:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Thiếu NumPy cho image quality gate") from exc
    if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim not in (2, 3):
        raise ValueError("quality input phải là ảnh OpenCV không rỗng")
    height, width = map(int, image.shape[:2])
    if image.ndim == 2:
        gray = image.astype(np.float32)
    else:
        bgr = image[:, :, :3].astype(np.float32)
        gray = 0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2]
    brightness = float(gray.mean())
    contrast = float(gray.std())
    horizontal = np.diff(gray, axis=1)
    vertical = np.diff(gray, axis=0)
    sharpness = float((horizontal.var() + vertical.var()) / 2.0)
    if min(width, height) < 240:
        reason, guidance = "resolution_too_low", "Ảnh quá nhỏ, hãy đưa camera gần hơn."
    elif brightness < 28.0:
        reason, guidance = "too_dark", "Ảnh quá tối, hãy tăng ánh sáng rồi thử lại."
    elif brightness > 235.0 and contrast < 25.0:
        reason, guidance = (
            "overexposed",
            "Ảnh bị quá sáng, hãy đổi góc camera rồi thử lại.",
        )
    elif sharpness < 22.0:
        reason, guidance = (
            "too_blurry",
            "Ảnh bị mờ, hãy giữ camera ổn định rồi thử lại.",
        )
    else:
        reason, guidance = "ok", "Ảnh đủ chất lượng."
    return ImageQuality(
        acceptable=reason == "ok",
        reason=reason,
        guidance_vi=guidance,
        brightness=round(brightness, 2),
        contrast=round(contrast, 2),
        sharpness=round(sharpness, 2),
        width=width,
        height=height,
    )
