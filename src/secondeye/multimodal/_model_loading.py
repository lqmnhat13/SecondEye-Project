"""Shared loading helpers for locally cached pretrained models."""

from __future__ import annotations

import os
from typing import Any


def _offline_requested() -> bool:
    return any(
        os.environ.get(name, "").strip().casefold() in {"1", "true", "yes", "on"}
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def _from_pretrained_offline_first(
    factory: Any, model_name: str, **pretrained_kwargs: Any
) -> Any:
    """Use an existing cache without a network metadata retry penalty."""
    try:
        return factory.from_pretrained(
            model_name, local_files_only=True, **pretrained_kwargs
        )
    except OSError as cache_error:
        if _offline_requested():
            raise RuntimeError(
                f"Cache model chưa đầy đủ cho '{model_name}' trong chế độ offline. "
                "Kết nối mạng một lần và chạy lại lệnh không đặt "
                "HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE."
            ) from cache_error
    try:
        return factory.from_pretrained(model_name, **pretrained_kwargs)
    except OSError as download_error:
        raise RuntimeError(
            f"Không tải được model '{model_name}' và cache local chưa đầy đủ. "
            "Kiểm tra kết nối mạng rồi chạy lại."
        ) from download_error
