"""Shared loading helpers for locally cached pretrained models."""

from __future__ import annotations

from typing import Any


def _from_pretrained_offline_first(factory: Any, model_name: str) -> Any:
    """Use an existing cache without a network metadata retry penalty."""
    try:
        return factory.from_pretrained(model_name, local_files_only=True)
    except OSError:
        return factory.from_pretrained(model_name)
