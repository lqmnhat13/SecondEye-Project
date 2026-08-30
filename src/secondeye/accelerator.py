"""Process-wide coordination for accelerator operations."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator


_MPS_LOCK = threading.RLock()


def _synchronize_mps(torch_module: Any | None) -> None:
    if torch_module is None:
        return
    mps = getattr(torch_module, "mps", None)
    synchronize = getattr(mps, "synchronize", None)
    if callable(synchronize):
        synchronize()


@contextmanager
def accelerator_guard(
    device: str | Any,
    torch_module: Any | None = None,
) -> Iterator[None]:
    """Serialize Apple MPS model loading and inference across worker threads.

    PyTorch/Metal operations are asynchronous. Synchronizing before releasing
    the process-wide lock prevents another model from opening a command encoder
    on a command buffer that is still being committed by the previous model.
    """
    if str(device).lower() != "mps":
        yield
        return
    with _MPS_LOCK:
        _synchronize_mps(torch_module)
        try:
            yield
        finally:
            _synchronize_mps(torch_module)
