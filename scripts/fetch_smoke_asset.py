"""Download the official Ultralytics smoke-test image reproducibly."""

from __future__ import annotations

import hashlib
import ssl
import urllib.request
from pathlib import Path

import certifi


URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "data" / "samples" / "ultralytics_bus.jpg"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(URL, headers={"User-Agent": "SecondEye-research/0.1"})
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=30, context=ssl_context) as response:
        payload = response.read()

    if not payload.startswith(b"\xff\xd8\xff"):
        raise ValueError("Downloaded payload is not a JPEG image")

    OUTPUT.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    print(f"Downloaded {OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
