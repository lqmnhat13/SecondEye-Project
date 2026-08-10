"""Local speech input/output adapters."""

from __future__ import annotations

import platform
import subprocess
import threading
import time
from pathlib import Path


class MacOSTextToSpeech:
    """Interruptible local TTS backed by the macOS `say` command."""

    def __init__(self, voice: str | None = None, rate: int = 185) -> None:
        if platform.system() != "Darwin":
            raise RuntimeError("MacOSTextToSpeech chỉ hỗ trợ macOS")
        self.voice = voice
        self.rate = rate
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def stop(self) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
            self._process = None

    def speak(self, text: str, *, interrupt: bool = False) -> None:
        text = text.strip()
        if not text:
            return
        with self._lock:
            if interrupt and self._process is not None and self._process.poll() is None:
                self._process.terminate()
            elif self._process is not None and self._process.poll() is None:
                return
            command = ["say", "-r", str(self.rate)]
            if self.voice:
                command.extend(["-v", self.voice])
            command.append(text)
            self._process = subprocess.Popen(command, text=True)


class WhisperSpeechToText:
    """Pretrained Whisper adapter for recorded local audio files."""

    def __init__(
        self, model_name: str = "openai/whisper-small", device: str = "auto"
    ) -> None:
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                'Thiếu STT runtime. Chạy: python -m pip install ".[multimodal]"'
            ) from exc
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda:0"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.model_name = model_name
        self.pipeline = pipeline(
            "automatic-speech-recognition", model=model_name, device=device
        )

    def transcribe(self, audio_path: Path) -> dict[str, object]:
        audio_path = audio_path.expanduser().resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        started = time.perf_counter()
        result = self.pipeline(str(audio_path), generate_kwargs={"language": "vi"})
        return {
            "schema_version": "1.0",
            "module": "stt",
            "success": True,
            "model": self.model_name,
            "transcript": str(result.get("text", "")).strip(),
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
