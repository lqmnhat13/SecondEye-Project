"""Local speech input/output adapters."""

from __future__ import annotations

import platform
import re
import subprocess
import threading
import time
from pathlib import Path


_SPEECH_REPLACEMENTS = {
    "SecondEye": "Se-cần Ai",
    "Second Eye": "Se-cần Ai",
    "YOLO": "Yô-lô",
    "OCR": "Ô Xi A",
    "VQA": "Vi Kiu Ây",
}

_VQA_VIETNAMESE = {
    "zero": "không",
    "one": "một",
    "two": "hai",
    "three": "ba",
    "four": "bốn",
    "five": "năm",
    "six": "sáu",
    "seven": "bảy",
    "eight": "tám",
    "nine": "chín",
    "ten": "mười",
    "yes": "có",
    "no": "không",
    "left": "bên trái",
    "right": "bên phải",
    "center": "ở giữa",
    "person": "người",
    "people": "người",
    "chair": "ghế",
    "table": "bàn",
    "sofa": "ghế sofa",
    "bed": "giường",
    "backpack": "ba lô",
    "handbag": "túi xách",
    "suitcase": "va li",
    "bottle": "chai",
    "plant": "chậu cây",
    "television": "ti vi",
    "tv": "ti vi",
    "laptop": "máy tính xách tay",
    "toilet": "bồn cầu",
    "sink": "bồn rửa",
    "refrigerator": "tủ lạnh",
}


def normalize_vietnamese_speech(text: str) -> str:
    """Normalize whitespace and project terms before sending text to `say`."""
    normalized = " ".join(text.strip().split())
    for source, target in _SPEECH_REPLACEMENTS.items():
        normalized = re.sub(re.escape(source), target, normalized, flags=re.IGNORECASE)
    return normalized


def localize_vqa_answer(answer: str) -> tuple[str, bool]:
    """Translate safe short BLIP answers; abstain from reading unknown English."""
    normalized = " ".join(answer.strip().lower().split())
    if not normalized:
        return "Tôi chưa đủ chắc chắn để trả lời.", True
    if normalized in _VQA_VIETNAMESE:
        return _VQA_VIETNAMESE[normalized], False
    words = re.findall(r"[a-z]+", normalized)
    if words and all(word in _VQA_VIETNAMESE for word in words):
        return " ".join(_VQA_VIETNAMESE[word] for word in words), False
    if words and normalized.isascii():
        return (
            "Tôi đã có kết quả bằng tiếng Anh nhưng chưa thể đọc tiếng Việt chính xác.",
            True,
        )
    return normalize_vietnamese_speech(answer), False


def macos_voice_available(voice: str) -> bool:
    if platform.system() != "Darwin":
        return False
    try:
        output = subprocess.check_output(
            ["say", "-v", "?"], text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return any(line.startswith(f"{voice} ") for line in output.splitlines())


class MacOSTextToSpeech:
    """Interruptible local TTS backed by the macOS `say` command."""

    def __init__(self, voice: str | None = "Linh", rate: int = 165) -> None:
        if platform.system() != "Darwin":
            raise RuntimeError("MacOSTextToSpeech chỉ hỗ trợ macOS")
        if rate <= 0:
            raise ValueError("Tốc độ TTS phải dương")
        if voice is not None and not voice.strip():
            raise ValueError("Tên giọng TTS không được rỗng")
        self.voice = None if voice is None else voice.strip()
        if self.voice is not None and not macos_voice_available(self.voice):
            raise RuntimeError(
                f"macOS chưa cài giọng '{self.voice}'. Mở System Settings > "
                "Accessibility > Spoken Content để tải giọng tiếng Việt."
            )
        self.rate = rate
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def stop(self) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
            self._process = None

    def wait(self) -> None:
        with self._lock:
            process = self._process
        if process is not None:
            process.wait()

    def speak(self, text: str, *, interrupt: bool = False) -> None:
        text = normalize_vietnamese_speech(text)
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
