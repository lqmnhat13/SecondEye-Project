"""Local speech input/output adapters."""

from __future__ import annotations

import array
import math
import platform
import re
import shutil
import subprocess
import threading
import time
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path

from secondeye.accelerator import accelerator_guard


_SPEECH_REPLACEMENTS = {
    "SecondEye": "Se-cần Ai",
    "Second Eye": "Se-cần Ai",
    "YOLO": "Yô-lô",
    "OCR": "Ô Xi A",
    "VQA": "Vi Kiu Ây",
}

_VQA_VIETNAMESE = {
    "a": "một",
    "an": "một",
    "and": "và",
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
    "blue": "xanh dương",
    "red": "đỏ",
    "black": "đen",
    "white": "trắng",
    "green": "xanh lá",
    "yellow": "vàng",
    "brown": "nâu",
    "gray": "xám",
    "grey": "xám",
    "orange": "cam",
    "pink": "hồng",
    "purple": "tím",
    "tan": "nâu nhạt",
    "multicolored": "nhiều màu",
    "blue and white": "xanh dương và trắng",
    "black and white": "đen và trắng",
    "red and white": "đỏ và trắng",
    "person": "người",
    "people": "người",
    "man": "người đàn ông",
    "woman": "người phụ nữ",
    "chair": "ghế",
    "chairs": "ghế",
    "table": "bàn",
    "tables": "bàn",
    "sofa": "ghế sofa",
    "bed": "giường",
    "backpack": "ba lô",
    "handbag": "túi xách",
    "suitcase": "va li",
    "bottle": "chai",
    "bottles": "chai",
    "plant": "chậu cây",
    "television": "ti vi",
    "tv": "ti vi",
    "laptop": "máy tính xách tay",
    "toilet": "bồn cầu",
    "sink": "bồn rửa",
    "refrigerator": "tủ lạnh",
    "bus": "xe buýt",
    "car": "ô tô",
    "truck": "xe tải",
    "bicycle": "xe đạp",
    "bike": "xe đạp",
    "motorcycle": "xe máy",
    "motorbike": "xe máy",
    "train": "tàu hỏa",
    "airplane": "máy bay",
    "plane": "máy bay",
    "boat": "thuyền",
    "bench": "ghế băng",
    "umbrella": "ô",
    "bird": "chim",
    "cat": "mèo",
    "dog": "chó",
    "horse": "ngựa",
    "cow": "bò",
    "sheep": "cừu",
    "elephant": "voi",
    "bear": "gấu",
    "zebra": "ngựa vằn",
    "giraffe": "hươu cao cổ",
    "cup": "cốc",
    "glass": "ly",
    "bowl": "bát",
    "fork": "nĩa",
    "knife": "dao",
    "spoon": "thìa",
    "plate": "đĩa",
    "banana": "chuối",
    "apple": "táo",
    "sandwich": "bánh mì kẹp",
    "broccoli": "bông cải xanh",
    "carrot": "cà rốt",
    "pizza": "bánh pizza",
    "donut": "bánh vòng",
    "cake": "bánh ngọt",
    "clock": "đồng hồ",
    "vase": "bình hoa",
    "book": "sách",
    "books": "sách",
    "scissors": "kéo",
    "mouse": "chuột máy tính",
    "keyboard": "bàn phím",
    "phone": "điện thoại",
    "cellphone": "điện thoại",
    "microwave": "lò vi sóng",
    "oven": "lò nướng",
    "toaster": "máy nướng bánh mì",
    "clothes": "quần áo",
    "shirt": "áo sơ mi",
    "t-shirt": "áo thun",
    "pants": "quần dài",
    "jeans": "quần jean",
    "dress": "váy",
    "jacket": "áo khoác",
    "coat": "áo khoác dài",
    "hat": "mũ",
    "shoes": "giày",
    "shorts": "quần đùi",
    "sitting": "đang ngồi",
    "standing": "đang đứng",
    "walking": "đang đi bộ",
    "running": "đang chạy",
    "eating": "đang ăn",
    "drinking": "đang uống",
    "holding": "đang cầm",
    "playing": "đang chơi",
    "riding": "đang đi xe",
}


def normalize_vietnamese_speech(text: str) -> str:
    """Normalize whitespace and project terms before sending text to `say`."""
    normalized = " ".join(text.strip().split())
    for source, target in _SPEECH_REPLACEMENTS.items():
        normalized = re.sub(re.escape(source), target, normalized, flags=re.IGNORECASE)
    return normalized


def localize_vqa_answer(answer: str) -> tuple[str, bool]:
    """Translate common short BLIP answers and flag text needing the full model."""
    normalized = " ".join(answer.strip().lower().split())
    if not normalized:
        return "Tôi chưa đủ chắc chắn để trả lời.", True
    if normalized in _VQA_VIETNAMESE:
        return _VQA_VIETNAMESE[normalized], False
    words = re.findall(r"[a-z]+", normalized)
    if (
        " and " in normalized
        and words
        and all(word in _VQA_VIETNAMESE for word in words)
    ):
        return " ".join(_VQA_VIETNAMESE[word] for word in words), False
    if words and normalized.isascii():
        return answer.strip(), True
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
        self,
        model_name: str = "openai/whisper-small",
        device: str = "auto",
        minimum_wav_rms: float = 0.025,
    ) -> None:
        if minimum_wav_rms < 0:
            raise ValueError("minimum_wav_rms không được âm")
        self.model_name = model_name
        self.requested_device = device
        self.minimum_wav_rms = minimum_wav_rms
        self.pipeline = None
        self.device: str | None = None
        self._torch = None
        self._lock = threading.Lock()

    def _get_pipeline(self):
        with self._lock:
            if self.pipeline is not None:
                return self.pipeline
            try:
                import torch
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError(
                    'Thiếu STT runtime. Chạy: python -m pip install ".[multimodal]"'
                ) from exc
            device = self.requested_device
            if device == "auto":
                if torch.cuda.is_available():
                    device = "cuda:0"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            with accelerator_guard(device, torch):
                self.pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=self.model_name,
                    device=device,
                )
            self.device = device
            self._torch = torch
            return self.pipeline

    @staticmethod
    def _wav_quality(audio_path: Path) -> dict[str, float] | None:
        try:
            with wave.open(str(audio_path), "rb") as stream:
                frames = stream.readframes(stream.getnframes())
                width = stream.getsampwidth()
                duration = stream.getnframes() / max(1, stream.getframerate())
        except (wave.Error, OSError):
            return None
        if not frames or width <= 0:
            return {"duration_seconds": round(duration, 3), "rms": 0.0}
        sample_types = {1: "B", 2: "h", 4: "i"}
        sample_type = sample_types.get(width)
        if sample_type is None:
            return None
        samples = array.array(sample_type)
        samples.frombytes(frames)
        if width == 1:
            centered = (float(value) - 128.0 for value in samples)
            maximum = 128.0
        else:
            centered = (float(value) for value in samples)
            maximum = float(1 << (8 * width - 1))
        squared_sum = sum(value * value for value in centered)
        rms = math.sqrt(squared_sum / max(1, len(samples))) / maximum
        return {"duration_seconds": round(duration, 3), "rms": round(rms, 6)}

    def transcribe(self, audio_path: Path) -> dict[str, object]:
        audio_path = audio_path.expanduser().resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        quality = self._wav_quality(audio_path)
        if quality is not None and quality["rms"] < self.minimum_wav_rms:
            return {
                "schema_version": "1.0",
                "module": "stt",
                "success": True,
                "model": self.model_name,
                "device": self.requested_device,
                "transcript": "",
                "abstained": True,
                "reason": "audio_too_quiet",
                "audio_quality": quality,
                "latency_ms": 0.0,
            }
        started = time.perf_counter()
        speech_pipeline = self._get_pipeline()
        with accelerator_guard(self.device, self._torch):
            result = speech_pipeline(
                str(audio_path), generate_kwargs={"language": "vi"}
            )
        transcript = str(result.get("text", "")).strip()
        normalized_transcript = unicodedata.normalize("NFD", transcript.casefold())
        ascii_transcript = normalized_transcript.encode("ascii", "ignore").decode(
            "ascii"
        )
        plain_transcript = " ".join(
            re.sub(r"[^a-z0-9]+", " ", ascii_transcript).split()
        )
        likely_silence_hallucination = bool(
            quality is not None
            and quality["rms"] < 0.035
            and any(
                phrase in plain_transcript
                for phrase in (
                    "hay subscribe",
                    "dang ky kenh",
                    "cam on cac ban da xem",
                    "khong bo lo nhung video",
                )
            )
        )
        return {
            "schema_version": "1.0",
            "module": "stt",
            "success": True,
            "model": self.model_name,
            "device": self.device,
            "transcript": "" if likely_silence_hallucination else transcript,
            "abstained": likely_silence_hallucination,
            "reason": (
                "likely_silence_hallucination"
                if likely_silence_hallucination
                else None
            ),
            "audio_quality": quality,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }


@dataclass(frozen=True, slots=True)
class AVFoundationAudioDevice:
    """One audio input reported by FFmpeg's AVFoundation backend."""

    index: str
    name: str


_AVFOUNDATION_DEVICE_LINE = re.compile(r"\]\s+\[(\d+)\]\s+(.+?)\s*$")
_VIRTUAL_AUDIO_TERMS = (
    "microsoft teams",
    "blackhole",
    "soundflower",
    "virtual",
)


def list_avfoundation_audio_devices(
    executable: str,
    *,
    timeout_seconds: float = 5.0,
) -> tuple[AVFoundationAudioDevice, ...]:
    """Return macOS audio inputs without relying on their unstable indexes."""
    try:
        completed = subprocess.run(
            [
                executable,
                "-nostdin",
                "-hide_banner",
                "-f",
                "avfoundation",
                "-list_devices",
                "true",
                "-i",
                "",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Hết thời gian dò danh sách microphone AVFoundation") from exc

    devices: list[AVFoundationAudioDevice] = []
    reading_audio = False
    for line in f"{completed.stderr}\n{completed.stdout}".splitlines():
        if "AVFoundation audio devices:" in line:
            reading_audio = True
            continue
        if not reading_audio:
            continue
        match = _AVFOUNDATION_DEVICE_LINE.search(line)
        if match:
            devices.append(
                AVFoundationAudioDevice(index=match.group(1), name=match.group(2))
            )
    if not devices:
        raise RuntimeError(
            "Không tìm thấy microphone AVFoundation. Kiểm tra quyền Microphone của Terminal/Python."
        )
    return tuple(devices)


def _automatic_microphone_rank(device: AVFoundationAudioDevice) -> tuple[int, int]:
    name = device.name.casefold()
    is_microphone = "microphone" in name or " mic" in name
    is_builtin = ("macbook" in name or "built-in" in name) and is_microphone
    is_virtual = any(term in name for term in _VIRTUAL_AUDIO_TERMS)
    if is_builtin:
        priority = 0
    elif is_microphone and not is_virtual:
        priority = 1
    elif not is_virtual:
        priority = 2
    else:
        priority = 3
    return priority, int(device.index)


class FFmpegMicrophoneRecorder:
    """Record a short push-to-talk WAV from a macOS AVFoundation audio device."""

    def __init__(self, device: str = "auto", duration_seconds: float = 4.0) -> None:
        if platform.system() != "Darwin":
            raise RuntimeError("Thu microphone hiện chỉ hỗ trợ macOS")
        if duration_seconds <= 0:
            raise ValueError("Thời lượng thu microphone phải dương")
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("Thiếu ffmpeg. Cài bằng: brew install ffmpeg")
        self.executable = executable
        self.device = str(device).strip()
        if not self.device:
            raise ValueError("Microphone không được để trống")
        self.duration_seconds = duration_seconds
        self.selected_device: AVFoundationAudioDevice | None = None

    def _resolve_device(self) -> AVFoundationAudioDevice:
        requested = self.device
        if requested.isdecimal():
            return AVFoundationAudioDevice(index=requested, name=f"device {requested}")

        devices = list_avfoundation_audio_devices(self.executable)
        if requested.casefold() == "auto":
            return min(devices, key=_automatic_microphone_rank)

        exact = next(
            (item for item in devices if item.name.casefold() == requested.casefold()),
            None,
        )
        if exact is not None:
            return exact
        available = ", ".join(f"{item.index}: {item.name}" for item in devices)
        raise RuntimeError(
            f"Không tìm thấy microphone '{requested}'. Thiết bị hiện có: {available}"
        )

    def record(self, output_path: Path) -> Path:
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        selected = self._resolve_device()
        self.selected_device = selected
        command = [
            self.executable,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "avfoundation",
            "-i",
            f":{selected.index}",
            "-t",
            str(self.duration_seconds),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-y",
            str(output_path),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.duration_seconds + 8.0,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Microphone '{selected.name}' không trả dữ liệu trong thời gian cho phép"
            ) from exc
        if completed.returncode != 0 or not output_path.is_file():
            detail = completed.stderr.strip().splitlines()
            message = detail[-1] if detail else "không rõ lỗi"
            raise RuntimeError(
                f"Không thu được microphone '{selected.name}' (index {selected.index}). "
                "Kiểm tra quyền Microphone: "
                + message
            )
        return output_path
