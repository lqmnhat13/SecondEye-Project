import json
import threading
import time
import wave

import numpy as np
import pytest

import secondeye.multimodal.speech as speech_module
from secondeye.multimodal.speech import (
    FFmpegMicrophoneRecorder,
    WhisperSpeechToText,
    list_avfoundation_audio_devices,
)
from secondeye.accelerator import accelerator_guard
from secondeye.system.audio import PriorityAudioManager
from secondeye.system.demo import (
    SemanticCommand,
    SemanticWorker,
    voice_intent_from_transcript,
)
from secondeye.system.orchestrator import AlertPriority
from secondeye.system.overlay import UnicodeTextRenderer
from secondeye.system.session import SessionLogger


class _AudioBackend:
    def __init__(self):
        self.spoken = []
        self.stop_count = 0

    def speak(self, text, *, interrupt=False):
        self.spoken.append((text, interrupt))

    def wait(self):
        time.sleep(0.01)

    def stop(self):
        self.stop_count += 1


def _wait_until(predicate, timeout=0.5):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert predicate()


def test_priority_audio_manager_serializes_and_repeats():
    backend = _AudioBackend()
    audio = PriorityAudioManager(backend)
    try:
        assert audio.submit("Đọc văn bản", priority=AlertPriority.SEMANTIC)
        _wait_until(lambda: len(backend.spoken) == 1)
        assert audio.submit("Cảnh báo", priority=AlertPriority.OBSTACLE)
        _wait_until(lambda: len(backend.spoken) == 2)
        assert audio.repeat()
        _wait_until(lambda: len(backend.spoken) == 3)
        assert backend.spoken[-1][0] == "Cảnh báo"
    finally:
        audio.close()


def test_priority_audio_manager_survives_a_backend_failure():
    class FailsOnceBackend(_AudioBackend):
        def __init__(self):
            super().__init__()
            self.fail_once = True

        def speak(self, text, *, interrupt=False):
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("temporary TTS failure")
            super().speak(text, interrupt=interrupt)

    backend = FailsOnceBackend()
    audio = PriorityAudioManager(backend)
    try:
        assert audio.submit("lỗi tạm thời")
        _wait_until(lambda: audio.last_error is not None)
        assert audio.submit("đã phục hồi")
        _wait_until(lambda: len(backend.spoken) == 1)
        assert backend.spoken[0][0] == "đã phục hồi"
    finally:
        audio.close()


@pytest.mark.parametrize(
    ("transcript", "intent"),
    [
        ("Hãy dừng lại", "stop"),
        ("Người này đang sử dụng điện thoại", "vqa"),
        ("Hải lọc chữ trong ảnh", "ocr"),
        ("Hãy mô tà không cạnh", "scene"),
        ("Lặp lại", "repeat"),
    ],
)
def test_voice_command_intent_tolerates_common_whisper_errors(transcript, intent):
    assert voice_intent_from_transcript(transcript) == intent


def test_session_logger_writes_reproducible_jsonl(tmp_path):
    logger = SessionLogger(tmp_path / "session.jsonl")
    logger.log("demo", {"latency_ms": 12.3})

    record = json.loads(logger.path.read_text(encoding="utf-8"))

    assert record["session_id"] == logger.session_id
    assert record["event"] == "demo"
    assert record["payload"]["latency_ms"] == 12.3


class _Recorder:
    def record(self, path):
        path.write_bytes(b"wav")
        return path


class _Transcriber:
    def transcribe(self, path):
        assert path.is_file()
        return {"transcript": "Có gì trước mặt tôi?"}


class _SemanticSystem:
    def __init__(self):
        self.questions = []

    def ask(self, frame, question, *, detection_result=None):
        assert detection_result is None
        self.questions.append(question)
        return {"answer": "ghế"}

    def announce(self, text, priority):
        raise AssertionError(text)


def test_push_to_talk_routes_transcript_to_vqa_without_blocking_submit(tmp_path):
    system = _SemanticSystem()
    worker = SemanticWorker(
        system,
        SessionLogger(tmp_path / "voice.jsonl"),
        default_question="unused",
        recorder_factory=_Recorder,
        transcriber_factory=_Transcriber,
    )
    try:
        assert worker.submit(SemanticCommand("microphone", frame=object()))
        _wait_until(lambda: worker.snapshot()[2] is not None)
        result = worker.snapshot()[2]
        assert result["intent"] == "vqa"
        assert system.questions == ["Có gì trước mặt tôi?"]
    finally:
        worker.close()


def test_semantic_worker_rejects_a_second_command_without_submit_race(tmp_path):
    started = threading.Event()
    release = threading.Event()

    class BlockingSystem:
        def read_text(self, frame):
            started.set()
            release.wait(timeout=0.5)
            return {"transcript": "xong"}

        def announce(self, text, priority):
            raise AssertionError(text)

    worker = SemanticWorker(
        BlockingSystem(),
        SessionLogger(tmp_path / "race.jsonl"),
        default_question="unused",
    )
    try:
        assert worker.submit(SemanticCommand("ocr", frame=object())) is True
        assert worker.submit(SemanticCommand("ocr", frame=object())) is False
        assert started.wait(timeout=0.5)
        release.set()
        _wait_until(lambda: worker.snapshot()[2] is not None)
    finally:
        release.set()
        worker.close()


def test_semantic_worker_routes_ocr_burst_to_multiframe_reader(tmp_path):
    class BurstSystem:
        def __init__(self):
            self.received = None

        def read_text_frames(self, frames):
            self.received = tuple(frames)
            return {"transcript": "ổn định"}

        def announce(self, text, priority):
            raise AssertionError(text)

    system = BurstSystem()
    worker = SemanticWorker(
        system,
        SessionLogger(tmp_path / "ocr-burst.jsonl"),
        default_question="unused",
    )
    try:
        burst = (object(), object(), object())
        assert worker.submit(SemanticCommand("ocr", frame=burst[-1], frames=burst))
        _wait_until(lambda: worker.snapshot()[2] is not None)
        assert system.received == burst
    finally:
        worker.close()


def test_stt_abstains_on_quiet_wav_before_loading_whisper(tmp_path):
    audio = tmp_path / "quiet.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 16000)

    result = WhisperSpeechToText().transcribe(audio)

    assert result["abstained"] is True
    assert result["reason"] == "audio_too_quiet"
    assert result["transcript"] == ""


_AVFOUNDATION_LISTING = """
[AVFoundation indev @ 0x1] AVFoundation video devices:
[AVFoundation indev @ 0x1] [0] FaceTime HD Camera
[AVFoundation indev @ 0x1] AVFoundation audio devices:
[AVFoundation indev @ 0x1] [0] nắng Microphone
[AVFoundation indev @ 0x1] [1] AirPods của lê
[AVFoundation indev @ 0x1] [2] MacBook Pro Microphone
[AVFoundation indev @ 0x1] [3] Microsoft Teams Audio
"""


def test_avfoundation_device_listing_parses_audio_only(monkeypatch):
    monkeypatch.setattr(
        speech_module.subprocess,
        "run",
        lambda *args, **kwargs: type(
            "Completed", (), {"returncode": 1, "stderr": _AVFOUNDATION_LISTING, "stdout": ""}
        )(),
    )

    devices = list_avfoundation_audio_devices("ffmpeg")

    assert [(item.index, item.name) for item in devices] == [
        ("0", "nắng Microphone"),
        ("1", "AirPods của lê"),
        ("2", "MacBook Pro Microphone"),
        ("3", "Microsoft Teams Audio"),
    ]


def test_auto_microphone_resolves_builtin_device_by_name(
    monkeypatch, tmp_path
):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if "-list_devices" in command:
            return type(
                "Completed",
                (),
                {"returncode": 1, "stderr": _AVFOUNDATION_LISTING, "stdout": ""},
            )()
        (tmp_path / "recording.wav").write_bytes(b"RIFF")
        return type(
            "Completed", (), {"returncode": 0, "stderr": "", "stdout": ""}
        )()

    monkeypatch.setattr(speech_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(speech_module.shutil, "which", lambda name: "/opt/ffmpeg")
    monkeypatch.setattr(speech_module.subprocess, "run", fake_run)
    recorder = FFmpegMicrophoneRecorder(duration_seconds=1.0)

    assert recorder.record(tmp_path / "recording.wav").is_file()
    assert recorder.selected_device is not None
    assert recorder.selected_device.index == "2"
    assert recorder.selected_device.name == "MacBook Pro Microphone"
    assert commands[-1][commands[-1].index("-i") + 1] == ":2"


def test_microphone_can_be_selected_by_exact_name(monkeypatch):
    monkeypatch.setattr(speech_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(speech_module.shutil, "which", lambda name: "/opt/ffmpeg")
    monkeypatch.setattr(
        speech_module.subprocess,
        "run",
        lambda *args, **kwargs: type(
            "Completed", (), {"returncode": 1, "stderr": _AVFOUNDATION_LISTING, "stdout": ""}
        )(),
    )
    recorder = FFmpegMicrophoneRecorder(device="nắng Microphone")

    selected = recorder._resolve_device()

    assert selected.index == "0"
    assert selected.name == "nắng Microphone"


def test_explicit_microphone_index_remains_supported(monkeypatch, tmp_path):
    def fake_run(command, **kwargs):
        assert "-list_devices" not in command
        (tmp_path / "explicit.wav").write_bytes(b"RIFF")
        return type(
            "Completed", (), {"returncode": 0, "stderr": "", "stdout": ""}
        )()

    monkeypatch.setattr(speech_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(speech_module.shutil, "which", lambda name: "/opt/ffmpeg")
    monkeypatch.setattr(speech_module.subprocess, "run", fake_run)

    recorder = FFmpegMicrophoneRecorder(device="7", duration_seconds=1.0)

    assert recorder.record(tmp_path / "explicit.wav").is_file()
    assert recorder.selected_device is not None
    assert recorder.selected_device.index == "7"


def test_microphone_record_timeout_is_reported(monkeypatch, tmp_path):
    def timeout(*args, **kwargs):
        raise speech_module.subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(speech_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(speech_module.shutil, "which", lambda name: "/opt/ffmpeg")
    monkeypatch.setattr(speech_module.subprocess, "run", timeout)
    recorder = FFmpegMicrophoneRecorder(device="2", duration_seconds=1.0)

    with pytest.raises(RuntimeError, match="không trả dữ liệu"):
        recorder.record(tmp_path / "timeout.wav")


def test_microphone_discovery_timeout_is_reported(monkeypatch):
    def timeout(*args, **kwargs):
        raise speech_module.subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(speech_module.subprocess, "run", timeout)

    with pytest.raises(RuntimeError, match="dò danh sách microphone"):
        list_avfoundation_audio_devices("ffmpeg", timeout_seconds=0.01)


def test_unicode_overlay_renders_vietnamese_without_question_mark_substitution():
    pytest.importorskip("PIL")
    image = np.zeros((80, 360, 3), dtype=np.uint8)
    renderer = UnicodeTextRenderer()

    returned = renderer.draw_bgr(
        image,
        [("Sẵn sàng — nhận diện tiếng Việt", (8, 8), (255, 255, 255), 22)],
    )

    assert returned is image
    assert np.count_nonzero(image) > 500
    assert renderer.font_path.lower().endswith((".ttf", ".otf", ".ttc"))


def test_mps_guard_serializes_accelerator_work_across_threads():
    first_entered = threading.Event()
    release_first = threading.Event()
    second_attempting = threading.Event()
    second_entered = threading.Event()

    def first_worker():
        with accelerator_guard("mps"):
            first_entered.set()
            assert release_first.wait(1.0)

    def second_worker():
        assert first_entered.wait(1.0)
        second_attempting.set()
        with accelerator_guard("mps"):
            second_entered.set()

    first = threading.Thread(target=first_worker)
    second = threading.Thread(target=second_worker)
    first.start()
    second.start()
    try:
        assert first_entered.wait(1.0)
        assert second_attempting.wait(1.0)
        assert not second_entered.wait(0.1)
    finally:
        release_first.set()
        first.join(timeout=1.0)
        second.join(timeout=1.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert second_entered.is_set()
