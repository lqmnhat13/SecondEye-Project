import json
import threading
import time
import wave

import numpy as np
import pytest

from secondeye.multimodal.speech import WhisperSpeechToText
from secondeye.accelerator import accelerator_guard
from secondeye.system.audio import PriorityAudioManager
from secondeye.system.demo import SemanticCommand, SemanticWorker
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

    def ask(self, frame, question):
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
