from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from secondeye.multimodal._model_loading import _from_pretrained_offline_first
from secondeye.multimodal.depth import DepthAnythingEstimator
from secondeye.multimodal.ocr import AppleVisionOcrReader, PaddleOcrReader
from secondeye.multimodal.questions import normalize_visual_question
from secondeye.multimodal.vqa import _is_uncertain_answer
from secondeye.system.demo import copy_frame_for_display
from secondeye.system.pipeline import SecondEyeSystem


def _quality_image() -> np.ndarray:
    pattern = (np.indices((300, 300)).sum(axis=0) % 2 * 255).astype(np.uint8)
    return np.repeat(pattern[:, :, None], 3, axis=2)


@pytest.mark.parametrize(
    ("question", "intent", "model_question", "target"),
    [
        (
            "Có bao nhiêu người trong bức ảnh?",
            "grounded_count",
            "How many people are visible?",
            "person",
        ),
        (
            "Có gì trước mặt tôi?",
            "grounded_scene",
            "What objects are directly in front of me?",
            None,
        ),
        (
            "Người này đang làm gì?",
            "model",
            "What is the person doing?",
            "person",
        ),
        (
            "Người này mặc gì?",
            "model",
            "What is the person wearing?",
            "person",
        ),
        (
            "Chiếc ghế màu gì?",
            "model",
            "What color is the chair?",
            "chair",
        ),
    ],
)
def test_vietnamese_questions_are_routed_before_blip(
    question, intent, model_question, target
):
    result = normalize_visual_question(question)

    assert result.language == "vi"
    assert result.intent == intent
    assert result.model_question == model_question
    assert result.target_label == target


def test_english_count_question_also_uses_grounded_detection():
    result = normalize_visual_question("How many people are visible?")

    assert result.language == "en"
    assert result.intent == "grounded_count"
    assert result.target_label == "person"


def test_unknown_vietnamese_question_is_not_sent_unchanged_to_english_blip():
    result = normalize_visual_question("Bạn nghĩ gì về căn phòng này?")

    assert result.intent == "unsupported"
    assert result.model_question is None


class _NeverCalledVqa:
    def ask_bgr(self, image, question):
        raise AssertionError("grounded queries must not call BLIP")


class _Detector:
    def predict_bgr(self, image):
        return {"detections": []}


def test_vietnamese_count_uses_grounded_detections_instead_of_blip():
    system = SecondEyeSystem(detector=_Detector(), vqa=_NeverCalledVqa())
    detection = {
        "detections": [
            {"label": "person", "direction": "center", "confidence": 0.91},
            {"label": "person", "direction": "left", "confidence": 0.83},
            {"label": "person", "direction": "right", "confidence": 0.31},
        ]
    }

    result = system.ask(
        _quality_image(),
        "Có bao nhiêu người trong bức ảnh?",
        detection_result=detection,
    )

    assert result["module"] == "grounded_visual_query"
    assert result["answer"] == "Tôi nhận diện được hai người trong ảnh."
    assert result["discarded_low_confidence"] == 1
    assert result["abstained"] is False


def test_default_front_query_describes_only_grounded_center_objects():
    system = SecondEyeSystem(detector=_Detector(), vqa=_NeverCalledVqa())
    detection = {
        "detections": [
            {
                "label": "chair",
                "direction": "center",
                "depth_zone": "near",
                "confidence": 0.9,
            },
            {
                "label": "person",
                "direction": "left",
                "depth_zone": "near",
                "confidence": 0.95,
            },
        ]
    }

    result = system.ask(
        _quality_image(),
        "What objects are directly in front of me?",
        detection_result=detection,
    )

    assert result["answer"] == "Tôi thấy ghế ở gần phía trước."
    assert "người" not in result["answer"]
    assert result["source"] == "pretrained_detection"


def test_scene_filters_low_confidence_and_orders_near_center_first():
    system = SecondEyeSystem(detector=_Detector())

    result = system.describe_scene(
        detection_result={
            "detections": [
                {
                    "label": "bottle",
                    "direction": "right",
                    "depth_zone": "far",
                    "confidence": 0.8,
                },
                {
                    "label": "person",
                    "direction": "center",
                    "depth_zone": "near",
                    "confidence": 0.9,
                },
                {
                    "label": "refrigerator",
                    "direction": "right",
                    "confidence": 0.36,
                },
            ]
        }
    )

    assert result["description"].startswith("Tôi thấy người ở gần phía trước")
    assert "chai ở xa bên phải" in result["description"]
    assert "tủ lạnh" not in result["description"]
    assert result["discarded_low_confidence"] == 1


class _OcrEngine:
    def predict(self, image):
        return [
            {
                "res": {
                    "rec_texts": ["VĂN BẢN RÕ", "chuỗi nhiễu", "MỘT KÝ TỰ"],
                    "rec_scores": [0.96, 0.41, 0.74],
                    "rec_boxes": [[0, 0, 20, 10], [0, 12, 20, 22], [0, 24, 20, 34]],
                }
            }
        ]


def test_ocr_discards_each_low_confidence_line_not_only_low_mean():
    reader = object.__new__(PaddleOcrReader)
    reader.language = "vi"
    reader.minimum_line_confidence = 0.75
    reader.engine = _OcrEngine()

    result = reader.read_bgr(np.zeros((50, 50, 3), dtype=np.uint8))

    assert result["transcript"] == "VĂN BẢN RÕ"
    assert result["raw_transcript"] == "VĂN BẢN RÕ chuỗi nhiễu MỘT KÝ TỰ"
    assert [line["text"] for line in result["discarded_lines"]] == [
        "chuỗi nhiễu",
        "MỘT KÝ TỰ",
    ]


def test_display_overlay_copy_cannot_contaminate_semantic_frame():
    raw = np.zeros((20, 20, 3), dtype=np.uint8)
    display = copy_frame_for_display(raw)

    display[:] = 255

    assert np.all(raw == 0)
    assert np.all(display == 255)


def test_apple_vision_adapter_preserves_vietnamese_and_converts_boxes(
    monkeypatch, tmp_path
):
    helper = tmp_path / "secondeye-vision-ocr"
    helper.write_text("probe", encoding="utf-8")
    helper.chmod(0o755)
    output = (
        '{"latency_ms":612.2,"lines":[{"text":"TẦNG 2 - PHÒNG 205",'
        '"confidence":1,"box":[0.1,0.5,0.8,0.7]}]}'
    )
    monkeypatch.setattr("secondeye.multimodal.ocr.platform.system", lambda: "Darwin")
    monkeypatch.setattr(
        "secondeye.multimodal.ocr.subprocess.run",
        lambda *args, **kwargs: type(
            "Completed",
            (),
            {"returncode": 0, "stdout": output, "stderr": ""},
        )(),
    )
    reader = AppleVisionOcrReader(executable=helper)

    result = reader.read_bgr(np.zeros((100, 200, 3), dtype=np.uint8))

    assert result["engine"] == "Apple Vision"
    assert result["transcript"] == "TẦNG 2 - PHÒNG 205"
    assert result["lines"][0]["box"] == [20.0, 30.0, 160.0, 50.0]


@pytest.mark.parametrize(
    "answer",
    ["I don't know", "i don ' t know", "unknown", "Cannot tell", "not sure"],
)
def test_vqa_uncertain_answers_always_abstain(answer):
    assert _is_uncertain_answer(answer)


def test_model_loader_uses_local_cache_without_network_probe():
    class Factory:
        calls = []

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            cls.calls.append((name, kwargs))
            return "cached"

    assert _from_pretrained_offline_first(Factory, "model") == "cached"
    assert Factory.calls == [("model", {"local_files_only": True})]


def test_model_loader_falls_back_to_download_only_when_cache_is_missing():
    class Factory:
        calls = []

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            cls.calls.append((name, kwargs))
            if kwargs.get("local_files_only"):
                raise OSError("not cached")
            return "downloaded"

    assert _from_pretrained_offline_first(Factory, "model") == "downloaded"
    assert Factory.calls == [
        ("model", {"local_files_only": True}),
        ("model", {}),
    ]


def test_depth_loader_uses_local_cache_for_processor_and_model(monkeypatch):
    calls = []

    class ProcessorFactory:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls.append(("processor", name, kwargs))
            return object()

    class Model:
        def to(self, device):
            return self

        def eval(self):
            return None

    class ModelFactory:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls.append(("model", name, kwargs))
            return Model()

    transformers = ModuleType("transformers")
    transformers.AutoImageProcessor = ProcessorFactory
    transformers.AutoModelForDepthEstimation = ModelFactory
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)

    DepthAnythingEstimator(model_name="depth-model", device="cpu")

    assert calls == [
        ("processor", "depth-model", {"local_files_only": True}),
        ("model", "depth-model", {"local_files_only": True}),
    ]
