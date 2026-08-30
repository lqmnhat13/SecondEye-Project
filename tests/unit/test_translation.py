from contextlib import nullcontext

import pytest

from secondeye.multimodal.translation import (
    PretrainedEnglishVietnameseTranslator,
    translate_visual_answer,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("yellow", "vàng"),
        ("a yellow bus", "một chiếc xe buýt màu vàng"),
        (
            "a yellow bus parked on the street",
            "một chiếc xe buýt màu vàng đỗ trên đường",
        ),
        ("many people", "nhiều người"),
        (
            "two people standing near a bus",
            "hai người đang đứng gần một chiếc xe buýt",
        ),
        ("the door is open", "cánh cửa đang mở"),
        ("a black shirt", "một chiếc áo sơ mi màu đen"),
        (
            "three chairs in front of a table",
            "ba chiếc ghế phía trước một cái bàn",
        ),
        ("a person on the left", "một người bên trái"),
        (
            "the bus is blue and white",
            "chiếc xe buýt màu xanh dương và trắng",
        ),
        ("people are standing near a bus", "người đang đứng gần một chiếc xe buýt"),
        ("a red umbrella", "một chiếc ô màu đỏ"),
        ("two black shirts", "hai chiếc áo sơ mi màu đen"),
        ("a green traffic light", "một chiếc đèn giao thông màu xanh lá"),
        ("the bus is parked on the street", "chiếc xe buýt đỗ trên đường"),
    ],
)
def test_visual_answer_regressions(source, expected):
    assert translate_visual_answer(source) == expected


@pytest.mark.parametrize(
    ("english", "vietnamese"),
    [
        ("black", "đen"),
        ("blue", "xanh dương"),
        ("brown", "nâu"),
        ("gray", "xám"),
        ("green", "xanh lá"),
        ("orange", "cam"),
        ("pink", "hồng"),
        ("purple", "tím"),
        ("red", "đỏ"),
        ("tan", "nâu nhạt"),
        ("white", "trắng"),
        ("yellow", "vàng"),
    ],
)
def test_every_supported_color_keeps_its_meaning(english, vietnamese):
    result = translate_visual_answer(f"a {english} bus")

    assert result == f"một chiếc xe buýt màu {vietnamese}"


@pytest.mark.parametrize(
    ("english", "vietnamese"),
    [
        ("one", "một"),
        ("two", "hai"),
        ("three", "ba"),
        ("four", "bốn"),
        ("five", "năm"),
        ("six", "sáu"),
        ("seven", "bảy"),
        ("eight", "tám"),
        ("nine", "chín"),
        ("ten", "mười"),
    ],
)
def test_every_supported_count_keeps_its_value(english, vietnamese):
    result = translate_visual_answer(f"{english} chairs")

    assert result == f"{vietnamese} chiếc ghế"


def test_visual_translation_normalizes_case_whitespace_and_punctuation():
    assert translate_visual_answer("  A   YELLOW BUS! ") == (
        "một chiếc xe buýt màu vàng"
    )


def test_safe_translation_does_not_load_marian(monkeypatch):
    translator = PretrainedEnglishVietnameseTranslator()
    monkeypatch.setattr(
        translator,
        "_load",
        lambda: pytest.fail("safe visual translation must not load Marian"),
    )

    result = translator.translate("a black shirt")

    assert result["translation"] == "một chiếc áo sơ mi màu đen"
    assert result["method"] == "visual_lexicon_v1"
    assert result["quality_assured"] is True
    assert result["model"] is None


def test_unknown_answer_fails_closed_without_loading_model(monkeypatch):
    translator = PretrainedEnglishVietnameseTranslator()
    monkeypatch.setattr(
        translator,
        "_load",
        lambda: pytest.fail("strict mode must abstain before loading Marian"),
    )

    with pytest.raises(RuntimeError, match="ngoài miền dịch thị giác an toàn"):
        translator.translate("an unusual abstract sculpture")


class _FakeTensor:
    def to(self, device):
        return self


class _FakeTokenizer:
    def __init__(self):
        self.seen_text = None

    def __call__(self, text, **kwargs):
        self.seen_text = text
        return {"input_ids": _FakeTensor()}

    def decode(self, value, **kwargs):
        return "một tác phẩm điêu khắc trừu tượng"


class _FakeModel:
    def generate(self, **kwargs):
        return [[1, 2, 3]]


class _FakeTorch:
    @staticmethod
    def inference_mode():
        return nullcontext()


def test_opt_in_model_fallback_receives_raw_source_without_prefix(monkeypatch):
    tokenizer = _FakeTokenizer()
    translator = PretrainedEnglishVietnameseTranslator(
        device="cpu", allow_unverified_model_fallback=True
    )
    translator.device = "cpu"
    monkeypatch.setattr(
        translator,
        "_load",
        lambda: (_FakeTorch(), tokenizer, _FakeModel()),
    )

    result = translator.translate("an unusual abstract sculpture")

    assert tokenizer.seen_text == "an unusual abstract sculpture"
    assert result["translation"] == "một tác phẩm điêu khắc trừu tượng"
    assert result["method"] == "unverified_marian_fallback"
    assert result["quality_assured"] is False


@pytest.mark.parametrize(
    "source",
    [
        "",
        "twenty-seven fluorescent objects beyond the unusually shaped sculpture",
        "đây đã là tiếng Việt",
    ],
)
def test_visual_translator_rejects_unsupported_input(source):
    assert translate_visual_answer(source) is None
