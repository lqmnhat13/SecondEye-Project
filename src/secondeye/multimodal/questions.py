"""Deterministic Vietnamese question routing for the English BLIP model."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass


_VIETNAMESE_MARKERS = (
    "anh",
    "bao nhieu",
    "buc anh",
    "co gi",
    "dang lam gi",
    "day la gi",
    "mau gi",
    "mac gi",
    "nguoi",
    "phia truoc",
    "truoc mat",
)

_OBJECT_TERMS = {
    "vat the": ("object", "objects"),
    "do vat": ("object", "objects"),
    "nguoi": ("person", "people"),
    "ghe": ("chair", "chairs"),
    "chai": ("bottle", "bottles"),
    "ba lo": ("backpack", "backpacks"),
    "tui xach": ("handbag", "handbags"),
    "va li": ("suitcase", "suitcases"),
    "giuong": ("bed", "beds"),
    "ban": ("table", "tables"),
    "may tinh": ("laptop", "laptops"),
    "laptop": ("laptop", "laptops"),
    "ti vi": ("tv", "televisions"),
    "tu lanh": ("refrigerator", "refrigerators"),
}


def plain_vietnamese(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.casefold())
    without_marks = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    return " ".join(re.sub(r"[^a-z0-9]+", " ", without_marks).split())


@dataclass(frozen=True, slots=True)
class VisualQuestion:
    original: str
    model_question: str | None
    language: str
    method: str
    intent: str
    target_label: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


def _object_term(plain: str) -> tuple[str, str] | None:
    for source in sorted(_OBJECT_TERMS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(source)}\b", plain):
            return _OBJECT_TERMS[source]
    return None


def normalize_visual_question(question: str) -> VisualQuestion:
    """Map supported Vietnamese visual questions to grounded/English intents.

    BLIP VQA base is English-first. Unknown Vietnamese phrasing is deliberately
    not sent unchanged to the model because that produced confident nonsense.
    """

    original = " ".join(question.strip().split())
    if not original:
        raise ValueError("question không được rỗng")
    plain = plain_vietnamese(original)
    has_vietnamese_marks = any(ord(character) > 127 for character in original)
    looks_vietnamese = has_vietnamese_marks or any(
        marker in plain for marker in _VIETNAMESE_MARKERS
    )
    if not looks_vietnamese:
        lowered = original.casefold()
        english_counts = {
            "people": "person",
            "persons": "person",
            "chairs": "chair",
            "bottles": "bottle",
            "backpacks": "backpack",
            "handbags": "handbag",
            "suitcases": "suitcase",
            "beds": "bed",
            "tables": "table",
            "laptops": "laptop",
            "televisions": "tv",
            "refrigerators": "refrigerator",
            "objects": "object",
        }
        count_match = re.search(r"\bhow many ([a-z ]+?)(?: are| do|\?|$)", lowered)
        counted_label = (
            english_counts.get(count_match.group(1).strip()) if count_match else None
        )
        if counted_label is not None:
            intent = "grounded_count"
        elif (
            "what objects" in lowered
            and ("in front" in lowered or "visible" in lowered)
        ):
            intent = "grounded_scene"
        else:
            intent = "model"
        return VisualQuestion(
            original=original,
            model_question=original,
            language="en",
            method="unchanged_english",
            intent=intent,
            target_label=counted_label,
        )

    target = _object_term(plain)
    if "bao nhieu" in plain and target is not None:
        label, plural = target
        return VisualQuestion(
            original=original,
            model_question=f"How many {plural} are visible?",
            language="vi",
            method="vi_template_v1",
            intent="grounded_count",
            target_label=label,
        )
    if any(
        marker in plain
        for marker in (
            "co gi truoc mat",
            "phia truoc co gi",
            "co gi phia truoc",
            "trong anh co gi",
            "buc anh co gi",
            "anh co gi",
            "co gi trong anh",
        )
    ):
        return VisualQuestion(
            original=original,
            model_question="What objects are directly in front of me?",
            language="vi",
            method="vi_template_v1",
            intent="grounded_scene",
        )
    if "mau gi" in plain and target is not None:
        label, _ = target
        return VisualQuestion(
            original=original,
            model_question=f"What color is the {label}?",
            language="vi",
            method="vi_template_v1",
            intent="model",
            target_label=label,
        )
    if "nguoi" in plain and any(
        marker in plain for marker in ("dang lam gi", "lam gi")
    ):
        return VisualQuestion(
            original=original,
            model_question="What is the person doing?",
            language="vi",
            method="vi_template_v1",
            intent="model",
            target_label="person",
        )
    if "nguoi" in plain and any(
        marker in plain for marker in ("mac gi", "dang mac")
    ):
        return VisualQuestion(
            original=original,
            model_question="What is the person wearing?",
            language="vi",
            method="vi_template_v1",
            intent="model",
            target_label="person",
        )
    if any(marker in plain for marker in ("day la gi", "vat gi day")):
        return VisualQuestion(
            original=original,
            model_question="What is this?",
            language="vi",
            method="vi_template_v1",
            intent="model",
        )
    return VisualQuestion(
        original=original,
        model_question=None,
        language="vi",
        method="unsupported_vietnamese",
        intent="unsupported",
    )
