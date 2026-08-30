"""Safe English-to-Vietnamese translation for short visual answers."""

from __future__ import annotations

import threading
import time
from collections import Counter
from typing import Any

from secondeye.accelerator import accelerator_guard


_COLORS = {
    "black": "đen",
    "blue": "xanh dương",
    "brown": "nâu",
    "gray": "xám",
    "green": "xanh lá",
    "grey": "xám",
    "orange": "cam",
    "pink": "hồng",
    "purple": "tím",
    "red": "đỏ",
    "tan": "nâu nhạt",
    "white": "trắng",
    "yellow": "vàng",
}

_QUANTITIES = {
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
    "many": "nhiều",
    "several": "vài",
}

# Values are (Vietnamese noun, classifier). Empty classifier means that the
# noun is naturally counted without one in short spoken answers.
_NOUNS = {
    "airplane": ("máy bay", "chiếc"),
    "apple": ("táo", "quả"),
    "backpack": ("ba lô", "chiếc"),
    "bed": ("giường", "chiếc"),
    "bench": ("ghế băng", "chiếc"),
    "bicycle": ("xe đạp", "chiếc"),
    "bird": ("chim", "con"),
    "boat": ("thuyền", "chiếc"),
    "book": ("sách", "quyển"),
    "bottle": ("chai", ""),
    "bowl": ("bát", "cái"),
    "broccoli": ("bông cải xanh", ""),
    "bus": ("xe buýt", "chiếc"),
    "cake": ("bánh ngọt", "chiếc"),
    "car": ("ô tô", "chiếc"),
    "carrot": ("cà rốt", "củ"),
    "cat": ("mèo", "con"),
    "cell phone": ("điện thoại", "chiếc"),
    "chair": ("ghế", "chiếc"),
    "clock": ("đồng hồ", "chiếc"),
    "coat": ("áo khoác dài", "chiếc"),
    "cow": ("bò", "con"),
    "cup": ("cốc", "cái"),
    "dining table": ("bàn ăn", "cái"),
    "door": ("cửa", "cánh"),
    "dog": ("chó", "con"),
    "donut": ("bánh vòng", "chiếc"),
    "dress": ("váy", "chiếc"),
    "elephant": ("voi", "con"),
    "fork": ("nĩa", "chiếc"),
    "giraffe": ("hươu cao cổ", "con"),
    "glass": ("ly", "cái"),
    "handbag": ("túi xách", "chiếc"),
    "hat": ("mũ", "chiếc"),
    "horse": ("ngựa", "con"),
    "jacket": ("áo khoác", "chiếc"),
    "jeans": ("quần jean", "chiếc"),
    "keyboard": ("bàn phím", "chiếc"),
    "knife": ("dao", "con"),
    "laptop": ("máy tính xách tay", "chiếc"),
    "man": ("người đàn ông", ""),
    "microwave": ("lò vi sóng", "chiếc"),
    "mouse": ("chuột máy tính", "con"),
    "motorcycle": ("xe máy", "chiếc"),
    "oven": ("lò nướng", "chiếc"),
    "pants": ("quần dài", "chiếc"),
    "person": ("người", ""),
    "pizza": ("bánh pizza", "chiếc"),
    "plate": ("đĩa", "cái"),
    "plant": ("chậu cây", ""),
    "potted plant": ("chậu cây", ""),
    "refrigerator": ("tủ lạnh", "chiếc"),
    "sandwich": ("bánh mì kẹp", "chiếc"),
    "scissors": ("kéo", "chiếc"),
    "sheep": ("cừu", "con"),
    "shirt": ("áo sơ mi", "chiếc"),
    "shoes": ("giày", "đôi"),
    "shorts": ("quần đùi", "chiếc"),
    "sign": ("biển báo", "tấm"),
    "sink": ("bồn rửa", "chiếc"),
    "spoon": ("thìa", "chiếc"),
    "sofa": ("ghế sofa", "chiếc"),
    "street": ("đường", ""),
    "suitcase": ("va li", "chiếc"),
    "table": ("bàn", "cái"),
    "toaster": ("máy nướng bánh mì", "chiếc"),
    "toilet": ("bồn cầu", "chiếc"),
    "traffic light": ("đèn giao thông", "chiếc"),
    "train": ("tàu hỏa", "chiếc"),
    "truck": ("xe tải", "chiếc"),
    "tv": ("ti vi", "chiếc"),
    "umbrella": ("ô", "chiếc"),
    "vase": ("bình hoa", "chiếc"),
    "window": ("cửa sổ", "cánh"),
    "woman": ("người phụ nữ", ""),
    "zebra": ("ngựa vằn", "con"),
}

_PLURAL_NOUNS = {
    "bicycles": "bicycle",
    "bottles": "bottle",
    "buses": "bus",
    "cars": "car",
    "cats": "cat",
    "chairs": "chair",
    "dogs": "dog",
    "people": "person",
    "shirts": "shirt",
    "tables": "table",
    "trucks": "truck",
    "windows": "window",
}

_PREDICATES = {
    "closed": "đang đóng",
    "empty": "trống",
    "full": "đầy",
    "large": "lớn",
    "open": "đang mở",
    "small": "nhỏ",
}

_ACTIONS = {
    "parked": "đỗ",
    "running": "đang chạy",
    "sitting": "đang ngồi",
    "standing": "đang đứng",
    "walking": "đang đi bộ",
}

_RELATIONS = (
    ("directly in front of ", "ngay phía trước "),
    ("in front of ", "phía trước "),
    ("next to ", "bên cạnh "),
    ("near ", "gần "),
    ("behind ", "phía sau "),
    ("under ", "bên dưới "),
    ("on ", "trên "),
    ("in ", "trong "),
)

_FIXED_LOCATIONS = {
    "in front": "phía trước",
    "in the center": "ở giữa",
    "on the left": "bên trái",
    "on the right": "bên phải",
}


def _normalized_english(text: str) -> str:
    return " ".join(text.strip().lower().rstrip(".!?").split())


def _translate_colors(text: str) -> str | None:
    parts = text.split(" and ")
    if not parts or any(part not in _COLORS for part in parts):
        return None
    return " và ".join(_COLORS[part] for part in parts)


def _translate_noun_phrase(text: str) -> str | None:
    words = text.split()
    if not words:
        return None
    article: str | None = None
    quantity: str | None = None
    if words[0] in {"a", "an", "the"}:
        article, words = words[0], words[1:]
    elif words[0] in _QUANTITIES:
        quantity, words = _QUANTITIES[words[0]], words[1:]
    if not words:
        return None

    noun: tuple[str, str] | None = None
    modifier_words = words
    for size in range(min(3, len(words)), 0, -1):
        raw_key = " ".join(words[-size:])
        noun_key = _PLURAL_NOUNS.get(raw_key, raw_key)
        if noun_key not in _NOUNS and noun_key.endswith("s"):
            singular = noun_key[:-1]
            noun_key = singular if singular in _NOUNS else noun_key
        noun = _NOUNS.get(noun_key)
        if noun is not None:
            modifier_words = words[:-size]
            break
    if noun is None:
        return None
    modifiers = " ".join(modifier_words)
    color = _translate_colors(modifiers) if modifiers else None
    if modifiers and color is None:
        return None

    translated_noun, classifier = noun
    if quantity is not None:
        prefix = f"{quantity} "
        if quantity not in {"nhiều", "vài"} and classifier:
            prefix += f"{classifier} "
    elif article in {"a", "an"}:
        prefix = "một " + (f"{classifier} " if classifier else "")
    elif article == "the" and classifier:
        prefix = f"{classifier} "
    else:
        prefix = ""
    suffix = f" màu {color}" if color else ""
    return f"{prefix}{translated_noun}{suffix}".strip()


def _translate_location(text: str) -> str | None:
    if text in _FIXED_LOCATIONS:
        return _FIXED_LOCATIONS[text]
    for source, target in _RELATIONS:
        if text.startswith(source):
            noun = _translate_noun_phrase(text[len(source) :])
            return None if noun is None else target + noun
    return None


def translate_visual_answer(text: str) -> str | None:
    """Safely translate a short, fully recognized visual-answer expression."""
    source = _normalized_english(text)
    if not source or len(source.split()) > 20 or not source.isascii():
        return None

    color = _translate_colors(source)
    if color is not None:
        return color

    copula = " is " if " is " in source else " are " if " are " in source else None
    if copula is not None:
        subject_text, predicate_text = source.split(copula, 1)
        subject = _translate_noun_phrase(subject_text)
        if subject is None:
            return None
        predicate = _translate_colors(predicate_text)
        if predicate is not None:
            return f"{subject} màu {predicate}"
        if predicate_text in _PREDICATES:
            return f"{subject} {_PREDICATES[predicate_text]}"
        if predicate_text in _ACTIONS:
            return f"{subject} {_ACTIONS[predicate_text]}"
        for action, translated_action in _ACTIONS.items():
            marker = f"{action} "
            if predicate_text.startswith(marker):
                location = _translate_location(predicate_text[len(marker) :])
                return (
                    None
                    if location is None
                    else f"{subject} {translated_action} {location}"
                )
        location = _translate_location(predicate_text)
        return None if location is None else f"{subject} {location}"

    for action, translated_action in _ACTIONS.items():
        marker = f" {action}"
        if marker not in source:
            continue
        subject_text, remainder = source.split(marker, 1)
        subject = _translate_noun_phrase(subject_text)
        if subject is None:
            return None
        remainder = remainder.strip()
        if not remainder:
            return f"{subject} {translated_action}"
        location = _translate_location(remainder)
        return None if location is None else f"{subject} {translated_action} {location}"

    for marker in (*_FIXED_LOCATIONS, *(item[0].strip() for item in _RELATIONS)):
        split_marker = f" {marker}"
        if split_marker not in source:
            continue
        subject_text, location_text = source.split(split_marker, 1)
        subject = _translate_noun_phrase(subject_text)
        location = _translate_location(marker + location_text)
        if subject is not None and location is not None:
            return f"{subject} {location}"
    return _translate_noun_phrase(source)


class PretrainedEnglishVietnameseTranslator:
    """Translate known visual answers and fail closed outside the safe domain."""

    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-en-vi",
        device: str = "auto",
        allow_unverified_model_fallback: bool = False,
    ) -> None:
        self.model_name = model_name
        self.requested_device = device
        self.allow_unverified_model_fallback = allow_unverified_model_fallback
        self.device: str | None = None
        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._lock = threading.Lock()

    def _load(self) -> tuple[Any, Any, Any]:
        with self._lock:
            if self._model is not None:
                return self._torch, self._tokenizer, self._model
            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    'Thiếu runtime dịch. Chạy: python -m pip install ".[multimodal]"'
                ) from exc
            device = self.requested_device
            if device == "auto":
                if torch.cuda.is_available():
                    device = "cuda:0"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            with accelerator_guard(device, torch):
                model = model.to(device)
            model.eval()
            self.device = device
            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model
            return torch, tokenizer, model

    def translate(self, text: str) -> dict[str, object]:
        source = " ".join(text.strip().split())
        if not source:
            raise ValueError("Nội dung cần dịch không được rỗng")
        started = time.perf_counter()
        safe_translation = translate_visual_answer(source)
        if safe_translation is not None:
            return {
                "schema_version": "1.0",
                "module": "translation_en_vi",
                "success": True,
                "model": None,
                "device": "cpu",
                "source": source,
                "translation": safe_translation,
                "method": "visual_lexicon_v1",
                "quality_assured": True,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            }
        if not self.allow_unverified_model_fallback:
            raise RuntimeError(
                "Câu trả lời nằm ngoài miền dịch thị giác an toàn; hệ thống đã từ chối dịch."
            )

        torch, tokenizer, model = self._load()
        inputs = tokenizer(
            source,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        with accelerator_guard(self.device, torch):
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_length=64,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.1,
                    renormalize_logits=True,
                )
            translation = tokenizer.decode(
                generated[0], skip_special_tokens=True
            ).strip()
        for prefix in ("Câu trả lời là ", "Câu trả lời: ", "Đáp án là ", "Đáp án: "):
            if translation.casefold().startswith(prefix.casefold()):
                translation = translation[len(prefix) :].strip()
                break
        if not translation:
            raise RuntimeError("Model dịch không trả kết quả")
        if "♪" in translation:
            raise RuntimeError("Model dịch trả kết quả chứa ký hiệu bất thường")
        words = translation.casefold().split()
        most_common = Counter(words).most_common(1)[0][1]
        if len(words) >= 8 and most_common > max(4, len(words) // 3):
            raise RuntimeError("Model dịch trả kết quả bị lặp bất thường")
        return {
            "schema_version": "1.0",
            "module": "translation_en_vi",
            "success": True,
            "model": self.model_name,
            "device": self.device,
            "source": source,
            "translation": translation,
            "method": "unverified_marian_fallback",
            "quality_assured": False,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        }
