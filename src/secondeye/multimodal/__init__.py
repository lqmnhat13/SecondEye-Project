"""Pretrained multimodal adapters used by the local integration baseline."""

from .depth import DepthAnythingEstimator, attach_depth_zones
from .ocr import PaddleOcrReader
from .speech import (
    MacOSTextToSpeech,
    WhisperSpeechToText,
    localize_vqa_answer,
    macos_voice_available,
    normalize_vietnamese_speech,
)
from .vqa import PretrainedVisualQuestionAnswering

__all__ = [
    "DepthAnythingEstimator",
    "MacOSTextToSpeech",
    "PaddleOcrReader",
    "PretrainedVisualQuestionAnswering",
    "WhisperSpeechToText",
    "attach_depth_zones",
    "localize_vqa_answer",
    "macos_voice_available",
    "normalize_vietnamese_speech",
]
