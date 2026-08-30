"""Pretrained multimodal adapters used by the local integration baseline."""

from .depth import DepthAnythingEstimator, attach_depth_zones
from .ocr import PaddleOcrReader
from .speech import (
    FFmpegMicrophoneRecorder,
    MacOSTextToSpeech,
    WhisperSpeechToText,
    localize_vqa_answer,
    macos_voice_available,
    normalize_vietnamese_speech,
)
from .translation import PretrainedEnglishVietnameseTranslator
from .vqa import PretrainedVisualQuestionAnswering

__all__ = [
    "DepthAnythingEstimator",
    "FFmpegMicrophoneRecorder",
    "MacOSTextToSpeech",
    "PaddleOcrReader",
    "PretrainedVisualQuestionAnswering",
    "PretrainedEnglishVietnameseTranslator",
    "WhisperSpeechToText",
    "attach_depth_zones",
    "localize_vqa_answer",
    "macos_voice_available",
    "normalize_vietnamese_speech",
]
