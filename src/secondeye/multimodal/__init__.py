"""Pretrained multimodal adapters used by the local integration baseline."""

from .depth import DepthAnythingEstimator, attach_depth_zones
from .ocr import AppleVisionOcrReader, AutomaticOcrReader, PaddleOcrReader
from .questions import VisualQuestion, normalize_visual_question, plain_vietnamese
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
    "AppleVisionOcrReader",
    "AutomaticOcrReader",
    "PaddleOcrReader",
    "VisualQuestion",
    "normalize_visual_question",
    "plain_vietnamese",
    "PretrainedVisualQuestionAnswering",
    "PretrainedEnglishVietnameseTranslator",
    "WhisperSpeechToText",
    "attach_depth_zones",
    "localize_vqa_answer",
    "macos_voice_available",
    "normalize_vietnamese_speech",
]
