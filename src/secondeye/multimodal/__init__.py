"""Pretrained multimodal adapters used by the local integration baseline."""

from .depth import DepthAnythingEstimator, attach_depth_zones
from .ocr import PaddleOcrReader
from .speech import MacOSTextToSpeech, WhisperSpeechToText
from .vqa import PretrainedVisualQuestionAnswering

__all__ = [
    "DepthAnythingEstimator",
    "MacOSTextToSpeech",
    "PaddleOcrReader",
    "PretrainedVisualQuestionAnswering",
    "WhisperSpeechToText",
    "attach_depth_zones",
]
