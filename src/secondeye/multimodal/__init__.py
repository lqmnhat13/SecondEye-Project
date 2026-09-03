"""Pretrained multimodal adapters used by the local integration baseline."""

from .depth import (
    DepthAnythingEstimator,
    DepthFusionConfig,
    attach_depth_zones,
    attach_metric_depth_zones,
)
from .depth_provider import AlignedMetricDepthFrame, SynchronizedDepthProvider
from .ocr import (
    AppleVisionOcrReader,
    AutomaticOcrReader,
    OcrConsensusConfig,
    PaddleOcrReader,
)
from .open_vocabulary import GroundingDinoDetector
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
    "DepthFusionConfig",
    "AlignedMetricDepthFrame",
    "SynchronizedDepthProvider",
    "FFmpegMicrophoneRecorder",
    "MacOSTextToSpeech",
    "AppleVisionOcrReader",
    "AutomaticOcrReader",
    "PaddleOcrReader",
    "OcrConsensusConfig",
    "GroundingDinoDetector",
    "VisualQuestion",
    "normalize_visual_question",
    "plain_vietnamese",
    "PretrainedVisualQuestionAnswering",
    "PretrainedEnglishVietnameseTranslator",
    "WhisperSpeechToText",
    "attach_depth_zones",
    "attach_metric_depth_zones",
    "localize_vqa_answer",
    "macos_voice_available",
    "normalize_vietnamese_speech",
]
