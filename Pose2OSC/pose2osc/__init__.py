"""Lightweight gesture enrollment and recognition for live OSC instruments."""

from .features import FeatureConfig, LANDMARK_NAMES, extract_features, normalize_landmarks
from .recognizer import (
    GestureModel,
    GestureSample,
    GestureStateTracker,
    Prediction,
    RecognitionConfig,
    StateConfig,
    StateUpdate,
)

__all__ = [
    "FeatureConfig",
    "GestureModel",
    "GestureSample",
    "GestureStateTracker",
    "LANDMARK_NAMES",
    "Prediction",
    "RecognitionConfig",
    "StateConfig",
    "StateUpdate",
    "extract_features",
    "normalize_landmarks",
]
