"""KNN gesture enrollment, prediction, and low-latency state tracking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

from .features import FeatureConfig, extract_features


@dataclass(slots=True)
class RecognitionConfig:
    k: int = 3
    min_vote_confidence: float = 0.55
    fallback_distance_threshold: float = 0.38
    min_label_distance_threshold: float = 0.22
    max_label_distance_threshold: float = 0.65
    threshold_stddevs: float = 2.5

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RecognitionConfig":
        if not data:
            return cls()
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in data.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GestureSample:
    label: str
    features: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "features": self.features,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GestureSample":
        return cls(
            label=str(data["label"]),
            features=[float(value) for value in data["features"]],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class Prediction:
    label: str | None
    accepted: bool
    distance: float
    confidence: float
    vote_confidence: float
    distance_confidence: float
    threshold: float
    votes: dict[str, float]

    @property
    def is_known(self) -> bool:
        return self.label is not None and self.accepted


class GestureModel:
    """A user-trained KNN recognizer over normalized hand-shape features."""

    version = 2

    def __init__(
        self,
        feature_config: FeatureConfig | None = None,
        recognition_config: RecognitionConfig | None = None,
        samples: Sequence[GestureSample] | None = None,
        thresholds: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
        label_metadata: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.feature_config = feature_config or FeatureConfig()
        self.recognition_config = recognition_config or RecognitionConfig()
        self.samples: list[GestureSample] = list(samples or [])
        self.thresholds: dict[str, float] = dict(thresholds or {})
        self.metadata: dict[str, Any] = {
            "format": "pose2osc_manifest",
            "created_at": _utc_now(),
            **(metadata or {}),
        }
        self.label_metadata: dict[str, dict[str, Any]] = {
            str(label): dict(values)
            for label, values in (label_metadata or {}).items()
        }
        if self.samples and not self.thresholds:
            self.fit_thresholds()

    @property
    def labels(self) -> list[str]:
        return sorted({sample.label for sample in self.samples})

    def add_sample(
        self,
        label: str,
        landmarks: Sequence[Any],
        handedness: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GestureSample:
        clean_label = label.strip()
        if not clean_label:
            raise ValueError("gesture label cannot be empty")
        sample = GestureSample(
            label=clean_label,
            features=extract_features(landmarks, self.feature_config, handedness),
            metadata={
                "created_at": datetime.now(timezone.utc).isoformat(),
                **(metadata or {}),
            },
        )
        self.samples.append(sample)
        self._update_label_metadata(clean_label, handedness, metadata)
        self.fit_thresholds()
        return sample

    def add_samples(
        self,
        label: str,
        frames: Sequence[Sequence[Any]],
        handedness: str | None = None,
        metadata: dict[str, Any] | None = None,
        max_samples: int | None = 64,
    ) -> int:
        clean_label = label.strip()
        if not clean_label:
            raise ValueError("gesture label cannot be empty")
        selected_frames = _uniform_sample(frames, max_samples)
        for index, frame in enumerate(selected_frames):
            frame_metadata = {
                "frame_index": index,
                "source": "enrollment",
                "handedness": handedness,
                **(metadata or {}),
            }
            self.samples.append(
                GestureSample(
                    label=clean_label,
                    features=extract_features(frame, self.feature_config, handedness),
                    metadata={
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        **frame_metadata,
                    },
                )
            )
        self._update_label_metadata(clean_label, handedness, metadata)
        self.fit_thresholds()
        return len(selected_frames)

    def remove_label(self, label: str) -> int:
        before = len(self.samples)
        self.samples = [sample for sample in self.samples if sample.label != label]
        self.thresholds.pop(label, None)
        self.label_metadata.pop(label, None)
        self.fit_thresholds()
        return before - len(self.samples)

    def predict(self, landmarks: Sequence[Any], handedness: str | None = None) -> Prediction:
        if not self.samples:
            return Prediction(
                label=None,
                accepted=False,
                distance=math.inf,
                confidence=0.0,
                vote_confidence=0.0,
                distance_confidence=0.0,
                threshold=self.recognition_config.fallback_distance_threshold,
                votes={},
            )

        vector = extract_features(landmarks, self.feature_config, handedness)
        neighbors = sorted(
            ((_euclidean(vector, sample.features), sample) for sample in self.samples),
            key=lambda item: item[0],
        )
        k = max(1, min(self.recognition_config.k, len(neighbors)))
        nearest = neighbors[:k]

        votes: dict[str, float] = {}
        best_distance_by_label: dict[str, float] = {}
        for distance, sample in nearest:
            votes[sample.label] = votes.get(sample.label, 0.0) + 1.0 / max(distance, 1e-9)
            best_distance_by_label[sample.label] = min(
                best_distance_by_label.get(sample.label, math.inf),
                distance,
            )

        label = max(votes.items(), key=lambda item: item[1])[0]
        total_vote = sum(votes.values())
        vote_confidence = votes[label] / total_vote if total_vote > 0.0 else 0.0
        distance = best_distance_by_label[label]
        threshold = self.thresholds.get(
            label,
            self.recognition_config.fallback_distance_threshold,
        )
        distance_confidence = math.exp(-((distance / max(threshold, 1e-9)) ** 2))
        confidence = vote_confidence * distance_confidence
        accepted = (
            distance <= threshold
            and vote_confidence >= self.recognition_config.min_vote_confidence
        )

        return Prediction(
            label=label if accepted else None,
            accepted=accepted,
            distance=distance,
            confidence=confidence,
            vote_confidence=vote_confidence,
            distance_confidence=distance_confidence,
            threshold=threshold,
            votes=votes,
        )

    def fit_thresholds(self) -> None:
        by_label: dict[str, list[list[float]]] = {}
        for sample in self.samples:
            by_label.setdefault(sample.label, []).append(sample.features)

        thresholds: dict[str, float] = {}
        cfg = self.recognition_config
        for label, vectors in by_label.items():
            if len(vectors) < 2:
                thresholds[label] = cfg.fallback_distance_threshold
                continue
            centroid = _centroid(vectors)
            distances = [_euclidean(vector, centroid) for vector in vectors]
            fitted = mean(distances) + pstdev(distances) * cfg.threshold_stddevs
            thresholds[label] = min(
                cfg.max_label_distance_threshold,
                max(cfg.min_label_distance_threshold, fitted),
            )
        self.thresholds = thresholds

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.metadata["updated_at"] = _utc_now()
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "GestureModel":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "pose2osc_manifest",
            "version": self.version,
            "metadata": self.metadata,
            "label_metadata": self.label_metadata,
            "feature_config": self.feature_config.to_dict(),
            "recognition_config": self.recognition_config.to_dict(),
            "thresholds": self.thresholds,
            "samples": [sample.to_dict() for sample in self.samples],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GestureModel":
        return cls(
            feature_config=FeatureConfig.from_dict(data.get("feature_config")),
            recognition_config=RecognitionConfig.from_dict(data.get("recognition_config")),
            samples=[GestureSample.from_dict(item) for item in data.get("samples", [])],
            thresholds={
                str(label): float(value)
                for label, value in data.get("thresholds", {}).items()
            },
            metadata=dict(data.get("metadata", {})),
            label_metadata={
                str(label): dict(values)
                for label, values in data.get("label_metadata", {}).items()
            },
        )

    def _update_label_metadata(
        self,
        label: str,
        handedness: str | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        existing = self.label_metadata.setdefault(label, {})
        existing.update(metadata or {})
        if handedness:
            existing["handedness"] = handedness
        existing["sample_count"] = sum(1 for sample in self.samples if sample.label == label)
        existing["updated_at"] = _utc_now()


@dataclass(slots=True)
class StateConfig:
    enter_frames: int = 1
    exit_frames: int = 1
    switch_frames: int = 1


@dataclass(slots=True)
class StateUpdate:
    active_label: str | None
    previous_label: str | None
    event: str
    prediction: Prediction
    active: bool


class GestureStateTracker:
    """Turns one-frame predictions into enter/hold/exit events.

    Defaults are intentionally immediate for performance instruments. Raising
    exit_frames to 2 or 3 can soften dropouts without adding entry latency.
    """

    def __init__(self, config: StateConfig | None = None) -> None:
        self.config = config or StateConfig()
        self.active_label: str | None = None
        self.candidate_label: str | None = None
        self.candidate_count = 0
        self.exit_count = 0

    def update(self, prediction: Prediction) -> StateUpdate:
        if prediction.accepted and prediction.label:
            self.exit_count = 0
            label = prediction.label
            if self.active_label == label:
                self.candidate_label = None
                self.candidate_count = 0
                return StateUpdate(label, None, "hold", prediction, True)

            if self.candidate_label == label:
                self.candidate_count += 1
            else:
                self.candidate_label = label
                self.candidate_count = 1

            required = (
                self.config.switch_frames
                if self.active_label is not None
                else self.config.enter_frames
            )
            if self.candidate_count >= max(1, required):
                previous = self.active_label
                self.active_label = label
                self.candidate_label = None
                self.candidate_count = 0
                event = "switch" if previous is not None else "enter"
                return StateUpdate(label, previous, event, prediction, True)

            return StateUpdate(
                self.active_label,
                None,
                "pending",
                prediction,
                self.active_label is not None,
            )

        self.candidate_label = None
        self.candidate_count = 0
        if self.active_label is None:
            self.exit_count = 0
            return StateUpdate(None, None, "none", prediction, False)

        self.exit_count += 1
        if self.exit_count >= max(1, self.config.exit_frames):
            exited = self.active_label
            self.active_label = None
            self.exit_count = 0
            return StateUpdate(None, exited, "exit", prediction, False)

        return StateUpdate(self.active_label, None, "hold", prediction, True)


def _uniform_sample(
    frames: Sequence[Sequence[Any]],
    max_samples: int | None,
) -> list[Sequence[Any]]:
    if max_samples is None or max_samples <= 0 or len(frames) <= max_samples:
        return list(frames)
    if max_samples == 1:
        return [frames[len(frames) // 2]]
    step = (len(frames) - 1) / float(max_samples - 1)
    return [frames[round(index * step)] for index in range(max_samples)]


def _centroid(vectors: Sequence[Sequence[float]]) -> list[float]:
    width = len(vectors[0])
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(width)]


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"feature length mismatch: {len(a)} != {len(b)}")
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b, strict=True)))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
