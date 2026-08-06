"""Position-invariant hand-shape features for MediaPipe hand landmarks.

The core model intentionally avoids a neural-network dependency. It turns one
frame of 21 MediaPipe hand landmarks into a compact shape vector that can be
matched with KNN or nearest-prototype logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import median
from typing import Any, Iterable, Sequence

Point3 = tuple[float, float, float]

LANDMARK_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

FINGER_CHAINS: tuple[tuple[int, int, int, int], ...] = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)

TIP_INDICES: tuple[int, ...] = (4, 8, 12, 16, 20)
MCP_INDICES: tuple[int, ...] = (2, 5, 9, 13, 17)
PALM_INDICES: tuple[int, ...] = (0, 5, 9, 13, 17)

PAIRWISE_DISTANCE_INDICES: tuple[tuple[int, int], ...] = (
    (4, 8),
    (4, 12),
    (4, 16),
    (4, 20),
    (8, 12),
    (8, 16),
    (8, 20),
    (12, 16),
    (12, 20),
    (16, 20),
    (5, 17),
    (0, 9),
)


@dataclass(slots=True)
class FeatureConfig:
    """Controls the hand-shape vector used by the recognizer.

    The defaults favor "same hand shape anywhere in the camera frame". Raw
    camera-space landmark streams should still be sent to Max/MSP separately
    for continuous theremin-style control.
    """

    origin: str = "wrist"
    include_z: bool = True
    include_normalized_points: bool = True
    include_pairwise_distances: bool = True
    include_joint_angles: bool = True
    mirror_left_hand: bool = True
    rotation_invariant_xy: bool = False
    l2_normalize: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "FeatureConfig":
        if not data:
            return cls()
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in data.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def coerce_landmarks(landmarks: Sequence[Any]) -> list[Point3]:
    """Accept MediaPipe landmark objects or plain x/y/z sequences."""

    if len(landmarks) != 21:
        raise ValueError(f"expected 21 MediaPipe hand landmarks, got {len(landmarks)}")

    points: list[Point3] = []
    for landmark in landmarks:
        if hasattr(landmark, "x") and hasattr(landmark, "y"):
            x = float(landmark.x)
            y = float(landmark.y)
            z = float(getattr(landmark, "z", 0.0))
        else:
            values = list(landmark)
            if len(values) < 2:
                raise ValueError("each landmark must have at least x and y")
            x = float(values[0])
            y = float(values[1])
            z = float(values[2]) if len(values) > 2 else 0.0
        points.append((x, y, z))
    return points


def normalize_landmarks(
    landmarks: Sequence[Any],
    config: FeatureConfig | None = None,
    handedness: str | None = None,
) -> list[Point3]:
    """Translate and scale landmarks so frame position does not affect matching."""

    cfg = config or FeatureConfig()
    points = coerce_landmarks(landmarks)

    if cfg.origin == "wrist":
        origin = points[0]
    elif cfg.origin == "palm_center":
        origin = _centroid(points, PALM_INDICES)
    else:
        raise ValueError(f"unsupported feature origin: {cfg.origin}")

    translated = [_sub(point, origin) for point in points]
    scale = _hand_scale(points)
    normalized = [(x / scale, y / scale, z / scale) for x, y, z in translated]

    if cfg.mirror_left_hand and handedness and handedness.lower().startswith("left"):
        normalized = [(-x, y, z) for x, y, z in normalized]

    if cfg.rotation_invariant_xy:
        normalized = _align_xy_axis(normalized, 9)

    if not cfg.include_z:
        normalized = [(x, y, 0.0) for x, y, _ in normalized]

    return normalized


def extract_features(
    landmarks: Sequence[Any],
    config: FeatureConfig | None = None,
    handedness: str | None = None,
) -> list[float]:
    """Create a compact feature vector from one landmark frame."""

    cfg = config or FeatureConfig()
    points = normalize_landmarks(landmarks, cfg, handedness)
    features: list[float] = []

    if cfg.include_normalized_points:
        for x, y, z in points:
            features.extend((x, y))
            if cfg.include_z:
                features.append(z)

    if cfg.include_pairwise_distances:
        palm_center = _centroid(points, PALM_INDICES)
        for first, second in PAIRWISE_DISTANCE_INDICES:
            features.append(_distance(points[first], points[second]))
        for index in TIP_INDICES:
            features.append(_distance(points[index], points[0]))
            features.append(_distance(points[index], palm_center))
        for tip, mcp in zip(TIP_INDICES, MCP_INDICES, strict=True):
            features.append(_distance(points[tip], points[mcp]))

    if cfg.include_joint_angles:
        for base, lower, upper, tip in FINGER_CHAINS:
            features.append(_joint_angle(points[base], points[lower], points[upper]))
            features.append(_joint_angle(points[lower], points[upper], points[tip]))

    return _l2_normalize(features) if cfg.l2_normalize else features


def _hand_scale(points: Sequence[Point3]) -> float:
    distances = [
        _distance(points[0], points[9]),
        _distance(points[5], points[17]),
        _distance(points[0], points[5]),
        _distance(points[0], points[17]),
    ]
    usable = [value for value in distances if value > 1e-6]
    return median(usable) if usable else 1.0


def _align_xy_axis(points: Sequence[Point3], landmark_index: int) -> list[Point3]:
    x, y, _ = points[landmark_index]
    if abs(x) < 1e-9 and abs(y) < 1e-9:
        return list(points)
    current = math.atan2(y, x)
    target = math.pi / 2.0
    theta = target - current
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    aligned = []
    for px, py, pz in points:
        aligned.append((px * cos_t - py * sin_t, px * sin_t + py * cos_t, pz))
    return aligned


def _joint_angle(a: Point3, b: Point3, c: Point3) -> float:
    ba = _sub(a, b)
    bc = _sub(c, b)
    denom = _norm(ba) * _norm(bc)
    if denom < 1e-9:
        return 0.0
    cosine = max(-1.0, min(1.0, _dot(ba, bc) / denom))
    return math.acos(cosine) / math.pi


def _l2_normalize(values: Iterable[float]) -> list[float]:
    vector = [float(value) for value in values]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm < 1e-12:
        return vector
    return [value / norm for value in vector]


def _centroid(points: Sequence[Point3], indices: Sequence[int]) -> Point3:
    count = float(len(indices))
    return (
        sum(points[index][0] for index in indices) / count,
        sum(points[index][1] for index in indices) / count,
        sum(points[index][2] for index in indices) / count,
    )


def _distance(a: Point3, b: Point3) -> float:
    return _norm(_sub(a, b))


def _sub(a: Point3, b: Point3) -> Point3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Point3, b: Point3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(point: Point3) -> float:
    return math.sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2])
