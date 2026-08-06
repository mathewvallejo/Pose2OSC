"""Gesture label helpers for display names and stable colors."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

COLOR_PALETTE = (
    "#00D1FF",
    "#FFB000",
    "#FF4FA3",
    "#35D07F",
    "#A88CFF",
    "#FF6B4A",
    "#46E6B2",
    "#F4D35E",
)


@dataclass(frozen=True, slots=True)
class LabelStyle:
    label: str
    display_label: str
    color: str


def generated_gesture_labels(count: int, start_index: int = 1) -> list[str]:
    if count < 1:
        raise ValueError("gesture count must be at least 1")
    if start_index < 1:
        raise ValueError("gesture start index must be at least 1")
    return [
        f"gesture_{index}"
        for index in range(start_index, start_index + count)
    ]


def label_style(
    label: str,
    metadata: dict[str, Any] | None = None,
    index: int | None = None,
) -> LabelStyle:
    values = metadata or {}
    fallback_color = default_color_hex(label, index)
    return LabelStyle(
        label=label,
        display_label=str(values.get("display_label") or default_display_label(label)),
        color=_normalize_color_hex(values.get("color"), fallback_color),
    )


def default_display_label(label: str) -> str:
    gesture_number = _gesture_number(label)
    if gesture_number is not None:
        return f"Gesture {gesture_number}"
    return label


def default_color_hex(label: str, index: int | None = None) -> str:
    gesture_number = _gesture_number(label)
    if gesture_number is not None:
        return COLOR_PALETTE[(gesture_number - 1) % len(COLOR_PALETTE)]

    if index is not None:
        return COLOR_PALETTE[index % len(COLOR_PALETTE)]

    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return COLOR_PALETTE[digest[0] % len(COLOR_PALETTE)]


def bgr_from_hex(color: str) -> tuple[int, int, int]:
    normalized = _normalize_color_hex(color, COLOR_PALETTE[0])
    red = int(normalized[1:3], 16)
    green = int(normalized[3:5], 16)
    blue = int(normalized[5:7], 16)
    return (blue, green, red)


def _normalize_color_hex(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    if (
        len(text) == 7
        and text.startswith("#")
        and all(char in "0123456789abcdefABCDEF" for char in text[1:])
    ):
        return f"#{text[1:].upper()}"
    return fallback


def _gesture_number(label: str) -> int | None:
    prefix = "gesture_"
    if not label.startswith(prefix):
        return None
    suffix = label[len(prefix):]
    if not suffix.isdigit():
        return None
    return int(suffix)
