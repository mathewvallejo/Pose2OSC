"""Optional MediaPipe camera runtime and OSC transport."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Iterable, Sequence

from .features import LANDMARK_NAMES
from .labels import LabelStyle, bgr_from_hex, label_style
from .recognizer import GestureModel, GestureStateTracker, Prediction, StateConfig

WINDOW_NAME = "Pose2OSC"


@dataclass(slots=True)
class OscConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    split_axis_messages: bool = False
    send_landmark_vectors: bool = True
    send_unknown_predictions: bool = False


def enroll_from_camera(
    *,
    label: str | Sequence[str],
    model_path: str,
    seconds: float = 2.0,
    capture_frames: int = 45,
    target_captures: int = 5,
    camera: int = 0,
    max_samples: int = 64,
    handedness: str | None = None,
    correct_handedness: bool = True,
    show: bool = False,
    timed: bool = False,
    replace: bool = False,
    width: int | None = None,
    height: int | None = None,
) -> int:
    """Record a short held gesture and append it to a JSON model file."""

    labels = _coerce_labels(label)
    interactive = show and not timed
    if len(labels) > 1 and not interactive:
        raise ValueError("enrolling multiple gesture labels in one session requires --show")

    cv2, mp = _load_camera_dependencies()
    model = _load_or_new_model(model_path)
    label_styles = _label_styles(model, labels)
    frames: list[list[tuple[float, float, float]]] = []
    recent_frames: list[tuple[str, list[tuple[float, float, float]]]] = []
    detected_handedness: str | None = None
    active_label_index = 0
    capture_counts = {
        gesture_label: 0 if replace else _existing_capture_count(model, gesture_label)
        for gesture_label in labels
    }
    saved_sample_count = 0
    replaced_labels: set[str] = set()

    with _open_capture(cv2, camera, width, height) as capture:
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=_max_hands_for_mode(handedness),
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        try:
            if interactive:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        continue

                    active_label = labels[active_label_index]
                    active_style = label_styles[active_label]
                    active_color = bgr_from_hex(active_style.color)
                    result = _process_frame(cv2, hands, frame)
                    selected = _select_hands(result, handedness, correct_handedness=correct_handedness)
                    if selected:
                        detected_handedness = _hand_mode_label(selected)
                        landmarks = _model_landmarks(selected)
                        recent_frames.append((detected_handedness, landmarks))
                        if len(recent_frames) > max(1, capture_frames):
                            recent_frames.pop(0)

                    capture_count = capture_counts[active_label]
                    current_mode = recent_frames[-1][0] if recent_frames else None
                    matching_recent_frames = [
                        landmarks
                        for mode, landmarks in recent_frames
                        if mode == current_mode
                    ]
                    status = [
                        "Gesture capture mode",
                        f"Gesture {active_label_index + 1}/{len(labels)}: {active_style.display_label}",
                        "Press Space to capture",
                        "Press q to quit",
                        f"Captured: {capture_count}/{target_captures}",
                    ]
                    if len(labels) > 1:
                        status.insert(3, "Press n for next | p for previous")
                    if current_mode:
                        status.append(
                            f"Detected: {current_mode} | Buffer: {len(matching_recent_frames)}/{capture_frames}"
                        )
                    if capture_count >= target_captures:
                        if len(labels) > 1 and active_label_index < len(labels) - 1:
                            status.append("Target reached: press n for next gesture")
                        else:
                            status.append("Target reached: press q to finish or Space for more")
                    if not selected:
                        status.append(_missing_hand_message(handedness))
                    _show_status(
                        cv2,
                        frame,
                        status,
                        mp=mp,
                        result=result,
                        line_colors=_line_colors(len(status), 1, active_color),
                    )
                    if _window_was_closed(cv2):
                        break
                    key = _read_key(cv2)

                    if key in {27, ord("q")}:
                        break
                    if key in {10, 13, ord("n")}:
                        if active_label_index < len(labels) - 1:
                            active_label_index += 1
                            recent_frames.clear()
                            detected_handedness = None
                        continue
                    if key == ord("p"):
                        if active_label_index > 0:
                            active_label_index -= 1
                            recent_frames.clear()
                            detected_handedness = None
                        continue
                    if key == ord(" "):
                        if not matching_recent_frames or not current_mode:
                            continue
                        if replace and active_label not in replaced_labels:
                            model.remove_label(active_label)
                            replaced_labels.add(active_label)
                        capture_counts[active_label] += 1
                        capture_count = capture_counts[active_label]
                        added = model.add_samples(
                            active_label,
                            matching_recent_frames,
                            handedness=current_mode,
                            metadata={
                                "capture_mode": "spacebar",
                                "capture_count": capture_count,
                                "capture_frames": len(matching_recent_frames),
                                "capture_seconds": None,
                                "hand_mode": current_mode,
                                "display_label": active_style.display_label,
                                "color": active_style.color,
                            },
                            max_samples=max_samples,
                        )
                        saved_sample_count += added
                        model.save(model_path)
                        feedback = [
                            "Captured and saved",
                            f"Gesture {active_label_index + 1}/{len(labels)}: {active_style.display_label}",
                            f"Hand mode: {current_mode}",
                            f"Samples added: {added}",
                            f"Captured: {capture_count}/{target_captures}",
                            "Press Space to capture again",
                            "Press q to quit",
                        ]
                        if len(labels) > 1:
                            feedback.insert(-1, "Press n for next gesture")
                        _show_status(
                            cv2,
                            frame,
                            feedback,
                            mp=mp,
                            result=result,
                            line_colors=_line_colors(len(feedback), 1, active_color),
                        )
                        _read_key(cv2, delay_ms=250)
                return saved_sample_count
            else:
                active_style = label_styles[labels[0]]
                start = time.monotonic()
                while time.monotonic() - start < seconds:
                    ok, frame = capture.read()
                    if not ok:
                        continue
                    result = _process_frame(cv2, hands, frame)
                    selected = _select_hands(result, handedness, correct_handedness=correct_handedness)
                    if selected:
                        detected_handedness = _hand_mode_label(selected)
                        landmarks = _model_landmarks(selected)
                        frames.append(landmarks)

                    if show:
                        remaining = max(0.0, seconds - (time.monotonic() - start))
                        _show_status(
                            cv2,
                            frame,
                            [
                                "Timed capture mode",
                                f"Label: {active_style.display_label}",
                                f"Remaining: {remaining:0.1f}s",
                                "Press q to quit",
                            ],
                            mp=mp,
                            result=result,
                            line_colors=_line_colors(4, 1, bgr_from_hex(active_style.color)),
                        )
                        if _should_stop(cv2, show):
                            break
        except KeyboardInterrupt:
            return saved_sample_count
        finally:
            hands.close()
            if show:
                cv2.destroyAllWindows()

    if not frames:
        raise RuntimeError("no hand landmarks were captured during enrollment")

    if replace:
        model.remove_label(labels[0])

    sample_count = model.add_samples(
        labels[0],
        frames,
        handedness=detected_handedness or handedness,
        metadata={
            "capture_mode": "timed",
            "capture_count": 1,
            "capture_frames": len(frames),
            "capture_seconds": seconds,
            "hand_mode": detected_handedness or handedness,
            "display_label": label_styles[labels[0]].display_label,
            "color": label_styles[labels[0]].color,
        },
        max_samples=max_samples,
    )
    model.save(model_path)
    return sample_count


def run_osc_camera(
    *,
    model_path: str,
    osc: OscConfig | None = None,
    camera: int = 0,
    handedness: str | None = None,
    correct_handedness: bool = True,
    show: bool = False,
    width: int | None = None,
    height: int | None = None,
    state_config: StateConfig | None = None,
) -> None:
    """Run the one-frame recognizer and stream OSC to Max/MSP."""

    cv2, mp = _load_camera_dependencies()
    udp_client = _load_osc_client()
    model = GestureModel.load(model_path)
    tracker = GestureStateTracker(state_config or StateConfig())
    osc_cfg = osc or OscConfig()
    client = udp_client.SimpleUDPClient(osc_cfg.host, osc_cfg.port)
    frame_index = 0

    with _open_capture(cv2, camera, width, height) as capture:
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=_max_hands_for_mode(handedness),
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    continue
                frame_index += 1
                result = _process_frame(cv2, hands, frame)
                selected = _select_hands(result, handedness, correct_handedness=correct_handedness)
                now_ms = int(time.time() * 1000)

                if not selected:
                    prediction = _unknown_prediction(model)
                    state = tracker.update(prediction)
                    _send_prediction(client, prediction, state, osc_cfg)
                    _send_no_hand(client, now_ms, frame_index)
                    if show:
                        _show_status(
                            cv2,
                            frame,
                            [
                                "Performance mode",
                                "Gesture: none",
                                _missing_hand_message(handedness),
                                "Press q to quit",
                            ],
                            mp=mp,
                            result=result,
                        )
                    if _should_stop(cv2, show):
                        break
                    continue

                detected_handedness = _hand_mode_label(selected)
                prediction = _predict_selection(model, selected)
                state = tracker.update(prediction)

                client.send_message("/pose2osc/hand/visible", 1)
                client.send_message("/pose2osc/hand/num_hands", len(selected))
                for hand_label, landmarks in selected:
                    _send_landmarks(client, hand_label.lower(), landmarks, osc_cfg)
                _send_prediction(client, prediction, state, osc_cfg)
                client.send_message("/pose2osc/frame", [frame_index, now_ms])

                if show:
                    label = state.active_label or "none"
                    display_label = _display_label(model, label)
                    label_color = _label_color_bgr(model, label)
                    _show_status(
                        cv2,
                        frame,
                        [
                            "Performance mode",
                            f"Gesture: {display_label}",
                            f"Detected: {detected_handedness}",
                            f"Confidence: {prediction.confidence:0.2f}",
                            "Press q to quit",
                        ],
                        mp=mp,
                        result=result,
                        line_colors=_line_colors(5, 1, label_color),
                    )
                if _should_stop(cv2, show):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            hands.close()
            if show:
                cv2.destroyAllWindows()


def _load_or_new_model(path: str) -> GestureModel:
    from pathlib import Path

    model_path = Path(path)
    if model_path.exists():
        return GestureModel.load(model_path)
    return GestureModel()


def _coerce_labels(label: str | Sequence[str]) -> list[str]:
    raw_labels = [label] if isinstance(label, str) else list(label)
    labels: list[str] = []
    seen: set[str] = set()
    for raw_label in raw_labels:
        clean_label = str(raw_label).strip()
        if not clean_label:
            raise ValueError("gesture labels cannot be empty")
        if clean_label in seen:
            raise ValueError(f"duplicate gesture label: {clean_label}")
        labels.append(clean_label)
        seen.add(clean_label)
    if not labels:
        raise ValueError("at least one gesture label is required")
    return labels


def _existing_capture_count(model: GestureModel, label: str) -> int:
    metadata_count = model.label_metadata.get(label, {}).get("capture_count")
    if isinstance(metadata_count, int):
        return metadata_count

    capture_ids = {
        sample.metadata.get("capture_count")
        for sample in model.samples
        if sample.label == label and sample.metadata.get("capture_count") is not None
    }
    return len(capture_ids)


def _label_styles(model: GestureModel, labels: Sequence[str]) -> dict[str, LabelStyle]:
    return {
        label: label_style(label, model.label_metadata.get(label), index)
        for index, label in enumerate(labels)
    }


def _display_label(model: GestureModel, label: str) -> str:
    if label == "none":
        return label
    return label_style(label, model.label_metadata.get(label)).display_label


def _label_color_bgr(model: GestureModel, label: str) -> tuple[int, int, int]:
    if label == "none":
        return (0, 255, 0)
    return bgr_from_hex(label_style(label, model.label_metadata.get(label)).color)


def _line_colors(
    count: int,
    index: int,
    color: tuple[int, int, int],
) -> list[tuple[int, int, int] | None]:
    colors: list[tuple[int, int, int] | None] = [None] * count
    if 0 <= index < count:
        colors[index] = color
    return colors


def _load_camera_dependencies() -> tuple[Any, Any]:
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Live camera commands require optional dependencies. "
            "Install them with: python -m pip install -e '.[live]'"
        ) from exc
    return cv2, mp


def _load_osc_client() -> Any:
    try:
        from pythonosc import udp_client  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "OSC output requires python-osc. Install with: python -m pip install -e '.[live]'"
        ) from exc
    return udp_client


class _CaptureContext:
    def __init__(self, capture: Any) -> None:
        self.capture = capture

    def __enter__(self) -> Any:
        return self.capture

    def __exit__(self, *_: object) -> None:
        self.capture.release()


def _open_capture(
    cv2: Any,
    camera: int,
    width: int | None,
    height: int | None,
) -> _CaptureContext:
    capture = cv2.VideoCapture(camera)
    if width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not capture.isOpened():
        raise RuntimeError(f"could not open camera {camera}")
    return _CaptureContext(capture)


def _process_frame(cv2: Any, hands: Any, frame: Any) -> Any:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    return hands.process(rgb)


def _max_hands_for_mode(handedness: str | None) -> int:
    return 1 if _specific_single_hand(handedness) else 2


def _specific_single_hand(handedness: str | None) -> bool:
    return bool(handedness and handedness.lower() in {"right", "left"})


def _missing_hand_message(handedness: str | None) -> str:
    if _specific_single_hand(handedness):
        return f"Waiting for {handedness} hand"
    return "No hand detected"


def _select_hands(
    result: Any,
    desired_handedness: str | None,
    *,
    correct_handedness: bool = True,
) -> list[tuple[str, list[tuple[float, float, float]]]]:
    if not result.multi_hand_landmarks:
        return []

    handedness_labels = _handedness_labels(
        result.multi_handedness,
        correct_handedness=correct_handedness,
    )
    selected: list[tuple[str, list[tuple[float, float, float]]]] = []
    for index, hand_landmarks in enumerate(result.multi_hand_landmarks):
        label = handedness_labels[index] if index < len(handedness_labels) else "unknown"
        if desired_handedness and desired_handedness.lower() != "any":
            if _specific_single_hand(desired_handedness) and label.lower() != desired_handedness.lower():
                continue
        selected.append((
            _canonical_hand_label(label),
            [
                (float(landmark.x), float(landmark.y), float(landmark.z))
                for landmark in hand_landmarks.landmark
            ],
        ))

    by_label = {label.lower(): landmarks for label, landmarks in selected}
    if "right" in by_label and "left" in by_label:
        return [("Right", by_label["right"]), ("Left", by_label["left"])]

    return selected[:1]


def _canonical_hand_label(label: str) -> str:
    lower = label.lower()
    if lower == "right":
        return "Right"
    if lower == "left":
        return "Left"
    return label


def _hand_mode_label(selected: Sequence[tuple[str, Sequence[tuple[float, float, float]]]]) -> str:
    if len(selected) == 2:
        labels = {label.lower() for label, _ in selected}
        if labels == {"right", "left"}:
            return "Both"
    return selected[0][0] if selected else "Any"


def _model_landmarks(
    selected: Sequence[tuple[str, list[tuple[float, float, float]]]],
) -> list[tuple[float, float, float]]:
    if len(selected) == 2:
        by_label = {label.lower(): landmarks for label, landmarks in selected}
        if "right" in by_label and "left" in by_label:
            return list(by_label["right"]) + list(by_label["left"])
    return list(selected[0][1]) if selected else []


def _predict_selection(
    model: GestureModel,
    selected: Sequence[tuple[str, list[tuple[float, float, float]]]],
) -> Prediction:
    candidates: list[tuple[str, list[tuple[float, float, float]]]] = []
    if len(selected) == 2:
        candidates.append(("Both", _model_landmarks(selected)))
    for label, landmarks in selected:
        candidates.append((label, list(landmarks)))

    predictions = [
        model.predict(landmarks, hand_mode)
        for hand_mode, landmarks in candidates
        if landmarks
    ]
    accepted = [prediction for prediction in predictions if prediction.accepted]
    if accepted:
        return max(accepted, key=lambda prediction: prediction.confidence)
    if predictions:
        return max(predictions, key=lambda prediction: prediction.confidence)
    return _unknown_prediction(model)


def _handedness_labels(
    multi_handedness: Iterable[Any] | None,
    *,
    correct_handedness: bool = True,
) -> list[str]:
    labels: list[str] = []
    if not multi_handedness:
        return labels
    for item in multi_handedness:
        try:
            labels.append(_correct_handedness_label(
                str(item.classification[0].label),
                correct_handedness=correct_handedness,
            ))
        except (AttributeError, IndexError):
            labels.append("unknown")
    return labels


def _correct_handedness_label(label: str, *, correct_handedness: bool = True) -> str:
    if not correct_handedness:
        return label
    lower = label.lower()
    if lower == "left":
        return "Right"
    if lower == "right":
        return "Left"
    return label


def _send_landmarks(
    client: Any,
    hand_key: str,
    landmarks: list[tuple[float, float, float]],
    osc_cfg: OscConfig,
) -> None:
    if not osc_cfg.send_landmark_vectors and not osc_cfg.split_axis_messages:
        return
    for index, (x, y, z) in enumerate(landmarks):
        name = LANDMARK_NAMES[index]
        if osc_cfg.send_landmark_vectors:
            client.send_message(f"/pose2osc/hand/{hand_key}/{name}", [x, y, z])
        if osc_cfg.split_axis_messages:
            client.send_message(f"/pose2osc/hand/{hand_key}/{name}/x", x)
            client.send_message(f"/pose2osc/hand/{hand_key}/{name}/y", y)
            client.send_message(f"/pose2osc/hand/{hand_key}/{name}/z", z)


def _send_prediction(client: Any, prediction: Any, state: Any, osc_cfg: OscConfig) -> None:
    if state.event == "switch" and state.previous_label:
        client.send_message(f"/pose2osc/gesture/{state.previous_label}/active", 0)

    if prediction.accepted and state.active_label:
        label = state.active_label
        client.send_message("/pose2osc/state/active", [label, prediction.confidence])
        client.send_message(f"/pose2osc/gesture/{label}/active", 1)
        client.send_message(f"/pose2osc/gesture/{label}/confidence", prediction.confidence)
    elif osc_cfg.send_unknown_predictions:
        client.send_message("/pose2osc/state/active", ["unknown", 0.0])

    if state.event in {"enter", "switch", "exit"}:
        label = state.active_label or state.previous_label or "none"
        client.send_message("/pose2osc/state/event", [state.event, label, prediction.confidence])
        if state.event in {"enter", "switch"} and state.active_label:
            client.send_message(f"/pose2osc/gesture/{state.active_label}/trigger", 1)
        if state.event == "exit" and state.previous_label:
            client.send_message(f"/pose2osc/gesture/{state.previous_label}/active", 0)
            client.send_message(f"/pose2osc/gesture/{state.previous_label}/confidence", 0.0)
            client.send_message("/pose2osc/state/active", ["none", 0.0])


def _send_no_hand(client: Any, now_ms: int, frame_index: int) -> None:
    client.send_message("/pose2osc/hand/visible", 0)
    client.send_message("/pose2osc/hand/num_hands", 0)
    client.send_message("/pose2osc/frame", [frame_index, now_ms])


def _unknown_prediction(model: GestureModel) -> Prediction:
    return Prediction(
        label=None,
        accepted=False,
        distance=float("inf"),
        confidence=0.0,
        vote_confidence=0.0,
        distance_confidence=0.0,
        threshold=model.recognition_config.fallback_distance_threshold,
        votes={},
    )


def _draw_mediapipe_landmarks(mp: Any, frame: Any, result: Any) -> None:
    if not result or not result.multi_hand_landmarks:
        return

    drawing_utils = mp.solutions.drawing_utils
    drawing_styles = getattr(mp.solutions, "drawing_styles", None)
    landmark_style = None
    connection_style = None
    if drawing_styles:
        landmark_style = drawing_styles.get_default_hand_landmarks_style()
        connection_style = drawing_styles.get_default_hand_connections_style()

    for hand_landmarks in result.multi_hand_landmarks:
        drawing_utils.draw_landmarks(
            frame,
            hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            landmark_style,
            connection_style,
        )


def _show_status(
    cv2: Any,
    frame: Any,
    text: str | Sequence[str],
    *,
    mp: Any | None = None,
    result: Any | None = None,
    line_colors: Sequence[tuple[int, int, int] | None] | None = None,
) -> None:
    if mp is not None and result is not None:
        _draw_mediapipe_landmarks(mp, frame, result)

    preview = cv2.flip(frame, 1)
    lines = [text] if isinstance(text, str) else list(text)
    panel_h = 22 + 30 * len(lines)
    cv2.rectangle(preview, (8, 8), (min(920, preview.shape[1] - 8), panel_h), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        y = 34 + index * 30
        scale = 0.78 if index == 0 else 0.62
        thickness = 2 if index == 0 else 1
        color = (
            line_colors[index]
            if line_colors is not None and index < len(line_colors) and line_colors[index]
            else (0, 255, 0)
        )
        cv2.putText(
            preview,
            str(line),
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    cv2.imshow(WINDOW_NAME, preview)


def _read_key(cv2: Any, delay_ms: int = 1) -> int:
    return cv2.waitKey(delay_ms) & 0xFF


def _should_stop(cv2: Any, show: bool) -> bool:
    if not show:
        return False
    if _window_was_closed(cv2):
        return True
    key = _read_key(cv2)
    return key in {27, ord("q")}


def _window_was_closed(cv2: Any) -> bool:
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
    except Exception:
        return False
