"""Optional MediaPipe camera runtime and OSC transport."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Iterable

from .features import LANDMARK_NAMES
from .recognizer import GestureModel, GestureStateTracker, Prediction, StateConfig


@dataclass(slots=True)
class OscConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    split_axis_messages: bool = False
    send_landmark_vectors: bool = True
    send_unknown_predictions: bool = False


def enroll_from_camera(
    *,
    label: str,
    model_path: str,
    seconds: float = 2.0,
    capture_frames: int = 45,
    camera: int = 0,
    max_samples: int = 64,
    handedness: str | None = None,
    show: bool = False,
    timed: bool = False,
    replace: bool = False,
    width: int | None = None,
    height: int | None = None,
) -> int:
    """Record a short held gesture and append it to a JSON model file."""

    cv2, mp = _load_camera_dependencies()
    model = _load_or_new_model(model_path)
    frames: list[list[tuple[float, float, float]]] = []
    recent_frames: list[list[tuple[float, float, float]]] = []
    detected_handedness: str | None = None
    capture_count = 0
    interactive = show and not timed

    with _open_capture(cv2, camera, width, height) as capture:
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
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

                    result = _process_frame(cv2, hands, frame)
                    selected = _select_hand(result, handedness)
                    if selected:
                        detected_handedness, landmarks = selected
                        recent_frames.append(landmarks)
                        if len(recent_frames) > max(1, capture_frames):
                            recent_frames.pop(0)

                    status = (
                        f"{label}: space capture | q save | "
                        f"captures {capture_count} | buffer {len(recent_frames)}/{capture_frames}"
                    )
                    if not selected:
                        status = f"{label}: no hand | space waits | q save"
                    _show_status(cv2, frame, status)
                    key = _read_key(cv2)

                    if key in {27, ord("q")}:
                        break
                    if key == ord(" "):
                        if not recent_frames:
                            continue
                        frames.extend(list(recent_frames))
                        capture_count += 1
                        _show_status(
                            cv2,
                            frame,
                            f"{label}: captured {len(recent_frames)} frames",
                        )
                        _read_key(cv2, delay_ms=250)
                cv2.destroyWindow("Pose2OSC")
            else:
                start = time.monotonic()
                while time.monotonic() - start < seconds:
                    ok, frame = capture.read()
                    if not ok:
                        continue
                    result = _process_frame(cv2, hands, frame)
                    selected = _select_hand(result, handedness)
                    if selected:
                        detected_handedness, landmarks = selected
                        frames.append(landmarks)

                    if show:
                        remaining = max(0.0, seconds - (time.monotonic() - start))
                        _show_status(cv2, frame, f"enrolling {label}: {remaining:0.1f}s")
                        if _should_stop(cv2, show):
                            break
        finally:
            hands.close()

    if not frames:
        raise RuntimeError("no hand landmarks were captured during enrollment")

    if replace:
        model.remove_label(label)

    sample_count = model.add_samples(
        label,
        frames,
        handedness=detected_handedness or handedness,
        metadata={
            "capture_mode": "spacebar" if interactive else "timed",
            "capture_count": capture_count if interactive else 1,
            "capture_frames": capture_frames if interactive else len(frames),
            "capture_seconds": seconds if not interactive else None,
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
            max_num_hands=1,
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
                selected = _select_hand(result, handedness)
                now_ms = int(time.time() * 1000)

                if not selected:
                    prediction = _unknown_prediction(model)
                    state = tracker.update(prediction)
                    _send_prediction(client, prediction, state, osc_cfg)
                    _send_no_hand(client, now_ms, frame_index)
                    if show:
                        _show_status(cv2, frame, "no hand")
                    if _should_stop(cv2, show):
                        break
                    continue

                detected_handedness, landmarks = selected
                hand_key = detected_handedness.lower()
                prediction = model.predict(landmarks, detected_handedness)
                state = tracker.update(prediction)

                client.send_message("/pose2osc/hand/visible", 1)
                _send_landmarks(client, hand_key, landmarks, osc_cfg)
                _send_prediction(client, prediction, state, osc_cfg)
                client.send_message("/pose2osc/frame", [frame_index, now_ms])

                if show:
                    label = state.active_label or "none"
                    _show_status(cv2, frame, f"{label} {prediction.confidence:0.2f}")
                if _should_stop(cv2, show):
                    break
        finally:
            hands.close()


def _load_or_new_model(path: str) -> GestureModel:
    from pathlib import Path

    model_path = Path(path)
    if model_path.exists():
        return GestureModel.load(model_path)
    return GestureModel()


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


def _select_hand(
    result: Any,
    desired_handedness: str | None,
) -> tuple[str, list[tuple[float, float, float]]] | None:
    if not result.multi_hand_landmarks:
        return None

    handedness_labels = _handedness_labels(result.multi_handedness)
    for index, hand_landmarks in enumerate(result.multi_hand_landmarks):
        label = handedness_labels[index] if index < len(handedness_labels) else "unknown"
        if desired_handedness and desired_handedness.lower() != "any":
            if label.lower() != desired_handedness.lower():
                continue
        return (
            label,
            [
                (float(landmark.x), float(landmark.y), float(landmark.z))
                for landmark in hand_landmarks.landmark
            ],
        )
    return None


def _handedness_labels(multi_handedness: Iterable[Any] | None) -> list[str]:
    labels: list[str] = []
    if not multi_handedness:
        return labels
    for item in multi_handedness:
        try:
            labels.append(str(item.classification[0].label))
        except (AttributeError, IndexError):
            labels.append("unknown")
    return labels


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


def _show_status(cv2: Any, frame: Any, text: str) -> None:
    preview = cv2.flip(frame, 1)
    cv2.rectangle(preview, (8, 8), (min(920, preview.shape[1] - 8), 48), (0, 0, 0), -1)
    cv2.putText(
        preview,
        text,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("Pose2OSC", preview)


def _read_key(cv2: Any, delay_ms: int = 1) -> int:
    return cv2.waitKey(delay_ms) & 0xFF


def _should_stop(cv2: Any, show: bool) -> bool:
    if not show:
        return False
    key = _read_key(cv2)
    return key in {27, ord("q")}
