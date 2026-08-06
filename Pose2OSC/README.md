# Pose2OSC

Pose2OSC is a standalone live gesture-control runtime. It does not depend on
the earlier camera preprocessing, GRU autoencoder, or OSC runtime pipeline.

It provides lightweight gesture enrollment and OSC output for a
MediaPipe-controlled Max/MSP instrument. It is designed for the expanded
theremin idea: continuous landmark streams stay available for expressive
control, while user-defined held gestures act as patch states or triggers.

The recognizer is intentionally small:

- no neural network at runtime
- one-frame KNN prediction
- translation and scale invariant hand-shape features
- default `enter_frames=1`, so gestures can enter on the first matching frame
- optional `exit_frames=2` or `3` only if you want more dropout tolerance

## Install

For stable live camera + OSC use, keep Pose2OSC in its own home-folder virtual
environment. This keeps MediaPipe, OpenCV, and NumPy away from the GRU
autoencoder and other project environments.

One-time setup:

```bash
deactivate 2>/dev/null || true
python3.11 -m venv ~/venvs/pose2osc_runtime_env
source ~/venvs/pose2osc_runtime_env/bin/activate
cd "/Volumes/MP_1/GSoC 2026/av_autoencoder/Pose2OSC"
python -m pip install --upgrade pip
python -m pip install -e ".[live]"
```

If `python3.11` is not available, use another Python 3.10+ interpreter in the
first line.

MediaPipe can fail in environments that have NumPy 2 installed alongside
extensions built for NumPy 1. The live install pins `numpy<2` to avoid that
runtime mismatch.

Every new terminal session:

```bash
deactivate 2>/dev/null || true
source ~/venvs/pose2osc_runtime_env/bin/activate
cd "/Volumes/MP_1/GSoC 2026/av_autoencoder/Pose2OSC"
```

Verify the command and live dependencies:

```bash
which pose2osc
pose2osc --help
python -c "import cv2, mediapipe, pythonosc, numpy; print('Pose2OSC OK', numpy.__version__)"
```

## Stage 1: Build A Gesture Manifest

Use enrollment at home or in rehearsal to author a portable gesture manifest.
With `--show`, hold the desired pose and press `space` to capture and save it
to the manifest. Repeat for natural variations, press `n` to move to the next
gesture, and press `q` or `Esc` only when the full gesture set is done.

```bash
pose2osc enroll --gestures 3 --manifest manifests/theremin_set.json --show
```

Each `space` press runs on the live MediaPipe hand-tracking feed, captures the
recent hand window, and writes those samples to the manifest immediately. The
capture UI asks for 5 captures per gesture by default, but you can press
`space` more times for extra variation. Use `n` for the next gesture and `p`
for the previous gesture without closing the camera window. Generated gestures
are stored with OSC-safe labels like `gesture_1`, but the preview displays
performer-friendly names like `Gesture 1`, `Gesture 2`, and `Gesture 3`.

Handedness is detected per capture and stored in the manifest as `Right`,
`Left`, or `Both`. By default, Pose2OSC tracks up to two hands and saves
whatever is visible when `space` is pressed. If you need to force one hand, use
`--hand Right` or `--hand Left`. A label can contain single-hand and two-hand
captures, but forced handedness is usually cleaner when a trigger is meant to be
performed by one hand.

The preview is mirrored for performer-facing use. Pose2OSC corrects
MediaPipe's Left/Right handedness labels before saving the manifest or sending
OSC, so the detected hand should match the performer's physical hand. If you
use a camera feed that is already mirrored before Pose2OSC receives it, add
`--raw-mediapipe-handedness`.

To add more gestures later, launch another capture session with the next
generated numbers:

```bash
pose2osc enroll --gestures 2 --start-index 4 --manifest manifests/theremin_set.json --show
```

To rebuild one label from scratch:

```bash
pose2osc enroll gesture_1 --manifest manifests/theremin_set.json --show --replace
```

Headless timed capture is still available:

```bash
pose2osc enroll gesture_1 --manifest manifests/theremin_set.json --seconds 2 --timed
```

`--model` remains an alias for `--manifest` for older scripts.

## Manifest Storage

Captures are stored in the JSON file passed to `--manifest`, for example:

```text
manifests/theremin_set.json
```

The manifest contains normalized KNN feature samples, not raw video. Each saved
sample stores its gesture label, capture metadata, timestamp, display name,
color, and detected `hand_mode` (`Right`, `Left`, or `Both`). The manifest also
stores per-label thresholds and hand-mode counts so the same file can be loaded
later for performance.

## Stage 2: Run The Performance Runtime

```bash
pose2osc run --manifest manifests/theremin_set.json --host 127.0.0.1 --port 8000 --split-axes --show
```

Runtime also uses automatic handedness. If both hands are visible, Pose2OSC
checks the two-hand pose and each single hand, then uses the strongest accepted
match. Continuous OSC landmarks are sent for every visible hand. The runtime
preview displays each active gesture with the color saved in the manifest.

Lowest-latency state settings are the defaults:

```bash
pose2osc run --enter-frames 1 --exit-frames 1 --switch-frames 1
```

If the active gesture flickers, keep entry immediate and only relax release:

```bash
pose2osc run --enter-frames 1 --exit-frames 2
```

## OSC Shape

Gesture state:

```text
/pose2osc/state/active gesture_1 0.92
/pose2osc/state/event enter gesture_1 0.92
/pose2osc/gesture/gesture_1/active 1
/pose2osc/gesture/gesture_1/trigger 1
/pose2osc/gesture/gesture_1/confidence 0.92
```

On exit, Pose2OSC sends:

```text
/pose2osc/state/event exit gesture_1 0.0
/pose2osc/gesture/gesture_1/active 0
/pose2osc/gesture/gesture_1/confidence 0.0
/pose2osc/state/active none 0.0
```

Continuous landmark vectors:

```text
/pose2osc/hand/right/index_mcp 0.42 0.71 -0.18
```

Axis-specific messages for Max dropdown routing with `--split-axes`:

```text
/pose2osc/hand/right/index_mcp/x 0.42
/pose2osc/hand/right/index_mcp/y 0.71
/pose2osc/hand/right/index_mcp/z -0.18
```

## Model Design

Gesture recognition does not use raw camera position. A frame is converted into
normalized hand-shape features:

1. Translate landmarks so the wrist or palm center is the origin.
2. Divide by palm scale so near/far camera distance has less effect.
3. Optionally mirror left hands into the same canonical shape space.
4. Compare normalized shape vectors with KNN.

That means the same held gesture can be recognized anywhere in the frame.

Raw MediaPipe `x/y/z` values are still sent to Max/MSP for continuous theremin
control. In practice, Max owns the musical mapping:

| Gesture | On Enter | While Held | On Exit |
| --- | --- | --- | --- |
| `gesture_1` | enable delay | `index_mcp/x` -> delay time | disable delay |
| `gesture_2` | enable filter mode | `index_mcp/y` -> cutoff | release |
| `gesture_3` | trigger freeze | `wrist/z` -> grain size | unfreeze |

## Inspect Or Remove Gestures

```bash
pose2osc inspect --manifest manifests/theremin_set.json
pose2osc remove gesture_1 --manifest manifests/theremin_set.json
```

## Notes For Max/MSP

Use the gesture messages as state gates, then route continuous landmark values
inside the matrix. For realtime performance, start with `enter_frames=1` and
only add smoothing or release hysteresis on the Max side where it is musically
useful.
