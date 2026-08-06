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

Core model code uses only the Python standard library:

```bash
python -m pip install -e .
```

For stable live camera + OSC use, create a dedicated virtual environment from
the repository root. This keeps MediaPipe, OpenCV, and NumPy away from other
project environments.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[live]"
```

If `python3.11` is not available, use a Python 3.10+ interpreter:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[live]"
```

MediaPipe can fail in environments that have NumPy 2 installed alongside
extensions built for NumPy 1. The live install pins `numpy<2` to avoid that
runtime mismatch.

Every new terminal session should activate the environment before running
Pose2OSC:

```bash
source .venv/bin/activate
```

## Stage 1: Build A Gesture Manifest

Use enrollment at home or in rehearsal to author a portable gesture manifest.
With `--show`, hold the desired pose and press `space` to capture it. Repeat
for natural variations, then press `q` or `Esc` to save and finish.

```bash
pose2osc enroll delay_hold --manifest manifests/theremin_set.json --hand Right --show
pose2osc enroll filter_grab --manifest manifests/theremin_set.json --hand Right --show
```

Each `space` press captures the recent hand window, and each enrollment stores
up to 64 frames by default. That keeps KNN fast while still capturing small
variations in the performer's held pose.

To rebuild one label from scratch:

```bash
pose2osc enroll delay_hold --manifest manifests/theremin_set.json --hand Right --show --replace
```

Headless timed capture is still available:

```bash
pose2osc enroll delay_hold --manifest manifests/theremin_set.json --seconds 2 --timed
```

`--model` remains an alias for `--manifest` for older scripts.

## Stage 2: Run The Performance Runtime

```bash
pose2osc run --manifest manifests/theremin_set.json --host 127.0.0.1 --port 8000 --split-axes --show
```

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
/pose2osc/state/active delay_hold 0.92
/pose2osc/state/event enter delay_hold 0.92
/pose2osc/gesture/delay_hold/active 1
/pose2osc/gesture/delay_hold/trigger 1
/pose2osc/gesture/delay_hold/confidence 0.92
```

On exit, Pose2OSC sends:

```text
/pose2osc/state/event exit delay_hold 0.0
/pose2osc/gesture/delay_hold/active 0
/pose2osc/gesture/delay_hold/confidence 0.0
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
| `delay_hold` | enable delay | `index_mcp/x` -> delay time | disable delay |
| `filter_grab` | enable filter mode | `index_mcp/y` -> cutoff | release |
| `freeze_pose` | trigger freeze | `wrist/z` -> grain size | unfreeze |

## Inspect Or Remove Gestures

```bash
pose2osc inspect --manifest manifests/theremin_set.json
pose2osc remove delay_hold --manifest manifests/theremin_set.json
```

## Notes For Max/MSP

Use the gesture messages as state gates, then route continuous landmark values
inside the matrix. For realtime performance, start with `enter_frames=1` and
only add smoothing or release hysteresis on the Max side where it is musically
useful.
