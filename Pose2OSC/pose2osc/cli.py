"""Command line entry points for Pose2OSC."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .live import OscConfig, enroll_from_camera, run_osc_camera
from .recognizer import GestureModel, StateConfig


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "enroll":
        sample_count = enroll_from_camera(
            label=args.label,
            model_path=args.model,
            seconds=args.seconds,
            capture_frames=args.capture_frames,
            camera=args.camera,
            max_samples=args.max_samples,
            handedness=args.hand,
            show=args.show,
            timed=args.timed,
            replace=args.replace,
            width=args.width,
            height=args.height,
        )
        mode = "replaced" if args.replace else "enrolled"
        print(f"{mode} {sample_count} samples for '{args.label}' in {args.model}")
        return 0

    if args.command == "run":
        run_osc_camera(
            model_path=args.model,
            osc=OscConfig(
                host=args.host,
                port=args.port,
                split_axis_messages=args.split_axes,
                send_landmark_vectors=not args.no_landmark_vectors,
                send_unknown_predictions=args.send_unknown,
            ),
            camera=args.camera,
            handedness=args.hand,
            show=args.show,
            width=args.width,
            height=args.height,
            state_config=StateConfig(
                enter_frames=args.enter_frames,
                exit_frames=args.exit_frames,
                switch_frames=args.switch_frames,
            ),
        )
        return 0

    if args.command == "inspect":
        model = _load_existing_model(args.model)
        counts = {
            label: sum(1 for sample in model.samples if sample.label == label)
            for label in model.labels
        }
        print(json.dumps({
            "metadata": model.metadata,
            "labels": counts,
            "label_metadata": model.label_metadata,
            "thresholds": model.thresholds,
        }, indent=2))
        return 0

    if args.command == "remove":
        model = _load_existing_model(args.model)
        removed = model.remove_label(args.label)
        model.save(args.model)
        print(f"removed {removed} samples for '{args.label}' from {args.model}")
        return 0

    parser.print_help()
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pose2osc",
        description="Lightweight live hand-gesture enrollment and OSC runtime.",
    )
    subparsers = parser.add_subparsers(dest="command")

    enroll = subparsers.add_parser("enroll", help="record a held gesture from the camera")
    _add_camera_args(enroll)
    enroll.add_argument("label", help="gesture label, e.g. delay_hold")
    _add_manifest_arg(enroll)
    enroll.add_argument("--seconds", type=float, default=2.0, help="timed capture duration")
    enroll.add_argument(
        "--capture-frames",
        type=int,
        default=45,
        help="recent hand frames saved each time space is pressed with --show",
    )
    enroll.add_argument(
        "--max-samples",
        type=int,
        default=64,
        help="cap stored frames per enrollment for faster KNN",
    )
    enroll.add_argument(
        "--timed",
        action="store_true",
        help="use the old countdown capture even when --show is enabled",
    )
    update_mode = enroll.add_mutually_exclusive_group()
    update_mode.add_argument(
        "--append",
        dest="replace",
        action="store_false",
        help="append captured samples to an existing label",
    )
    update_mode.add_argument(
        "--replace",
        dest="replace",
        action="store_true",
        help="remove the existing label before saving this capture",
    )
    enroll.set_defaults(replace=False)

    run = subparsers.add_parser("run", help="stream live OSC predictions and landmarks")
    _add_camera_args(run)
    _add_manifest_arg(run)
    run.add_argument("--host", default="127.0.0.1", help="OSC host")
    run.add_argument("--port", type=int, default=8000, help="OSC UDP port")
    run.add_argument("--split-axes", action="store_true", help="send /x /y /z OSC paths")
    run.add_argument(
        "--no-landmark-vectors",
        action="store_true",
        help="skip /hand/<hand>/<landmark> x y z vector messages",
    )
    run.add_argument("--send-unknown", action="store_true", help="emit unknown predictions")
    run.add_argument("--enter-frames", type=int, default=1, help="frames before gesture enter")
    run.add_argument("--exit-frames", type=int, default=1, help="frames before gesture exit")
    run.add_argument("--switch-frames", type=int, default=1, help="frames before active switch")

    inspect = subparsers.add_parser("inspect", help="print labels and thresholds")
    _add_manifest_arg(inspect)

    remove = subparsers.add_parser("remove", help="remove all samples for one label")
    remove.add_argument("label", help="gesture label to remove")
    _add_manifest_arg(remove)

    return parser


def _add_manifest_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--manifest",
        "--model",
        dest="model",
        default="models/gestures.json",
        help="portable gesture manifest JSON",
    )


def _add_camera_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--hand", default="Any", help="Right, Left, or Any")
    parser.add_argument("--show", action="store_true", help="show debug camera window")
    parser.add_argument("--width", type=int, default=None, help="requested camera width")
    parser.add_argument("--height", type=int, default=None, help="requested camera height")


def _load_existing_model(path: str) -> GestureModel:
    try:
        return GestureModel.load(path)
    except FileNotFoundError as exc:
        raise SystemExit(f"model file not found: {path}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
