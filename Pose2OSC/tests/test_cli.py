import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from pose2osc.cli import _resolve_enroll_labels, build_parser, main
from pose2osc.recognizer import GestureModel
from tests.test_features import open_hand


class CliTests(unittest.TestCase):
    def test_enroll_parser_accepts_multiple_interactive_labels(self):
        args = build_parser().parse_args([
            "enroll",
            "delay_hold",
            "filter_grab",
            "--manifest",
            "manifests/theremin_set.json",
            "--show",
        ])

        self.assertEqual(args.labels, ["delay_hold", "filter_grab"])
        self.assertTrue(args.show)

    def test_enroll_parser_generates_generic_labels(self):
        parser = build_parser()
        args = parser.parse_args([
            "enroll",
            "--gestures",
            "3",
            "--manifest",
            "manifests/theremin_set.json",
            "--show",
        ])

        self.assertEqual(_resolve_enroll_labels(parser, args), [
            "gesture_1",
            "gesture_2",
            "gesture_3",
        ])

    def test_multi_label_enroll_requires_interactive_show(self):
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            main([
                "enroll",
                "--gestures",
                "2",
                "--manifest",
                "manifests/theremin_set.json",
            ])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("multiple gesture labels require --show", stderr.getvalue())

    def test_inspect_prints_manifest_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "gestures.json"
            model = GestureModel(metadata={"name": "test_set"})
            model.add_samples("open", [open_hand(), open_hand(scale=1.01)])
            model.save(manifest)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["inspect", "--manifest", str(manifest)])

            self.assertEqual(exit_code, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["metadata"]["name"], "test_set")
            self.assertEqual(summary["labels"]["open"], 2)
            self.assertEqual(summary["label_metadata"]["open"]["sample_count"], 2)
            self.assertIn("open", summary["thresholds"])


if __name__ == "__main__":
    unittest.main()
