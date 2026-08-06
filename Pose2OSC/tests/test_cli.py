import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from pose2osc.cli import main
from pose2osc.recognizer import GestureModel
from tests.test_features import open_hand


class CliTests(unittest.TestCase):
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
