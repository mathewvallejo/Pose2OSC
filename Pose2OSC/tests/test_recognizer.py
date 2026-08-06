import unittest

from pose2osc.recognizer import GestureModel, GestureStateTracker, StateConfig
from tests.test_features import open_hand


def fist(offset=(0.0, 0.0, 0.0), scale=1.0):
    hand = open_hand(offset=(0.0, 0.0, 0.0), scale=1.0)
    curled = list(hand)
    for index in (8, 12, 16, 20):
        x, _, z = curled[index]
        curled[index] = (x * 0.45, -0.50, z)
    ox, oy, oz = offset
    return [(ox + x * scale, oy + y * scale, oz + z * scale) for x, y, z in curled]


class RecognizerTests(unittest.TestCase):
    def test_knn_recognizes_same_shape_anywhere_in_frame(self):
        model = GestureModel()
        model.add_samples("open", [open_hand(scale=1.0), open_hand(scale=1.05)])
        model.add_samples("fist", [fist(scale=1.0), fist(scale=0.95)])

        prediction = model.predict(open_hand(offset=(0.4, -0.2, 0.1), scale=0.6))

        self.assertTrue(prediction.accepted)
        self.assertEqual(prediction.label, "open")

    def test_state_tracker_enters_immediately(self):
        model = GestureModel()
        model.add_samples("open", [open_hand(), open_hand(scale=1.01)])
        tracker = GestureStateTracker(StateConfig(enter_frames=1, exit_frames=1))

        prediction = model.predict(open_hand(offset=(0.5, 0.2, 0.0), scale=0.8))
        update = tracker.update(prediction)

        self.assertEqual(update.event, "enter")
        self.assertEqual(update.active_label, "open")
        self.assertIsNone(update.previous_label)

    def test_state_tracker_reports_previous_label_on_switch(self):
        model = GestureModel()
        model.add_samples("open", [open_hand(), open_hand(scale=1.01)])
        model.add_samples("fist", [fist(), fist(scale=1.01)])
        tracker = GestureStateTracker(StateConfig(enter_frames=1, exit_frames=1, switch_frames=1))

        tracker.update(model.predict(open_hand()))
        update = tracker.update(model.predict(fist()))

        self.assertEqual(update.event, "switch")
        self.assertEqual(update.active_label, "fist")
        self.assertEqual(update.previous_label, "open")

    def test_state_tracker_reports_previous_label_on_exit(self):
        model = GestureModel()
        model.add_samples("open", [open_hand(), open_hand(scale=1.01)])
        tracker = GestureStateTracker(StateConfig(enter_frames=1, exit_frames=1))

        tracker.update(model.predict(open_hand()))
        update = tracker.update(model.predict(fist(offset=(4.0, 4.0, 0.0))))

        self.assertEqual(update.event, "exit")
        self.assertIsNone(update.active_label)
        self.assertEqual(update.previous_label, "open")

    def test_manifest_metadata_round_trips(self):
        model = GestureModel(metadata={"name": "home_setup"})
        model.add_samples(
            "open",
            [open_hand(), open_hand(scale=1.01)],
            handedness="Right",
            metadata={"notes": "flat hand"},
        )
        restored = GestureModel.from_dict(model.to_dict())

        self.assertEqual(restored.metadata["name"], "home_setup")
        self.assertEqual(restored.label_metadata["open"]["handedness"], "Right")
        self.assertEqual(restored.label_metadata["open"]["sample_count"], 2)
        self.assertEqual(restored.label_metadata["open"]["hand_modes"]["Right"], 2)

    def test_prediction_respects_manifest_hand_mode(self):
        model = GestureModel()
        model.add_samples("right_open", [open_hand(), open_hand(scale=1.01)], handedness="Right")

        prediction = model.predict(open_hand(), handedness="Left")

        self.assertFalse(prediction.accepted)
        self.assertIsNone(prediction.label)

    def test_threshold_fitting_allows_one_and_two_hand_samples_for_same_label(self):
        model = GestureModel()
        two_hand = open_hand() + open_hand(offset=(0.5, 0.0, 0.0), scale=0.95)
        two_hand_variant = open_hand(scale=1.01) + open_hand(offset=(0.48, 0.0, 0.0), scale=0.96)

        model.add_samples("hold", [open_hand(), open_hand(scale=1.01)], handedness="Right")
        model.add_samples("hold", [two_hand, two_hand_variant], handedness="Both")

        self.assertIn("hold", model.thresholds)
        self.assertTrue(model.predict(open_hand(scale=1.0), handedness="Right").accepted)
        self.assertTrue(model.predict(two_hand, handedness="Both").accepted)


if __name__ == "__main__":
    unittest.main()
