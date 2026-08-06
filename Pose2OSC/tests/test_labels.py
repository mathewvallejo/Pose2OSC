import unittest

from pose2osc.labels import (
    bgr_from_hex,
    generated_gesture_labels,
    label_style,
)


class LabelTests(unittest.TestCase):
    def test_generated_gesture_labels_are_osc_safe(self):
        self.assertEqual(
            generated_gesture_labels(3),
            ["gesture_1", "gesture_2", "gesture_3"],
        )

    def test_generated_gesture_labels_can_start_later(self):
        self.assertEqual(
            generated_gesture_labels(2, start_index=4),
            ["gesture_4", "gesture_5"],
        )
        self.assertEqual(label_style("gesture_4").display_label, "Gesture 4")
        self.assertEqual(label_style("gesture_4").color, "#35D07F")

    def test_generated_label_displays_as_human_gesture_name(self):
        style = label_style("gesture_2")

        self.assertEqual(style.display_label, "Gesture 2")
        self.assertEqual(style.color, "#FFB000")

    def test_manifest_metadata_overrides_default_style(self):
        style = label_style(
            "gesture_2",
            {"display_label": "Bow", "color": "#123abc"},
        )

        self.assertEqual(style.display_label, "Bow")
        self.assertEqual(style.color, "#123ABC")
        self.assertEqual(bgr_from_hex(style.color), (0xBC, 0x3A, 0x12))


if __name__ == "__main__":
    unittest.main()
