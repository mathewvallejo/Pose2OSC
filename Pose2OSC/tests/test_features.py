import unittest

from pose2osc.features import FeatureConfig, extract_features, normalize_landmarks


def open_hand(offset=(0.0, 0.0, 0.0), scale=1.0):
    ox, oy, oz = offset
    base = [
        (0.00, 0.00, 0.00),
        (-0.20, -0.12, 0.00),
        (-0.34, -0.28, 0.00),
        (-0.45, -0.43, 0.00),
        (-0.58, -0.58, 0.00),
        (-0.18, -0.38, 0.00),
        (-0.22, -0.68, 0.00),
        (-0.23, -0.92, 0.00),
        (-0.24, -1.16, 0.00),
        (0.00, -0.42, 0.00),
        (0.00, -0.76, 0.00),
        (0.00, -1.02, 0.00),
        (0.00, -1.30, 0.00),
        (0.18, -0.38, 0.00),
        (0.22, -0.68, 0.00),
        (0.23, -0.92, 0.00),
        (0.24, -1.16, 0.00),
        (0.35, -0.30, 0.00),
        (0.47, -0.54, 0.00),
        (0.57, -0.72, 0.00),
        (0.68, -0.92, 0.00),
    ]
    return [(ox + x * scale, oy + y * scale, oz + z * scale) for x, y, z in base]


class FeatureTests(unittest.TestCase):
    def test_normalized_landmarks_ignore_translation_and_scale(self):
        first = normalize_landmarks(open_hand())
        second = normalize_landmarks(open_hand(offset=(0.62, 0.41, -0.2), scale=0.55))

        for point_a, point_b in zip(first, second):
            for value_a, value_b in zip(point_a, point_b):
                self.assertAlmostEqual(value_a, value_b, places=6)

    def test_features_ignore_translation_and_scale(self):
        config = FeatureConfig()
        first = extract_features(open_hand(), config)
        second = extract_features(open_hand(offset=(-0.3, 0.7, 0.2), scale=1.8), config)

        self.assertEqual(len(first), len(second))
        distance = sum((a - b) ** 2 for a, b in zip(first, second)) ** 0.5
        self.assertLess(distance, 1e-6)


if __name__ == "__main__":
    unittest.main()
