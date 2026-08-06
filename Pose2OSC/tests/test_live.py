import types
import unittest

from pose2osc.live import OscConfig, _model_landmarks, _select_hands, _send_prediction
from pose2osc.recognizer import Prediction, StateUpdate


class DummyClient:
    def __init__(self):
        self.messages = []

    def send_message(self, path, value):
        self.messages.append((path, value))


def prediction(label="open", accepted=True, confidence=0.82):
    return Prediction(
        label=label if accepted else None,
        accepted=accepted,
        distance=0.1,
        confidence=confidence,
        vote_confidence=1.0 if accepted else 0.0,
        distance_confidence=confidence,
        threshold=0.3,
        votes={label: 1.0} if label else {},
    )


class LiveOscTests(unittest.TestCase):
    def test_enter_sends_gate_and_trigger(self):
        client = DummyClient()
        pred = prediction("open")
        state = StateUpdate(
            active_label="open",
            previous_label=None,
            event="enter",
            prediction=pred,
            active=True,
        )

        _send_prediction(client, pred, state, OscConfig())

        self.assertIn(("/pose2osc/gesture/open/active", 1), client.messages)
        self.assertIn(("/pose2osc/gesture/open/trigger", 1), client.messages)

    def test_exit_sends_gate_off_and_global_none(self):
        client = DummyClient()
        pred = prediction(label=None, accepted=False, confidence=0.0)
        state = StateUpdate(
            active_label=None,
            previous_label="open",
            event="exit",
            prediction=pred,
            active=False,
        )

        _send_prediction(client, pred, state, OscConfig())

        self.assertIn(("/pose2osc/gesture/open/active", 0), client.messages)
        self.assertIn(("/pose2osc/gesture/open/confidence", 0.0), client.messages)
        self.assertIn(("/pose2osc/state/active", ["none", 0.0]), client.messages)

    def test_select_both_hands_orders_right_then_left_for_model(self):
        left = [(1.0, 0.0, 0.0)] * 21
        right = [(2.0, 0.0, 0.0)] * 21
        result = fake_result([("Left", left), ("Right", right)])

        selected = _select_hands(result, "Both")
        model_landmarks = _model_landmarks(selected)

        self.assertEqual([label for label, _ in selected], ["Right", "Left"])
        self.assertEqual(model_landmarks[:21], right)
        self.assertEqual(model_landmarks[21:], left)


def fake_result(hands):
    return types.SimpleNamespace(
        multi_hand_landmarks=[
            types.SimpleNamespace(
                landmark=[
                    types.SimpleNamespace(x=x, y=y, z=z)
                    for x, y, z in landmarks
                ]
            )
            for _, landmarks in hands
        ],
        multi_handedness=[
            types.SimpleNamespace(
                classification=[types.SimpleNamespace(label=label)]
            )
            for label, _ in hands
        ],
    )


if __name__ == "__main__":
    unittest.main()
