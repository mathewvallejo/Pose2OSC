import unittest

from pose2osc.live import OscConfig, _send_prediction
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


if __name__ == "__main__":
    unittest.main()
