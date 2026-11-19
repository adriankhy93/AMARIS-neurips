from types import SimpleNamespace
import unittest

import torch

from utils.val import validate


class RecordingModel:
    def __init__(self):
        self.parameter = torch.nn.Parameter(torch.zeros(()))
        self.labels_seen = []

    def parameters(self):
        return iter((self.parameter,))

    def eval(self):
        return self

    def __call__(self, *, labels, **_):
        self.labels_seen.append(labels.detach().clone())
        return SimpleNamespace(loss=torch.tensor(2.0))


class ValidationTests(unittest.TestCase):
    def test_uses_unmasked_labels_when_no_assistant_tokens_exist(self):
        model = RecordingModel()
        data = [{
            "input_ids": torch.tensor([[1, 2, 3]]),
            "position_ids": torch.tensor([[0, 1, 2]]),
            "assistant_mask": torch.tensor([[0, 0, 0]]),
            "labels": torch.tensor([[1, 2, 3]]),
        }]

        self.assertEqual(validate(model, data), 2.0)
        self.assertTrue(torch.equal(model.labels_seen[0], data[0]["labels"]))


if __name__ == "__main__":
    unittest.main()
