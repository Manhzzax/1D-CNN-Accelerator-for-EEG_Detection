"""Tests for the source-patient GroupDRO objective."""

import unittest

import torch

from src.group_dro import GroupDROObjective


class GroupDROTests(unittest.TestCase):
    def test_higher_loss_group_receives_higher_next_weight(self):
        objective = GroupDROObjective(group_count=2, eta=0.5)
        losses = torch.tensor([1.0, 1.0, 3.0, 3.0], requires_grad=True)
        groups = torch.tensor([0, 0, 1, 1])

        objective(losses, groups).backward()

        weights = objective.weights()
        self.assertGreater(float(weights[1]), float(weights[0]))
        self.assertTrue(torch.all(torch.isfinite(losses.grad)))


if __name__ == "__main__":
    unittest.main()
