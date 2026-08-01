"""Tests for the training-only supervised contrastive objective."""

import unittest

import torch

from src.supervised_contrastive import supervised_contrastive_loss


class SupervisedContrastiveTests(unittest.TestCase):
    def test_loss_is_finite_and_differentiable(self):
        features = torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], requires_grad=True
        )
        labels = torch.tensor([0, 0, 1, 1])
        loss = supervised_contrastive_loss(features, labels, temperature=0.1)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_grouped_embeddings_outperform_mixed_embeddings(self):
        labels = torch.tensor([0, 0, 1, 1])
        grouped = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
        mixed = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        self.assertLess(
            float(supervised_contrastive_loss(grouped, labels)),
            float(supervised_contrastive_loss(mixed, labels)),
        )


if __name__ == "__main__":
    unittest.main()
