"""Unit tests for the training-only subject-domain objective."""

import unittest

import torch

from src.model import SeparableEEG1DCNN, SubjectDiscriminator, gradient_reverse


class SubjectAdversarialTests(unittest.TestCase):
    def test_gradient_reversal_preserves_values_and_reverses_gradient(self):
        inputs = torch.tensor([[1.0, -2.0]], requires_grad=True)
        outputs = gradient_reverse(inputs, 0.25)
        self.assertTrue(torch.equal(inputs, outputs))
        outputs.sum().backward()
        self.assertTrue(torch.equal(inputs.grad, torch.tensor([[-0.25, -0.25]])))

    def test_separable_features_support_training_only_domain_head(self):
        model = SeparableEEG1DCNN(17, 2, temporal_filters_per_channel=3)
        model.eval()
        inputs = torch.randn(3, 17, 512)
        with torch.no_grad():
            features = model.forward_features(inputs)
            logits = model(inputs)
            self.assertEqual(features.shape, (3, 32))
            self.assertTrue(torch.allclose(logits, model.classifier(features)))
            self.assertEqual(SubjectDiscriminator(32, 16, 4)(features).shape, (3, 4))


if __name__ == "__main__":
    unittest.main()
