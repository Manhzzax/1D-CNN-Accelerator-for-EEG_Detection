"""Tests for the training-only EEG augmentation contract."""

import unittest

import torch

from src.eeg_augmentation import mild_eeg_augmentation


class MildEEGAugmentationTests(unittest.TestCase):
    def test_zero_augmentation_is_identity(self):
        inputs = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        outputs = mild_eeg_augmentation(inputs, gain_delta=0.0, noise_std=0.0)
        self.assertTrue(torch.equal(inputs, outputs))

    def test_gain_is_shared_across_channels_and_time(self):
        torch.manual_seed(7)
        inputs = torch.ones((2, 3, 4), dtype=torch.float32)
        outputs = mild_eeg_augmentation(inputs, gain_delta=0.1, noise_std=0.0)
        self.assertTrue(torch.allclose(outputs, outputs[:, :1, :1].expand_as(outputs)))
        self.assertTrue(torch.all(outputs >= 0.9))
        self.assertTrue(torch.all(outputs <= 1.1))


if __name__ == "__main__":
    unittest.main()
