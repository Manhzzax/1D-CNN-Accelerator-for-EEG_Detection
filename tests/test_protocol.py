import unittest

from eegkv.protocol import CANONICAL_CHANNELS, canonical_channel_indices, patient_group


class ProtocolTests(unittest.TestCase):
    def test_chb01_and_chb21_are_one_participant(self):
        self.assertEqual(patient_group("chb01"), "subject_01_21")
        self.assertEqual(patient_group("chb21"), "subject_01_21")
        self.assertNotEqual(patient_group("chb02"), "subject_01_21")

    def test_strict_montage_reorders_only_complete_named_channels(self):
        labels = list(reversed(CANONICAL_CHANNELS))
        indices = canonical_channel_indices(labels)
        self.assertEqual([labels[index] for index in indices], list(CANONICAL_CHANNELS))
        self.assertIsNone(canonical_channel_indices(labels[:-1]))
        self.assertIsNone(canonical_channel_indices(["FP1-F7"] + list(CANONICAL_CHANNELS[1:])))

