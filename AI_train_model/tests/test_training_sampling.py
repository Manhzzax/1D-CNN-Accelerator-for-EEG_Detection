"""Tests for source patient-group-balanced training sampling."""

import unittest

import numpy as np

from src.training_sampling import patient_group_balanced_weights


class PatientGroupBalancedSamplingTests(unittest.TestCase):
    def test_every_observed_class_patient_group_stratum_has_unit_weight_mass(self):
        labels = np.asarray([0, 0, 0, 1, 1, 0, 1], dtype=np.int64)
        groups = np.asarray([0, 0, 1, 0, 0, 1, 1], dtype=np.int64)
        importance = np.asarray([1, 3, 2, 1, 1, 4, 2], dtype=np.float64)
        weights, strata = patient_group_balanced_weights(labels, groups, importance)

        self.assertEqual(len(strata), 4)
        for class_index in (0, 1):
            for group_index in (0, 1):
                mask = (labels == class_index) & (groups == group_index)
                self.assertAlmostEqual(float(weights[mask].sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
