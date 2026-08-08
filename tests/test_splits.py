import unittest

from eegkv.splits import make_loso_folds


def row(case, group):
    return {"recording_id": f"{case}_01.edf", "case_id": case, "split_group": group, "channel_coverage": "complete"}


class SplitTests(unittest.TestCase):
    def test_loso_has_no_subject_leakage(self):
        rows = [row("chb01", "subject_01_21"), row("chb21", "subject_01_21")]
        rows += [row(f"chb{index:02d}", f"subject_{index:02d}") for index in range(2, 24)]
        folds = make_loso_folds(rows)
        self.assertEqual(len(folds), 23)
        for fold in folds:
            train, validation, test = set(fold["training_subjects"]), set(fold["validation_subjects"]), fold["outer_test_subject"]
            self.assertFalse(train & validation)
            self.assertNotIn(test, train)
            self.assertNotIn(test, validation)
            self.assertEqual(len(validation), 4)

