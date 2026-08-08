import unittest
from pathlib import Path


class G1ContractTests(unittest.TestCase):
    def test_contract_explicitly_forbids_cross_recording_windows_and_splits(self):
        contract = (Path(__file__).resolve().parents[1] / "docs/q1_data_contract.md").read_text(encoding="utf-8")
        self.assertIn("separate temporal object", contract)
        self.assertIn("no train/validation/test split", contract)

    def test_source_does_not_copy_or_write_raw_edf_data(self):
        source = (Path(__file__).resolve().parents[1] / "src/g1_audit.py").read_text(encoding="utf-8")
        self.assertNotIn("copy2(", source)
        self.assertNotIn("copyfile(", source)
        self.assertNotIn("np.save", source)

