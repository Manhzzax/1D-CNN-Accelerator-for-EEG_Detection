import unittest
from pathlib import Path


class G1ContractTests(unittest.TestCase):
    def test_active_contract_has_no_montage_or_split_selection(self):
        contract = Path("docs/PROJECT_CONTRACT.md").read_text(encoding="utf-8")
        self.assertIn("separate temporal object", contract)
        self.assertIn("no 17/18/19-channel", contract)
        self.assertIn("creates no split", contract)

    def test_g1_source_never_copies_raw_or_writes_prepared_arrays(self):
        source = Path("src/eegkv/audit.py").read_text(encoding="utf-8")
        self.assertNotIn("copyfile(", source)
        self.assertNotIn("copy2(", source)
        self.assertNotIn("np.save", source)
