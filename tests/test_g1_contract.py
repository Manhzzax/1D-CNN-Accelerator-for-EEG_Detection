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

    def test_execution_model_requires_an_approved_sha_and_server_only_data_access(self):
        model = Path("docs/EXECUTION_MODEL.md").read_text(encoding="utf-8")
        self.assertIn("approved commit SHA", model)
        self.assertIn("SERVER-02", model)
        self.assertIn("no access to SERVER-02", model)

    def test_errata_keeps_montage_as_a_pending_g1b_decision(self):
        errata = Path("docs/spec/ERRATA.md").read_text(encoding="utf-8")
        self.assertIn("141 `RECORDS-WITH-SEIZURES`", errata)
        self.assertIn("pending proposal", errata)

    def test_server_script_rejects_dirty_worktree_before_running_tests(self):
        script = Path("scripts/run_g1_audit.sh").read_text(encoding="utf-8")
        self.assertIn("git status --porcelain", script)
        self.assertLess(script.index("git status --porcelain"), script.index("unittest discover"))
        self.assertIn("Refusing G1A audit", script)

    def test_server_prerequisites_are_documented_without_auto_installation(self):
        execution_model = Path("docs/EXECUTION_MODEL.md").read_text(encoding="utf-8")
        self.assertIn("Python\n3.11 or newer", execution_model)
        self.assertIn("pyedflib", execution_model)
        self.assertIn("wfdb", execution_model)
        self.assertIn("do not install packages", execution_model)
