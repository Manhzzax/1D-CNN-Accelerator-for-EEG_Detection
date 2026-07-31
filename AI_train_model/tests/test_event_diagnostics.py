"""Regression tests for event-diagnostics CSV schemas."""

import unittest

from src.event_diagnostics import PER_RECORDING_FIELDS


class EventDiagnosticsTests(unittest.TestCase):
    def test_per_recording_schema_includes_causal_alarm_timestamp_mode(self):
        self.assertIn("alarm_timestamp_mode", PER_RECORDING_FIELDS)

    def test_per_recording_schema_identifies_the_independent_patient_group(self):
        self.assertIn("patient_group", PER_RECORDING_FIELDS)


if __name__ == "__main__":
    unittest.main()
