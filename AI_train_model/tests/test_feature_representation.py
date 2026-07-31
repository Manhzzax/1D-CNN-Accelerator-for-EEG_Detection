"""Feature-contract tests independent of the selected EEG representation."""

import unittest

from src.feature_representation import get_feature_spec


class FeatureRepresentationTests(unittest.TestCase):
    def test_raw_spec_uses_the_configured_window_length(self):
        spec = get_feature_spec({
            "feature_representation": "raw",
            "sample_rate_hz": 256,
            "window_sec": 2,
        })
        self.assertEqual(spec, {"name": "raw", "input_shape": [17, 512]})


if __name__ == "__main__":
    unittest.main()
