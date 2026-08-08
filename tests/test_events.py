import unittest

from eegkv.events import Event, false_positives_per_day, score_events
from eegkv.preprocess import label_window


class EventTests(unittest.TestCase):
    def test_tolerance_and_one_to_one_matching(self):
        result = score_events([Event(100, 140)], [Event(75, 80), Event(110, 120), Event(500, 510)])
        self.assertEqual(result["true_positive_events"], 1)
        self.assertEqual(result["false_positive_events"], 1)
        self.assertEqual(result["false_negative_events"], 0)
        self.assertAlmostEqual(result["event_f1"], 2 / 3)

    def test_long_event_is_split_and_false_alarm_rate_is_scaled(self):
        result = score_events([Event(0, 650)], [Event(10, 20), Event(310, 320), Event(610, 620)])
        self.assertEqual(result["true_positive_events"], 3)
        self.assertAlmostEqual(false_positives_per_day(2, 43200), 4.0)

    def test_window_labels_exclude_guard_band(self):
        intervals = [[100.0, 140.0]]
        self.assertEqual(label_window(110, 114, intervals), 1)
        self.assertEqual(label_window(10, 14, intervals), 0)
        self.assertIsNone(label_window(75, 79, intervals))
