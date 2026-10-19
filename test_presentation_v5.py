import unittest
from pathlib import Path
import numpy as np
from datetime import datetime, timezone
from integrated_pipeline import make_dual_window, process_dual, train_models

class PresentationTests(unittest.TestCase):
    def test_waveform_has_noise_before_and_after_pulse(self):
        rng = np.random.default_rng(2026)
        signals, _ = make_dual_window(rng, demo=True)
        bundles = train_models(samples_per_class=20)
        _, captures = process_dual(signals, bundles, datetime.now(timezone.utc), 1)
        self.assertTrue(captures)
        for w in captures.values():
            a = np.asarray(w['samples'])
            self.assertGreater(np.std(a[:100]), 0.001)
            self.assertGreater(np.std(a[-100:]), 0.001)
    def test_dashboard_uses_millisecond_timestamps(self):
        html = Path(__file__).with_name('dashboard.html').read_text()
        self.assertIn('e.timestamp.substring(11,23)', html)
        self.assertNotIn('Demo thresholds; not a field alarm', html)

if __name__ == '__main__':
    unittest.main()
