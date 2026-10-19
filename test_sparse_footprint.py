import unittest
from datetime import datetime,timezone
import numpy as np
from integrated_pipeline import make_dual_window,process_dual
class SparseFootprintTests(unittest.TestCase):
 def test_synthetic_event_max_four_channels(self):
  sig,_=make_dual_window(np.random.default_rng(11),demo=True)
  events,_=process_dual(sig,None,datetime.now(timezone.utc),1)
  self.assertTrue(events)
  self.assertTrue(all(e['sensor_count']<=4 for e in events))
  self.assertTrue(all(len(set(e['evidence']['sensor_ids']))<=4 for e in events))
if __name__=='__main__':unittest.main()
