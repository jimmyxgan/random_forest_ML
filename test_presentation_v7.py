import numpy as np
from integrated_pipeline import _clean_display_wave

def test_quiet_continuous_baseline_and_pd_burst():
    rng=np.random.default_rng(9)
    n=1024
    t=np.arange(n)
    raw=rng.normal(0,.09,n)
    raw+=np.where(t>=420, np.exp(-(t-420)/80)*np.sin(2*np.pi*.11*(t-420)),0)
    out=_clean_display_wave(raw,'test-event')
    before=np.std(out[:200]); during=np.max(np.abs(out[420:530])); after=np.std(out[800:])
    assert before>0 and after>0
    assert during>20*before
    assert during>20*after
    assert np.all(np.isfinite(out))
