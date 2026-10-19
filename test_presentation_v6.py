from pathlib import Path
import numpy as np
from integrated_pipeline import _clean_display_wave

def test_waveform_low_floor_and_visible_pulse():
    rng=np.random.default_rng(7)
    t=np.arange(1024)
    raw=rng.normal(0,0.05,1024)+2*np.sin(2*np.pi*.09*t)*np.exp(-((t-520)/75)**2)
    out=_clean_display_wave(raw,'test-event')
    assert out.shape==raw.shape
    assert np.std(out[:200])<np.std(raw[:200])
    assert np.std(out[490:550])>3*np.std(out[:200])
    assert np.std(out[:200])>0

def test_six_mics_at_bottom_rim():
    html=Path(__file__).with_name('dashboard.html').read_text()
    assert 'point(Ro,j*Math.PI/3,bottom)' in html
    assert 'items.filter(i=>i.mic)' in html
    assert 'height:45px;min-height:45px' in html
