import numpy as np
from integrated_pipeline import _clean_display_wave

def test_pulse_moves_left_without_wraparound():
    n=1024
    t=np.arange(n)
    raw=np.zeros(n)
    raw[630:850]=np.exp(-np.arange(220)/65)*np.sin(2*np.pi*.12*np.arange(220))
    out=_clean_display_wave(raw,'shift-test')
    energy=np.convolve(out*out,np.ones(17)/17,mode='same')
    assert 320 <= np.argmax(energy) <= 440
    assert np.std(out[:150])>0
    assert np.max(np.abs(out[950:])) < .15*np.max(np.abs(out))

def test_all_sensor_labels():
    html=open('dashboard.html',encoding='utf8').read()
    assert 'txt(labelX,labelY,item.id' in html
    assert 'const elbow=' not in html
    assert "for(const item of items)" in html
