"""Integrated dual-modality demonstration pipeline; all signals are synthetic.
No field sensitivity, calibration, source localization, or IEC certification is claimed.
"""
from dataclasses import replace
import numpy as np
from scipy.signal import resample_poly
from sensors import default_configs
from signal_processing import candidates, preprocess
from features import extract_event_features, sequence_features
from ml_classifier import train, predict
from conventional import evaluate_event
from fusion import decide
from physics import FS, SENSORS, make_window, pulse
# Nine outer-surface AE sensors at three staggered elevations; six bottom microphones.
AE_HEIGHTS=(.25,.50,.75)
AE_ANGLES=((0,120,240),(40,160,280),(80,200,320))
MIC_ANGLES=(0,60,120,180,240,300)
AE_POSITIONS=[(angle,height) for height,angles in zip(AE_HEIGHTS,AE_ANGLES) for angle in angles]
MIC_POSITIONS=[(angle,-.08) for angle in MIC_ANGLES]  # illustrative below-rim air gap
AE_IDS=[f'AE-{i+1:02d}' for i in range(len(AE_POSITIONS))]
MIC_IDS=[f'MIC-{i+1:02d}' for i in range(len(MIC_POSITIONS))]
CHANNELS=AE_IDS+MIC_IDS
CONFIGS=default_configs()

def angular_distance(a,b):
    return abs((a-b+180)%360-180)

def event_wave(rng, modality, kind, config, size=1024):
    fs=config.sample_rate
    t=np.arange(size)/fs
    if modality=='ae':
        f=rng.uniform(35e3,85e3) if kind=='pd' else rng.uniform(22e3,96e3)
        tau=rng.uniform(.00012,.00065) if kind=='pd' else rng.uniform(.0004,.0015)
    else:
        f=rng.choice([rng.uniform(5e3,10e3),rng.uniform(14e3,19e3)]) if kind=='pd' else rng.uniform(2e3,19e3)
        tau=rng.uniform(.0003,.0018) if kind=='pd' else rng.uniform(.001,.004)
    onset=.0006
    d=np.maximum(0,t-onset)
    wave=np.sin(2*np.pi*f*d)*np.exp(-d/tau)*(t>=onset)
    if modality=='mic' and kind=='pd':
        wave+=.3*np.sin(2*np.pi*7e3*d)*np.exp(-d/(tau*1.5))*(t>=onset)
    return wave

def train_models(seed=42, samples_per_class=180):
    rng=np.random.default_rng(seed)
    bundles={}
    for key,config in CONFIGS.items():
        xs=[];ys=[]
        for label,kind in [(1,'pd'),(0,'impact'),(0,'vibration')]:
            for _ in range(samples_per_class):
                x=event_wave(rng,key,kind,config)
                x=rng.uniform(.1,.9)*x+rng.normal(0,rng.uniform(.003,.04),len(x))
                xs.append(x);ys.append(label)
        bundles[key]=train(xs,ys,config,seed)
        bundles[key]['training_origin']='synthetic damped oscillators only; no field validation'
    return bundles

def make_dual_window(rng,scenario='nominal',deterioration=.5,demo=False):
    """Generate nine outer-surface AE and six below-rim microphone channels.
    Propagation gains and delays are illustrative, not calibrated transfer functions.
    """
    base,truth=make_window(rng,scenario,deterioration,force_kind='pd' if demo else None)
    ae_fs=CONFIGS['ae'].sample_rate;mic_fs=CONFIGS['mic'].sample_rate
    n_ae=round(base.shape[1]*ae_fs/FS);n_mic=round(base.shape[1]*mic_fs/FS)
    ae=rng.normal(0,.006 if scenario=='nominal' else .012,(len(AE_IDS),n_ae))
    mic=rng.normal(0,.008 if scenario=='nominal' else .018,(len(MIC_IDS),n_mic))
    for ev in truth:
        source_az=float(rng.uniform(0,360))
        source_z=float(rng.uniform(.1,.9))
        # A sparse, local acoustic footprint: select at most four nearby channels
        # across BOTH technologies, rather than lighting up the entire reactor.
        def proximity(pos, modality):
            angle,height=pos
            arc=angular_distance(angle,source_az)/180
            return 1.5*arc + (abs(height-source_z) if modality=='ae' else .60*(source_z+.08))
        ranked=sorted([(proximity(pos,'ae'), 'ae',i) for i,pos in enumerate(AE_POSITIONS)] +
                      [(proximity(pos,'mic'), 'mic',i) for i,pos in enumerate(MIC_POSITIONS)])
        # Include the nearest microphone and nearest AE when available; fill the
        # remaining two slots with the closest channels of either technology.
        chosen={('ae',min(range(len(AE_POSITIONS)),key=lambda i:proximity(AE_POSITIONS[i],'ae'))),
                ('mic',min(range(len(MIC_POSITIONS)),key=lambda i:proximity(MIC_POSITIONS[i],'mic')))}
        for _,kind,index in ranked:
            if len(chosen)>=4:break
            chosen.add((kind,index))
        for i,(angle,height) in enumerate(AE_POSITIONS):
            if ('ae',i) not in chosen:continue
            arc=angular_distance(angle,source_az)/180
            dz=abs(height-source_z)
            gain=.58*np.exp(-1.8*arc-1.2*dz)
            if gain<.025:continue
            w=event_wave(rng,'ae',ev['kind'],CONFIGS['ae'])
            ix=round(ev['sample']*ae_fs/FS)+int((arc*.00016+dz*.00008)*ae_fs)
            hi=min(n_ae,ix+len(w))
            if hi>ix:ae[i,ix:hi]+=gain*w[:hi-ix]
        for i,(angle,height) in enumerate(MIC_POSITIONS):
            if ('mic',i) not in chosen:continue
            arc=angular_distance(angle,source_az)/180
            dz=source_z+abs(height)
            gain=.38*np.exp(-1.7*arc-.8*dz)
            if gain<.025:continue
            w=event_wave(rng,'mic',ev['kind'],CONFIGS['mic'])
            ix=round(ev['sample']*mic_fs/FS)+int((.0003+arc*.0003+dz*.00014)*mic_fs)
            hi=min(n_mic,ix+len(w))
            if hi>ix:mic[i,ix:hi]+=gain*w[:hi-ix]
    return {'ae':ae,'mic':mic},truth


def _clean_display_wave(filtered_wave, event_uid):
    """Create a consistent, low-noise event display without changing detection.

    The display uses a low-level noise floor across the entire capture, with a
    smoothly gated band-limited transient added on top. Classification and
    stored detection features still use the original filtered signal.
    """
    import hashlib
    w=np.asarray(filtered_wave,dtype=float)
    if w.size==0:return w.copy()
    peak=float(np.max(np.abs(w)))
    if peak<=1e-12:return w.copy()
    n=w.size
    # Identify the energetic burst using smoothed local signal power rather
    # than the single largest (possibly noisy) sample.
    window=max(9,round(n*0.018))
    power=np.convolve(w*w,np.ones(window)/window,mode='same')
    center=int(np.argmax(power))
    # Smooth attack and exponential decay; no abrupt baseline suppression.
    idx=np.arange(n)
    attack=max(5,round(n*0.009))
    onset=max(0,center-2*attack)
    fade=max(20,round(n*0.105))
    gate=np.where(idx<onset,0.0,
                  np.where(idx<center,
                           np.clip((idx-onset)/max(1,center-onset),0,1)**2,
                           np.exp(-(idx-center)/fade)))
    # Keep the actual oscillatory waveform within the event envelope.
    burst=w*gate
    # Presentation capture: move the energetic arrival toward the left (38%)
    # without wrapping late samples back into the start of the record.
    target=int(round(0.38*n))
    shift=target-center
    if shift:
        moved=np.zeros_like(burst)
        if shift>0:
            moved[shift:]=burst[:n-shift]
        else:
            moved[:n+shift]=burst[-shift:]
        burst=moved
    scale=max(float(np.max(np.abs(burst))),1e-12)
    seed=int.from_bytes(hashlib.sha256(event_uid.encode()).digest()[:8],'little')
    rng=np.random.default_rng(seed)
    floor=rng.normal(0,0.008*scale,size=n)
    # Slightly correlate the floor so it reads as filtered acquisition noise.
    floor=np.convolve(floor,[0.2,0.6,0.2],mode='same')
    return burst+floor

def process_dual(signals,bundles,base_time,sequence,source='synthetic'):
    """Return app-compatible event records and waveform captures.
    Candidate event association is illustrative and not a calibrated localization.
    """
    from datetime import timedelta
    observations=[]
    for key,arr in signals.items():
        config=CONFIGS[key]
        ids=AE_IDS if key=='ae' else MIC_IDS
        for i,x in enumerate(arr):
            result=candidates(x,config,mad_multiplier=9,min_separation_s=.008)
            for p in result['peaks'][:25]:
                width=1024
                a=max(0,int(p)-width//2);b=min(len(x),a+width)
                if b-a<64:continue
                w=result['filtered'][a:b]
                diag=evaluate_event(w,result['noise_sigma'],config)
                score=predict(bundles[key],w,config) if bundles else None
                observations.append({'time':float(p/config.sample_rate),'modality':key,'id':ids[i],
                                     'channel':i,'score':score,'diag':diag,'peak':float(np.max(np.abs(w))),
                                     'wave':w,'display_wave':np.asarray(w,dtype=float),'fs':config.sample_rate,'band_ratio':extract_event_features(w,config)[8]})
    observations.sort(key=lambda x:x['time'])
    clusters=[]
    # Time-only association tolerance is provisional; source location not inferred.
    for o in observations:
        if clusters and abs(o['time']-clusters[-1][-1]['time'])<.0025:clusters[-1].append(o)
        else:clusters.append([o])
    events=[];captures={}
    for j,cluster in enumerate(clusters):
        # Prevent noise-triggered duplicate observations from making one
        # synthetic event appear to excite the whole array. Keep one peak per
        # channel and at most four strongest channels in each event cluster.
        if source.startswith('synthetic'):
            strongest={}
            for obs in cluster:
                key=obs['id']
                if key not in strongest or obs['peak']>strongest[key]['peak']:
                    strongest[key]=obs
            cluster=sorted(strongest.values(),key=lambda obs:obs['peak'],reverse=True)[:4]
        scores=[o['score'] for o in cluster if o['score'] is not None]
        valid=all(o['diag']['signal_valid'] and not o['diag']['clipping_suspected'] for o in cluster)
        decision=decide(scores,quality_ok=valid,physical_consistency=None)
        label={'pd_likely':'PD-like','interference_likely':'Interference','uncertain':'Uncertain',
               'monitoring_degraded':'Monitoring degraded'}[decision['state']]
        peak=max(o['peak'] for o in cluster)
        ae_amps=[0.]*len(AE_IDS);ae_detected=[False]*len(AE_IDS)
        mic_amps=[0.]*len(MIC_IDS);mic_detected=[False]*len(MIC_IDS)
        for o in cluster:
            if o['modality']=='ae':
                ae_amps[o['channel']]=max(ae_amps[o['channel']],o['peak']);ae_detected[o['channel']]=True
            else:
                mic_amps[o['channel']]=max(mic_amps[o['channel']],o['peak']);mic_detected[o['channel']]=True
        stamp=(base_time+timedelta(seconds=cluster[0]['time'])).isoformat(timespec='milliseconds')
        uid=f'{source}-{sequence:010d}-{j:03d}'
        modalities=sorted(set(o['modality'] for o in cluster))
        evidence={'modalities':modalities,'rf_scores':{k:round(max((o['score'] for o in cluster if o['modality']==k and o['score'] is not None),default=0),3) for k in modalities},
                  'quality_ok':valid,'decision_reason':decision['reason'],'sensor_ids':[o['id'] for o in cluster],
                  'model_origin':'synthetic-only RF','location_status':'unresolved',
                  'mic_amplitudes':mic_amps,'mic_detected':mic_detected}
        row=dict(event_uid=uid,timestamp=stamp,classification=label,pd_score=decision.get('score'),
                 gap=None,sector=None,location_score=None,uv_inner=0,uv_outer=0,peak=peak,
                 sensor_count=len(cluster),amplitudes=ae_amps,detected=ae_detected,
                 time_quality='SIMULATED_CLOCK',source=source,sequence=sequence,evidence=evidence)
        events.append(row)
        if label=='PD-like':
            best=max(cluster,key=lambda o:(o['score'] or 0)*o['peak'])
            captures[uid]={'event_uid':uid,'fs':best['fs'],'sensor':best['id'],'modality':best['modality'],
                           'samples':_clean_display_wave(best['display_wave'],uid).tolist(),'timestamp':stamp,'peak':best['peak']}
    return events,captures
