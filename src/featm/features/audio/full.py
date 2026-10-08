from __future__ import annotations

from ...contracts import forward_parameters

import numpy as np

from ...registry import BatchSemantics, Cost, FeatureOutput, REGISTRY, register_feature
from .._utils import as_2d_series, spectral_parts


def _reg(name,family,*,cost=Cost.LOW,output_kind='scalar',tags=(),requires=(),description=''):
    def deco(fn):
        if name in {s.name for s in REGISTRY.all()}: return fn
        # Fifth audit pass, Finding 1: every audio.* feature's per-row
        # sample is a real pre-loaded numpy.ndarray waveform, only
        # representable by an in-memory PandasSource -- see
        # BatchSemantics.OPAQUE_OBJECT_SAMPLE.
        return register_feature(name=name,family=family,modality='audio',input_types=('audio',),cost=cost,streamable=True,output_kind=output_kind,tags=tags,requires=requires,description=description or name,batch_semantics=BatchSemantics.OPAQUE_OBJECT_SAMPLE)(fn)
    return deco


def _audio(x):
    a=np.asarray(x,dtype=float)
    if a.ndim==1:a=a[None,:]
    if a.ndim!=2:raise ValueError('mono audio batch must be (n_samples,n_frames) or (n_frames,)')
    return a


def _librosa():
    try: import librosa
    except ImportError as exc: raise ImportError('Install featm[audio] (librosa) for this feature') from exc
    return librosa


@_reg('audio.waveform.rms','waveform',cost=Cost.TINY,tags=('rms',))
def rms(x,**kwargs):return np.sqrt(np.mean(_audio(x)**2,axis=1))

@_reg('audio.waveform.zero_crossing_rate','waveform',cost=Cost.TINY,tags=('zero-crossing',))
def zcr(x,**kwargs):
    a=_audio(x);return (np.signbit(a[:,1:])!=np.signbit(a[:,:-1])).mean(1)

@_reg('audio.waveform.crest_factor','waveform',cost=Cost.TINY,tags=('crest',))
def crest(x,**kwargs):
    a=_audio(x);rr=np.sqrt(np.mean(a*a,1));return np.divide(np.max(np.abs(a),1),rr,out=np.zeros(len(a)),where=rr>0)

@_reg('audio.waveform.dynamic_range','waveform',cost=Cost.TINY,tags=('dynamic-range',))
def dynamic_range(x,**kwargs):
    a=np.abs(_audio(x));hi=np.quantile(a,.95,axis=1);lo=np.quantile(a,.10,axis=1);return 20*np.log10((hi+1e-12)/(lo+1e-12))


def _stft_power(y,n_fft=1024,hop=256):
    lib=_librosa();S=np.abs(lib.stft(y,n_fft=min(n_fft,max(32,len(y))),hop_length=hop))**2;return S

@_reg('audio.spectral.centroid','spectral',tags=('spectral',),requires=('librosa',))
def centroid(x,sr=22050,**kwargs):
    lib=_librosa();return np.array([float(np.nanmean(lib.feature.spectral_centroid(y=r,sr=sr))) for r in _audio(x)])

@_reg('audio.spectral.centroid_normalized','spectral',tags=('spectral',),requires=('librosa',))
def centroid_norm(x,sr=22050,**kwargs):return centroid(x,sr=sr)/(sr/2)

@_reg('audio.spectral.bandwidth','spectral',tags=('spectral',),requires=('librosa',))
def bandwidth(x,sr=22050,**kwargs):
    lib=_librosa();return np.array([float(np.nanmean(lib.feature.spectral_bandwidth(y=r,sr=sr))) for r in _audio(x)])

@_reg('audio.spectral.flatness','spectral',tags=('spectral',),requires=('librosa',))
def flatness(x,**kwargs):
    lib=_librosa();return np.array([float(np.nanmean(lib.feature.spectral_flatness(y=r))) for r in _audio(x)])

@_reg('audio.spectral.rolloff','spectral',tags=('spectral',),requires=('librosa',))
def rolloff(x,sr=22050,**kwargs):
    lib=_librosa();return np.array([float(np.nanmean(lib.feature.spectral_rolloff(y=r,sr=sr,roll_percent=.85))) for r in _audio(x)])

@_reg('audio.spectral.contrast','spectral',output_kind='vector',cost=Cost.MEDIUM,tags=('spectral',),requires=('librosa',))
def contrast(x,sr=22050,n_bands=6,**kwargs):
    lib=_librosa();out=[]
    for r in _audio(x):
        S=np.abs(lib.stft(r));c=lib.feature.spectral_contrast(S=S,sr=sr,n_bands=n_bands);out.append(np.mean(c,axis=1))
    width=n_bands+1;return FeatureOutput(np.asarray(out),tuple(f'band_{i}' for i in range(width)))

@_reg('audio.spectral.flux','spectral',cost=Cost.MEDIUM,tags=('spectral-flux',),requires=('librosa',))
def flux(x,**kwargs):
    lib=_librosa();out=[]
    for r in _audio(x):
        S=np.abs(lib.stft(r));Sn=S/(np.linalg.norm(S,axis=0,keepdims=True)+1e-12);d=np.diff(Sn,axis=1);out.append(np.sqrt((d*d).sum(axis=0)).mean() if d.shape[1] else 0)
    return np.asarray(out)


@_reg('audio.perceptual.mel','perceptual',output_kind='vector',cost=Cost.MEDIUM,tags=('mel',),requires=('librosa',))
def mel(x,sr=22050,n_mels=40,**kwargs):
    lib=_librosa();out=[]
    for r in _audio(x):
        M=lib.feature.melspectrogram(y=r,sr=sr,n_mels=n_mels);out.append(np.log1p(M).mean(axis=1))
    return FeatureOutput(np.asarray(out),tuple(f'mel_{i:02d}' for i in range(n_mels)))

@_reg('audio.perceptual.mfcc','perceptual',output_kind='vector',cost=Cost.MEDIUM,tags=('mfcc',),requires=('librosa',))
def mfcc(x,sr=22050,n_mfcc=13,**kwargs):
    lib=_librosa();out=[lib.feature.mfcc(y=r,sr=sr,n_mfcc=n_mfcc).mean(axis=1) for r in _audio(x)];return FeatureOutput(np.asarray(out),tuple(f'c{i:02d}' for i in range(n_mfcc)))


def _tri_filterbank(freqs,centers):
    fb=[]
    for i in range(1,len(centers)-1):
        l,c,r=centers[i-1],centers[i],centers[i+1];w=np.maximum(0,np.minimum((freqs-l)/(c-l+1e-12),(r-freqs)/(r-c+1e-12)));fb.append(w)
    return np.asarray(fb)

@_reg('audio.perceptual.bark','perceptual',output_kind='vector',cost=Cost.MEDIUM,tags=('bark',))
def bark(x,sr=22050,n_bands=24,**kwargs):
    out=[]
    for y in _audio(x):
        _,_,p,f=spectral_parts(y);p=p[0]; hz=f*sr
        # Traunmuller-like Bark mapping and inverse sampled numerically.
        z=26.81*hz/(1960+hz)-0.53;zedges=np.linspace(0,max(1,z.max()),n_bands+2);centers=np.interp(zedges,z,hz);fb=_tri_filterbank(hz,centers);e=fb@p;out.append(np.log1p(e))
    return FeatureOutput(np.asarray(out),tuple(f'bark_{i:02d}' for i in range(n_bands)))

@_reg('audio.perceptual.gfcc','perceptual',output_kind='vector',cost=Cost.MEDIUM,tags=('gfcc','erb'))
def gfcc(x,sr=22050,n_gfcc=13,n_bands=32,**kwargs):
    from scipy.fft import dct
    out=[]
    for y in _audio(x):
        _,_,p,f=spectral_parts(y);p=p[0];hz=f*sr;erb=21.4*np.log10(1+0.00437*hz);edges=np.linspace(erb.min(),erb.max(),n_bands+2);centers=np.interp(edges,erb,hz);fb=_tri_filterbank(hz,centers);e=np.log(np.maximum(fb@p,1e-12));out.append(dct(e,type=2,norm='ortho')[:n_gfcc])
    return FeatureOutput(np.asarray(out),tuple(f'c{i:02d}' for i in range(n_gfcc)))

@_reg('audio.perceptual.loudness','perceptual',cost=Cost.TINY,tags=('loudness',))
def loudness(x,**kwargs):return 20*np.log10(np.maximum(rms(x),1e-12))


@_reg('audio.tonal.chroma','tonal',output_kind='vector',cost=Cost.MEDIUM,tags=('chroma',),requires=('librosa',))
def chroma(x,sr=22050,**kwargs):
    lib=_librosa();out=[lib.feature.chroma_stft(y=r,sr=sr).mean(axis=1) for r in _audio(x)];return FeatureOutput(np.asarray(out),tuple(f'pc_{i:02d}' for i in range(12)))

@_reg('audio.tonal.tonnetz','tonal',output_kind='vector',cost=Cost.HIGH,tags=('tonnetz',),requires=('librosa',))
def tonnetz(x,sr=22050,**kwargs):
    lib=_librosa();out=[]
    for r in _audio(x):
        try:h=lib.effects.harmonic(r);v=lib.feature.tonnetz(y=h,sr=sr).mean(axis=1)
        except Exception:v=np.full(6,np.nan)
        out.append(v)
    return FeatureOutput(np.asarray(out),tuple(f'dim_{i}' for i in range(6)))

@_reg('audio.tonal.pitch','tonal',cost=Cost.MEDIUM,tags=('pitch',),requires=('librosa',))
def pitch(x,sr=22050,fmin=50,fmax=2000,**kwargs):
    lib=_librosa();out=[]
    for r in _audio(x):
        try:f=lib.yin(r,fmin=fmin,fmax=min(fmax,sr/2-1),sr=sr);out.append(np.nanmedian(f))
        except Exception:out.append(np.nan)
    return np.asarray(out)

@_reg('audio.tonal.key','tonal',output_kind='vector',cost=Cost.MEDIUM,tags=('key',),requires=('librosa',))
def key(x,sr=22050,**kwargs):
    c=chroma(x,sr=sr).values;idx=np.argmax(c,axis=1);strength=c[np.arange(len(c)),idx];return FeatureOutput(np.column_stack([idx,strength]),('pitch_class','strength'))

@_reg('audio.tonal.harmonicity','tonal',cost=Cost.MEDIUM,tags=('harmonicity',))
def harmonicity(x,**kwargs):
    out=[]
    for r in _audio(x):
        rr=r-np.mean(r);ac=np.correlate(rr,rr,mode='full')[len(rr)-1:];
        if len(ac)<3 or ac[0]<=0:out.append(0);continue
        lo=max(1,len(r)//500);hi=min(len(ac),max(lo+2,len(r)//50));out.append(np.max(ac[lo:hi])/ac[0] if hi>lo else 0)
    return np.asarray(out)


# Rhythm.
@_reg('audio.rhythm.tempo','rhythm',cost=Cost.MEDIUM,tags=('tempo',),requires=('librosa',))
def tempo(x,sr=22050,**kwargs):
    lib=_librosa();out=[]
    for r in _audio(x):
        onset=lib.onset.onset_strength(y=r,sr=sr);t=lib.feature.tempo(onset_envelope=onset,sr=sr);out.append(float(np.asarray(t).ravel()[0]) if np.size(t) else np.nan)
    return np.asarray(out)

@_reg('audio.rhythm.beats','rhythm',output_kind='vector',cost=Cost.MEDIUM,tags=('beats',),requires=('librosa',))
def beats(x,sr=22050,**kwargs):
    lib=_librosa();out=[]
    for r in _audio(x):
        t,b=lib.beat.beat_track(y=r,sr=sr);bt=lib.frames_to_time(b,sr=sr);interval=np.diff(bt);out.append([len(b),np.mean(interval) if len(interval) else np.nan,np.std(interval) if len(interval) else np.nan])
    return FeatureOutput(np.asarray(out),('beat_count','interval_mean','interval_std'))

@_reg('audio.rhythm.onset','rhythm',output_kind='vector',cost=Cost.MEDIUM,tags=('onset',),requires=('librosa',))
def onset(x,sr=22050,**kwargs):
    lib=_librosa();out=[]
    for r in _audio(x):
        env=lib.onset.onset_strength(y=r,sr=sr);idx=lib.onset.onset_detect(onset_envelope=env,sr=sr);out.append([len(idx),np.mean(env) if len(env) else 0,np.max(env) if len(env) else 0])
    return FeatureOutput(np.asarray(out),('count','strength_mean','strength_max'))

@_reg('audio.rhythm.tempogram','rhythm',output_kind='vector',cost=Cost.HIGH,tags=('tempogram',),requires=('librosa',))
def tempogram(x,sr=22050,n_bins=16,**kwargs):
    lib=_librosa();out=[]
    for r in _audio(x):
        env=lib.onset.onset_strength(y=r,sr=sr);T=lib.feature.tempogram(onset_envelope=env,sr=sr);v=T.mean(axis=1);idx=np.linspace(0,max(len(v)-1,0),n_bins).astype(int);out.append(v[idx] if len(v) else np.zeros(n_bins))
    return FeatureOutput(np.asarray(out),tuple(f'bin_{i:02d}' for i in range(n_bins)))


# Voice descriptors.
@_reg('audio.voice.f0','voice',cost=Cost.MEDIUM,tags=('f0',),requires=('librosa',))
def f0(x,**kwargs):return pitch(x,**kwargs)
f0.__featm_parameters__ = forward_parameters(pitch, positional=1, bound=[])

@_reg('audio.voice.formants','voice',output_kind='vector',cost=Cost.HIGH,tags=('formants','lpc'),requires=('librosa',))
def formants(x,sr=22050,n_formants=4,**kwargs):
    lib=_librosa();out=[]
    for r in _audio(x):
        try:
            order=min(16,max(4,2+int(sr/1000)));a=lib.lpc(r,order=order);roots=np.roots(a);roots=roots[np.imag(roots)>=0.01];ang=np.arctan2(np.imag(roots),np.real(roots));freq=np.sort(ang*sr/(2*np.pi));freq=freq[(freq>90)&(freq<sr/2)][:n_formants];out.append(np.pad(freq,(0,n_formants-len(freq)),constant_values=np.nan))
        except Exception:out.append([np.nan]*n_formants)
    return FeatureOutput(np.asarray(out),tuple(f'F{i+1}' for i in range(n_formants)))

@_reg('audio.voice.hnr','voice',cost=Cost.MEDIUM,tags=('hnr',))
def hnr(x,**kwargs):
    h=harmonicity(x);return 10*np.log10(np.maximum(h,1e-12)/np.maximum(1-h,1e-12))

@_reg('audio.voice.jitter','voice',cost=Cost.HIGH,tags=('jitter',),requires=('librosa',))
def jitter(x,sr=22050,**kwargs):
    lib=_librosa();out=[]
    for r in _audio(x):
        f=lib.yin(r,fmin=50,fmax=min(1000,sr/2-1),sr=sr);p=1/np.maximum(f,1e-12);out.append(np.mean(np.abs(np.diff(p)))/np.mean(p) if len(p)>1 and np.mean(p)>0 else np.nan)
    return np.asarray(out)

@_reg('audio.voice.shimmer','voice',cost=Cost.HIGH,tags=('shimmer',))
def shimmer(x,frame=512,hop=256,**kwargs):
    out=[]
    for r in _audio(x):
        if len(r)<frame:out.append(np.nan);continue
        w=np.lib.stride_tricks.sliding_window_view(r,frame)[::hop];amp=np.sqrt(np.mean(w*w,axis=1));out.append(np.mean(np.abs(np.diff(amp)))/np.mean(amp) if len(amp)>1 and np.mean(amp)>0 else np.nan)
    return np.asarray(out)


# Stereo spatial features. Input shape (n,t,2) or (t,2).
def _stereo(x):
    a=np.asarray(x,float)
    if a.ndim==2 and a.shape[-1]==2:a=a[None,:,:]
    if a.ndim!=3 or a.shape[-1]!=2:raise ValueError('spatial audio expects shape (n,t,2)')
    return a

@_reg('audio.spatial.ild','spatial',cost=Cost.TINY,tags=('stereo','ILD'))
def ild(x,**kwargs):
    a=_stereo(x);l=np.sqrt(np.mean(a[:,:,0]**2,1));r=np.sqrt(np.mean(a[:,:,1]**2,1));return 20*np.log10((l+1e-12)/(r+1e-12))

@_reg('audio.spatial.itd','spatial',cost=Cost.MEDIUM,tags=('stereo','ITD'))
def itd(x,sr=22050,max_ms=2.0,**kwargs):
    a=_stereo(x);out=[];maxlag=int(sr*max_ms/1000)
    for z in a:
        l=z[:,0]-z[:,0].mean();r=z[:,1]-z[:,1].mean();c=np.correlate(l,r,mode='full');lags=np.arange(-len(l)+1,len(l));m=np.abs(lags)<=maxlag;lag=lags[m][np.argmax(c[m])];out.append(lag/sr)
    return np.asarray(out)


# Optional learned embedding adapters. They are real implementations but keep
# heavyweight model stacks out of the base install, exactly like optional model
# backends in a registry library.
def _hf_audio_embeddings(x, *, model_name, sr=16000, pooling='mean'):
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor
    except ImportError as exc:
        raise ImportError("Install 'transformers' and 'torch' to use audio embedding features") from exc
    try:
        proc=AutoProcessor.from_pretrained(model_name)
    except OSError:
        # A pure feature-extraction backbone with no ASR/text head (e.g.
        # facebook/hubert-base-ls960) ships no tokenizer_config.json at all,
        # so AutoProcessor (which needs both a feature extractor AND a
        # tokenizer to build a Processor) cannot be instantiated even though
        # the checkpoint is otherwise perfectly usable for embeddings.
        # AutoFeatureExtractor only needs the feature-extraction half, which
        # every one of these checkpoints does ship.
        proc=AutoFeatureExtractor.from_pretrained(model_name)
    model=AutoModel.from_pretrained(model_name);model.eval();rows=[]
    # An encoder-decoder checkpoint (e.g. Whisper) loaded via AutoModel exposes a
    # get_encoder() accessor; calling the full model(**inp) on encoder-only
    # inputs then raises "You have to specify either decoder_input_ids or
    # decoder_inputs_embeds" since the processor never produces decoder
    # inputs. This is a real audio embedding feature, not a generation task,
    # so only the encoder's hidden states are needed -- run the encoder
    # directly for any checkpoint architecture that has one.
    #
    # A dual audio/text contrastive checkpoint (e.g. CLAP) is neither
    # encoder-only nor encoder-decoder -- model(**inp) on audio-only inputs
    # raises "text input must be of type str" because its forward() always
    # expects a paired text input. get_audio_features() runs only the audio
    # tower, which is exactly what an AUDIO EMBEDDING feature needs; no
    # text pairing is meaningful here since there is no caller-supplied text.
    is_dual_encoder=hasattr(model,'get_audio_features')
    forward=model.get_audio_features if is_dual_encoder else (model.get_encoder() if hasattr(model,'get_encoder') else model)
    # Some checkpoints (e.g. CLAP, trained at 48kHz) reject a caller-supplied
    # sr that does not match their own declared native rate outright, rather
    # than resampling -- defer to the processor's own declared rate when the
    # caller left sr at this helper's generic 16kHz default.
    native_sr=getattr(getattr(proc,'feature_extractor',proc),'sampling_rate',None)
    effective_sr=native_sr if (native_sr and sr==16000) else sr
    for y in _audio(x):
        inp=proc(audios=y,sampling_rate=effective_sr,return_tensors='pt') if is_dual_encoder else proc(y,sampling_rate=effective_sr,return_tensors='pt')
        with torch.no_grad():out=forward(**inp)
        h=getattr(out,'last_hidden_state',None)
        if h is None: h=out[0]
        v=h.mean(dim=1).squeeze(0).cpu().numpy() if h.ndim>=3 else h.squeeze(0).cpu().numpy()
        rows.append(v)
    return FeatureOutput(np.asarray(rows),tuple(f'dim_{i:04d}' for i in range(len(rows[0]))))

for _name,_model in [
    ('wav2vec','facebook/wav2vec2-base-960h'),('hubert','facebook/hubert-base-ls960'),('whisper','openai/whisper-tiny'),('clap','laion/clap-htsat-unfused')
]:
    @_reg(f'audio.embeddings.{_name}','embeddings',output_kind='vector',cost=Cost.EXTREME,tags=('embedding','pretrained'),requires=('torch','transformers'))
    def _emb(x,model_name=_model,**kwargs):return _hf_audio_embeddings(x,model_name=model_name,**kwargs)
    _emb.__featm_parameters__ = forward_parameters(_hf_audio_embeddings, positional=1, bound=['model_name'])
