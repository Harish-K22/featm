from __future__ import annotations

import numpy as np
from ...registry import BatchSemantics, Cost, FeatureOutput, REGISTRY, register_feature
from .._utils import gray_videos, as_videos


def _reg(name,family,*,cost=Cost.LOW,output_kind='scalar',tags=(),requires=()):
    def deco(fn):
        if name in {s.name for s in REGISTRY.all()}:return fn
        # Fifth audit pass, Finding 1: every video.* feature's per-row
        # sample is a real pre-loaded numpy.ndarray clip, only representable
        # by an in-memory PandasSource -- see BatchSemantics.OPAQUE_OBJECT_SAMPLE.
        return register_feature(name=name,family=family,modality='video',input_types=('video',),cost=cost,streamable=True,output_kind=output_kind,tags=tags,requires=requires,description=name,batch_semantics=BatchSemantics.OPAQUE_OBJECT_SAMPLE)(fn)
    return deco

@_reg('video.frame_statistics.mean_intensity','frame_statistics',cost=Cost.TINY,tags=('intensity',))
def mean_intensity(x,**kwargs):return np.mean(gray_videos(x),axis=(1,2,3))
@_reg('video.frame_statistics.intensity','frame_statistics',output_kind='vector',cost=Cost.TINY,tags=('intensity',))
def intensity(x,**kwargs):
    a=gray_videos(x);f=np.mean(a,axis=(2,3));return FeatureOutput(np.column_stack([f.mean(1),f.std(1),f.min(1),f.max(1)]),('mean','std','min','max'))
@_reg('video.temporal_statistics.temporal_std','temporal_statistics',cost=Cost.TINY,tags=('temporal',))
def temporal_std(x,**kwargs):
    a=gray_videos(x);return np.mean(np.std(a,axis=1),axis=(1,2))
@_reg('video.temporal_statistics.variation','temporal_statistics',output_kind='vector',cost=Cost.TINY,tags=('variation',))
def variation(x,**kwargs):
    a=gray_videos(x);d=np.diff(a,axis=1);v=np.mean(np.abs(d),axis=(2,3));return FeatureOutput(np.column_stack([v.mean(1),v.std(1),v.max(1)]),('mean_abs_frame_delta','std','max'))
@_reg('video.motion.frame_difference_energy','motion',cost=Cost.TINY,tags=('motion',))
def frame_diff(x,**kwargs):
    a=gray_videos(x);d=np.diff(a,axis=1);return np.mean(d*d,axis=(1,2,3))


def _flow_summary(x):
    import cv2
    out=[];fields=[]
    for vid in gray_videos(x):
        mags=[];angs=[];dx=[];dy=[]
        for a,b in zip(vid[:-1],vid[1:]):
            aa=np.nan_to_num(a).astype(np.float32);bb=np.nan_to_num(b).astype(np.float32);flow=cv2.calcOpticalFlowFarneback(aa,bb,None,.5,3,15,3,5,1.2,0);m,ang=cv2.cartToPolar(flow[...,0],flow[...,1]);mags.append(m.mean());angs.append(np.angle(np.mean(np.exp(1j*ang))));dx.append(np.median(flow[...,0]));dy.append(np.median(flow[...,1]))
        out.append([np.mean(mags) if mags else 0,np.std(mags) if mags else 0,np.max(mags) if mags else 0,np.mean(dx) if dx else 0,np.mean(dy) if dy else 0,np.std(angs) if angs else 0])
    return np.asarray(out)

@_reg('video.motion.optical_flow','motion',output_kind='vector',cost=Cost.HIGH,tags=('optical-flow',),requires=('opencv-python',))
def optical_flow(x,**kwargs):return FeatureOutput(_flow_summary(x),('magnitude_mean','magnitude_std','magnitude_max','global_dx','global_dy','angle_std'))
@_reg('video.motion.motion_vectors','motion',output_kind='vector',cost=Cost.HIGH,tags=('motion-vectors',),requires=('opencv-python',))
def motion_vectors(x,**kwargs):
    f=_flow_summary(x);return FeatureOutput(f[:,3:5],('dx','dy'))
@_reg('video.camera_motion.global_flow','camera_motion',output_kind='vector',cost=Cost.HIGH,tags=('camera-motion',),requires=('opencv-python',))
def global_flow(x,**kwargs):return motion_vectors(x)
@_reg('video.motion.motion_histograms','motion',output_kind='vector',cost=Cost.HIGH,tags=('motion-histogram',),requires=('opencv-python',))
def motion_hist(x,bins=8,**kwargs):
    import cv2;out=[]
    for vid in gray_videos(x):
        h=np.zeros(bins)
        for a,b in zip(vid[:-1],vid[1:]):
            flow=cv2.calcOpticalFlowFarneback(np.nan_to_num(a).astype(np.float32),np.nan_to_num(b).astype(np.float32),None,.5,3,15,3,5,1.2,0);m,ang=cv2.cartToPolar(flow[...,0],flow[...,1]);idx=np.floor(ang/(2*np.pi)*bins).astype(int)%bins
            for k in range(bins):h[k]+=m[idx==k].sum()
        if h.sum():h/=h.sum()
        out.append(h)
    return FeatureOutput(np.asarray(out),tuple(f'direction_{i:02d}' for i in range(bins)))
@_reg('video.motion.trajectories','motion',output_kind='vector',cost=Cost.HIGH,tags=('trajectory',),requires=('opencv-python',))
def trajectories(x,**kwargs):
    f=_flow_summary(x);disp=np.sqrt(f[:,3]**2+f[:,4]**2);return FeatureOutput(np.column_stack([f[:,3],f[:,4],disp]),('mean_dx','mean_dy','net_speed'))
@_reg('video.track_geometry.velocity','track_geometry',output_kind='vector',cost=Cost.HIGH,tags=('velocity',),requires=('opencv-python',))
def track_velocity(x,**kwargs):return trajectories(x)

@_reg('video.scene_change.scene_change_max','scene_change',cost=Cost.LOW,tags=('scene-change',))
def scene_change_max(x,**kwargs):
    a=gray_videos(x);d=np.mean(np.abs(np.diff(a,axis=1)),axis=(2,3));return np.max(d,axis=1) if d.shape[1] else np.zeros(len(a))
@_reg('video.scene_change.cuts','scene_change',output_kind='vector',cost=Cost.LOW,tags=('cuts',))
def cuts(x,threshold_std=3.0,**kwargs):
    a=gray_videos(x);out=[]
    for vid in a:
        d=np.mean(np.abs(np.diff(vid,axis=0)),axis=(1,2));thr=np.mean(d)+threshold_std*np.std(d);idx=np.where(d>thr)[0];out.append([len(idx),np.mean(d[idx]) if len(idx) else 0,thr])
    return FeatureOutput(np.asarray(out),('cut_count','cut_strength_mean','threshold'))
@_reg('video.shot_statistics.duration','shot_statistics',output_kind='vector',cost=Cost.LOW,tags=('shots',))
def shot_duration(x,fps=30,**kwargs):
    a=gray_videos(x);out=[]
    for vid in a:
        d=np.mean(np.abs(np.diff(vid,axis=0)),axis=(1,2));thr=np.mean(d)+3*np.std(d);idx=np.r_[0,np.where(d>thr)[0]+1,len(vid)];dur=np.diff(idx)/fps;out.append([np.mean(dur),np.std(dur),np.max(dur),len(dur)])
    return FeatureOutput(np.asarray(out),('mean','std','max','shot_count'))

@_reg('video.temporal_frequency.fft','temporal_frequency',output_kind='vector',cost=Cost.MEDIUM,tags=('fft',))
def temporal_fft(x,bins=8,**kwargs):
    a=gray_videos(x);f=np.mean(a,axis=(2,3));P=np.abs(np.fft.rfft(f-f.mean(1,keepdims=True),axis=1))**2;k=min(bins,P.shape[1]);return FeatureOutput(P[:,:k],tuple(f'power_{i:02d}' for i in range(k)))

@_reg('video.dynamic_texture.lbp_top','dynamic_texture',output_kind='vector',cost=Cost.HIGH,tags=('lbp-top',),requires=('scikit-image',))
def lbp_top(x,bins=10,**kwargs):
    from skimage.feature import local_binary_pattern
    out=[]
    for vid in gray_videos(x):
        planes=[vid[len(vid)//2],vid[:,vid.shape[1]//2,:],vid[:,:,vid.shape[2]//2]];vals=[]
        for pl in planes:
            u=np.asarray(pl,float);lo,hi=u.min(),u.max();q=((u-lo)/(hi-lo+1e-12)*255).astype(np.uint8);v=local_binary_pattern(q,8,1,method='uniform');h,_=np.histogram(v,bins=bins,range=(0,10));h=h/h.sum() if h.sum() else h;vals.extend(h)
        out.append(vals)
    return FeatureOutput(np.asarray(out),tuple(f'{plane}_b{i:02d}' for plane in ('xy','ty','tx') for i in range(bins)))

@_reg('video.action.statistics','action',output_kind='vector',cost=Cost.MEDIUM,tags=('action','motion'))
def action_stats(x,**kwargs):
    return FeatureOutput(np.column_stack([frame_diff(x),temporal_std(x),scene_change_max(x)]),('motion_energy','temporal_std','max_scene_change'))

# External detector/pose/landmark adapters.
for _name,_family in [('video.object_tracks.track_count','object_tracks'),('video.pose_dynamics.joint_velocity','pose_dynamics'),('video.facial_motion.landmarks','facial_motion')]:
    @_reg(_name,_family,output_kind='vector',cost=Cost.EXTREME,tags=('user-model',),requires=('user-model',))
    def _model_feature(x,model=None,_name=_name,**kwargs):
        if model is None:raise ValueError(f'{_name} requires a user-supplied callable model(video)->fixed feature vector')
        rows=[np.asarray(model(v),float).ravel() for v in as_videos(x)];w=len(rows[0]);
        if any(len(v)!=w for v in rows):raise ValueError('model output width must be fixed')
        return FeatureOutput(np.asarray(rows),tuple(f'dim_{i:03d}' for i in range(w)))

# Video embeddings are deliberately model adapters. This keeps featm from
# forcing multi-GB dependencies/models onto users who only want classic features.
for _name in ('3d_cnn','timesformer','video_mae'):
    @_reg(f'video.embeddings.{_name}','embeddings',output_kind='vector',cost=Cost.EXTREME,tags=('embedding',_name),requires=('user-model',))
    def _video_emb(x,model=None,_name=_name,**kwargs):
        if model is None:raise ValueError(f'video.embeddings.{_name} requires callable model(video)->fixed embedding')
        rows=[np.asarray(model(v),float).ravel() for v in as_videos(x)];w=len(rows[0]);return FeatureOutput(np.asarray(rows),tuple(f'dim_{i:04d}' for i in range(w)))
