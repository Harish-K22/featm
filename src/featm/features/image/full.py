from __future__ import annotations

from ...contracts import forward_parameters

import math
import numpy as np

from ...registry import BatchSemantics, Cost, FeatureOutput, REGISTRY, register_feature
from .._utils import as_images, gray_images, shannon_entropy_prob


def _reg(name,family,*,cost=Cost.LOW,output_kind='scalar',tags=(),requires=()):
    def deco(fn):
        if name in {s.name for s in REGISTRY.all()}:return fn
        # Fifth audit pass, Finding 1: every image.* feature's per-row
        # sample is a real pre-loaded numpy.ndarray, only representable by
        # an in-memory PandasSource -- see BatchSemantics.OPAQUE_OBJECT_SAMPLE.
        return register_feature(name=name,family=family,modality='image',input_types=('image',),cost=cost,streamable=True,output_kind=output_kind,tags=tags,requires=requires,description=name,batch_semantics=BatchSemantics.OPAQUE_OBJECT_SAMPLE)(fn)
    return deco


def _gray_uint8(im):
    g=np.asarray(im,float)
    if g.ndim==3:
        g=0.2126*g[...,0]+0.7152*g[...,1]+0.0722*g[...,2] if g.shape[-1]>=3 else g[...,0]
    lo,hi=np.nanmin(g),np.nanmax(g)
    if hi>lo:g=(g-lo)/(hi-lo)*255
    return np.nan_to_num(g).clip(0,255).astype(np.uint8)


def _masks(x):
    gs=gray_images(x);return np.asarray([g>np.nanmean(g) for g in gs])

# Color.
for _ch,_idx in [('red_mean',0),('green_mean',1),('blue_mean',2)]:
    @_reg(f'image.color.{_ch}','color',cost=Cost.TINY,tags=('color',))
    def _color_mean(x,_idx=_idx,**kwargs):
        a=as_images(x)
        if a.ndim==3:return np.nanmean(a,axis=(1,2))
        idx=min(_idx,a.shape[-1]-1);return np.nanmean(a[...,idx],axis=(1,2))

@_reg('image.color.colorfulness','color',cost=Cost.TINY,tags=('colorfulness',))
def colorfulness(x,**kwargs):
    a=as_images(x)
    if a.ndim!=4 or a.shape[-1]<3:return np.zeros(len(a))
    R,G,B=a[...,0],a[...,1],a[...,2];rg=R-G;yb=.5*(R+G)-B
    return np.sqrt(np.std(rg,axis=(1,2))**2+np.std(yb,axis=(1,2))**2)+.3*np.sqrt(np.mean(rg,axis=(1,2))**2+np.mean(yb,axis=(1,2))**2)


def _hist_color(x,space,bins=8):
    import cv2
    a=as_images(x);out=[]
    for im in a:
        u=np.nan_to_num(im).astype(np.float32)
        if u.ndim==2:u=np.repeat(u[...,None],3,axis=-1)
        if u.max()<=1.5:u=u*255
        u=np.clip(u,0,255).astype(np.uint8)
        if space=='hsv':v=cv2.cvtColor(u[...,:3],cv2.COLOR_RGB2HSV)
        elif space=='lab':v=cv2.cvtColor(u[...,:3],cv2.COLOR_RGB2LAB)
        else:v=u[...,:3]
        vals=[]
        for c in range(3):
            h,_=np.histogram(v[...,c],bins=bins,range=(0,256),density=False);h=h/h.sum() if h.sum() else h;vals.extend(h)
        out.append(vals)
    return FeatureOutput(np.asarray(out),tuple(f'{space}_c{c}_b{b:02d}' for c in range(3) for b in range(bins)))

for _name,_space in [('rgb_histogram','rgb'),('hsv_histogram','hsv'),('lab_histogram','lab')]:
    @_reg(f'image.color.{_name}','color',output_kind='vector',cost=Cost.MEDIUM,tags=('histogram','color'),requires=('opencv-python',))
    def _hist(x,_space=_space,**kwargs):return _hist_color(x,_space,**kwargs)
    _hist.__featm_parameters__ = forward_parameters(_hist_color, positional=2, bound=[])

# Intensity.
@_reg('image.intensity.entropy','intensity',cost=Cost.LOW,tags=('entropy',))
def intensity_entropy(x,bins=64,**kwargs):
    out=[]
    for g in gray_images(x):
        h,_=np.histogram(g[np.isfinite(g)],bins=bins);out.append(float(shannon_entropy_prob(h)))
    return np.asarray(out)

# Edges and gradients.
@_reg('image.edges.edge_density','edges',cost=Cost.LOW,tags=('edges',),requires=('scikit-image',))
def edge_density(x,**kwargs):
    from skimage.feature import canny
    return np.asarray([np.mean(canny(_gray_uint8(g)/255.0)) for g in gray_images(x)])

@_reg('image.gradient.hog','gradient',output_kind='vector',cost=Cost.MEDIUM,tags=('hog',),requires=('scikit-image',))
def hog(x,orientations=9,**kwargs):
    from skimage.feature import hog as skhog
    out=[]
    for g in gray_images(x):
        h=skhog(_gray_uint8(g),orientations=orientations,pixels_per_cell=(8,8),cells_per_block=(1,1),feature_vector=False)
        # Average spatial cells to fixed orientation histogram.
        vals=np.asarray(h).reshape(-1,orientations).mean(0) if np.size(h) else np.zeros(orientations);out.append(vals)
    return FeatureOutput(np.asarray(out),tuple(f'orientation_{i:02d}' for i in range(orientations)))

@_reg('image.gradient.structure_tensor','gradient',output_kind='vector',cost=Cost.MEDIUM,tags=('structure-tensor',),requires=('scikit-image',))
def structure_tensor(x,**kwargs):
    from skimage.feature import structure_tensor, structure_tensor_eigenvalues
    out=[]
    for g in gray_images(x):
        A=structure_tensor(_gray_uint8(g)/255.0,sigma=1);ev=structure_tensor_eigenvalues(A);l1,l2=ev[0],ev[1];coh=(l1-l2)/(l1+l2+1e-12);out.append([np.mean(l1),np.mean(l2),np.mean(coh),np.std(coh)])
    return FeatureOutput(np.asarray(out),('eig1_mean','eig2_mean','coherence_mean','coherence_std'))

# Frequency.
@_reg('image.frequency.fft2','frequency',output_kind='vector',cost=Cost.MEDIUM,tags=('fft2',))
def fft2_feature(x,bins=12,**kwargs):
    out=[]
    for g in gray_images(x):
        F=np.fft.fftshift(np.fft.fft2(np.nan_to_num(g-g.mean())));P=np.abs(F)**2;h,w=P.shape;yy,xx=np.indices(P.shape);r=np.sqrt((yy-h/2)**2+(xx-w/2)**2);edges=np.linspace(0,r.max()+1e-9,bins+1);vals=[]
        for i in range(bins):m=(r>=edges[i])&(r<edges[i+1]);vals.append(np.mean(P[m]) if m.any() else 0)
        out.append(vals)
    return FeatureOutput(np.asarray(out),tuple(f'radial_band_{i:02d}' for i in range(bins)))

@_reg('image.frequency.dct','frequency',output_kind='vector',cost=Cost.MEDIUM,tags=('dct',),requires=('scipy',))
def dct_feature(x,size=4,**kwargs):
    from scipy.fft import dctn
    out=[]
    for g in gray_images(x):
        D=dctn(np.nan_to_num(g),norm='ortho');v=D[:size,:size].ravel();out.append(v)
    return FeatureOutput(np.asarray(out),tuple(f'c_{i:02d}_{j:02d}' for i in range(size) for j in range(size)))

@_reg('image.frequency.wavelet','frequency',output_kind='vector',cost=Cost.MEDIUM,tags=('haar','wavelet'))
def image_wavelet(x,levels=3,**kwargs):
    out=[]
    for g in gray_images(x):
        cur=np.nan_to_num(g);vals=[]
        for _ in range(levels):
            h,w=cur.shape;h=h//2*2;w=w//2*2
            if h<2 or w<2:vals.extend([np.nan]*3);continue
            z=cur[:h,:w];lo=(z[:,::2]+z[:,1::2])/np.sqrt(2);hi=(z[:,::2]-z[:,1::2])/np.sqrt(2);ll=(lo[::2]+lo[1::2])/np.sqrt(2);lh=(lo[::2]-lo[1::2])/np.sqrt(2);hl=(hi[::2]+hi[1::2])/np.sqrt(2);hh=(hi[::2]-hi[1::2])/np.sqrt(2);vals.extend([np.mean(lh*lh),np.mean(hl*hl),np.mean(hh*hh)]);cur=ll
        out.append(vals)
    return FeatureOutput(np.asarray(out),tuple(f'level_{l}_{b}' for l in range(1,levels+1) for b in ('LH','HL','HH')))

# Texture.
def _glcm_props(g,levels=16):
    from skimage.feature import graycomatrix, graycoprops
    u=_gray_uint8(g);u=(u.astype(int)*levels//256).clip(0,levels-1).astype(np.uint8)
    gl=graycomatrix(u,[1],[0,np.pi/4,np.pi/2,3*np.pi/4],levels=levels,symmetric=True,normed=True)
    props=['contrast','dissimilarity','homogeneity','ASM','energy','correlation']
    vals=[float(np.mean(graycoprops(gl,p))) for p in props]
    return vals,gl

@_reg('image.texture.glcm','texture',output_kind='vector',cost=Cost.MEDIUM,tags=('glcm',),requires=('scikit-image',))
def glcm(x,**kwargs):
    out=[_glcm_props(g)[0] for g in gray_images(x)];return FeatureOutput(np.asarray(out),('contrast','dissimilarity','homogeneity','asm','energy','correlation'))

@_reg('image.texture.haralick','texture',output_kind='vector',cost=Cost.MEDIUM,tags=('haralick',),requires=('scikit-image',))
def haralick(x,**kwargs):
    # Standard GLCM properties plus entropy/max-probability summaries.
    out=[]
    for g in gray_images(x):
        vals,gl=_glcm_props(g);P=gl.mean(axis=(2,3));ent=-np.sum(P*np.log(np.maximum(P,1e-15)));vals.extend([ent,P.max()]);out.append(vals)
    return FeatureOutput(np.asarray(out),('contrast','dissimilarity','homogeneity','asm','energy','correlation','entropy','max_probability'))

@_reg('image.texture.lbp','texture',output_kind='vector',cost=Cost.MEDIUM,tags=('lbp',),requires=('scikit-image',))
def lbp(x,P=8,R=1,bins=10,**kwargs):
    from skimage.feature import local_binary_pattern
    out=[]
    for g in gray_images(x):
        v=local_binary_pattern(_gray_uint8(g),P,R,method='uniform');h,_=np.histogram(v,bins=bins,range=(0,P+2));h=h/h.sum() if h.sum() else h;out.append(h)
    return FeatureOutput(np.asarray(out),tuple(f'bin_{i:02d}' for i in range(bins)))

@_reg('image.texture.gabor','texture',output_kind='vector',cost=Cost.HIGH,tags=('gabor',),requires=('skimage',))
def gabor(x,**kwargs):
    from skimage.filters import gabor
    freqs=(.1,.2,.3);thetas=(0,np.pi/4,np.pi/2,3*np.pi/4);out=[]
    for g in gray_images(x):
        u=_gray_uint8(g)/255.0;vals=[]
        for f in freqs:
            for th in thetas:
                re,im=gabor(u,frequency=f,theta=th);m=np.hypot(re,im);vals.append(np.mean(m))
        out.append(vals)
    return FeatureOutput(np.asarray(out),tuple(f'f{fi}_t{ti}' for fi in range(len(freqs)) for ti in range(len(thetas))))

@_reg('image.texture.laws','texture',output_kind='vector',cost=Cost.MEDIUM,tags=('laws-texture',),requires=('scipy',))
def laws(x,**kwargs):
    from scipy.signal import convolve2d
    vecs={'L5':np.array([1,4,6,4,1]),'E5':np.array([-1,-2,0,2,1]),'S5':np.array([-1,0,2,0,-1]),'R5':np.array([1,-4,6,-4,1])};pairs=[('L5','E5'),('E5','E5'),('S5','S5'),('R5','R5'),('E5','S5'),('L5','S5')];out=[]
    for g in gray_images(x):
        u=_gray_uint8(g).astype(float);vals=[]
        for a,b in pairs:
            k=np.outer(vecs[a],vecs[b]);resp=convolve2d(u,k,mode='same',boundary='symm');vals.append(np.mean(np.abs(resp)))
        out.append(vals)
    return FeatureOutput(np.asarray(out),tuple(f'{a}_{b}' for a,b in pairs))

@_reg('image.texture.fractal','texture',cost=Cost.MEDIUM,tags=('fractal','box-counting'))
def image_fractal(x,**kwargs):
    def one(g):
        Z=_gray_uint8(g)>np.mean(_gray_uint8(g));p=min(Z.shape);sizes=2**np.arange(int(np.log2(max(p,2))),1,-1);cnt=[];ss=[]
        for s in sizes:
            h,w=Z.shape;hh=h//s*s;ww=w//s*s
            if hh==0 or ww==0:continue
            blocks=Z[:hh,:ww].reshape(hh//s,s,ww//s,s);occ=blocks.any(axis=(1,3));c=occ.sum()
            if c>0:cnt.append(c);ss.append(s)
        return float(-np.polyfit(np.log(ss),np.log(cnt),1)[0]) if len(cnt)>1 else np.nan
    return np.asarray([one(g) for g in gray_images(x)])

# Quality and symmetry.
@_reg('image.quality.blur','quality',cost=Cost.LOW,tags=('blur','sharpness'),requires=('opencv-python',))
def blur(x,**kwargs):
    import cv2;return np.asarray([cv2.Laplacian(_gray_uint8(g),cv2.CV_64F).var() for g in gray_images(x)])

@_reg('image.quality.noise','quality',cost=Cost.LOW,tags=('noise',),requires=('scipy',))
def noise(x,**kwargs):
    from scipy.ndimage import gaussian_filter
    return np.asarray([np.std(np.asarray(g,float)-gaussian_filter(np.asarray(g,float),1)) for g in gray_images(x)])

@_reg('image.symmetry.horizontal','symmetry',cost=Cost.TINY,tags=('symmetry',))
def sym_h(x,**kwargs):
    out=[]
    for g in gray_images(x):
        h=g.shape[0]//2;a=g[:h];b=np.flipud(g[-h:]);out.append(1-np.mean(np.abs(a-b))/(np.std(g)+1e-12))
    return np.asarray(out)

@_reg('image.symmetry.vertical','symmetry',cost=Cost.TINY,tags=('symmetry',))
def sym_v(x,**kwargs):
    out=[]
    for g in gray_images(x):
        w=g.shape[1]//2;a=g[:,:w];b=np.fliplr(g[:,-w:]);out.append(1-np.mean(np.abs(a-b))/(np.std(g)+1e-12))
    return np.asarray(out)

# Binary-mask shape descriptors.
def _region(g):
    from skimage.measure import label,regionprops
    m=_gray_uint8(g)>np.mean(_gray_uint8(g));lab=label(m);props=regionprops(lab)
    return max(props,key=lambda p:p.area) if props else None,m

for _name,_getter in [
    ('area',lambda p: p.area),('perimeter',lambda p:p.perimeter),('eccentricity',lambda p:p.eccentricity),('solidity',lambda p:p.solidity),
]:
    @_reg(f'image.shape.{_name}','shape',cost=Cost.LOW,tags=('shape',),requires=('scikit-image',))
    def _shape(x,_getter=_getter,**kwargs):
        out=[]
        for g in gray_images(x):
            p,_=_region(g);out.append(float(_getter(p)) if p is not None else 0.0)
        return np.asarray(out)

@_reg('image.shape.circularity','shape',cost=Cost.LOW,tags=('shape',),requires=('scikit-image',))
def circularity(x,**kwargs):
    out=[]
    for g in gray_images(x):
        p,_=_region(g);out.append(4*np.pi*p.area/(p.perimeter**2+1e-12) if p is not None else 0)
    return np.asarray(out)

@_reg('image.shape.hu_moments','shape',output_kind='vector',cost=Cost.LOW,tags=('hu-moments',),requires=('scikit-image',))
def hu(x,**kwargs):
    from skimage.measure import moments_central,moments_normalized,moments_hu
    out=[]
    for g in gray_images(x):
        _,m=_region(g);mu=moments_central(m.astype(float));nu=moments_normalized(mu);out.append(moments_hu(nu))
    return FeatureOutput(np.asarray(out),tuple(f'hu_{i}' for i in range(7)))

@_reg('image.shape.fourier_descriptors','shape',output_kind='vector',cost=Cost.MEDIUM,tags=('fourier-descriptor',),requires=('opencv-python',))
def fourier_desc(x,n=10,**kwargs):
    import cv2;out=[]
    for g in gray_images(x):
        m=(_gray_uint8(g)>np.mean(_gray_uint8(g))).astype(np.uint8)*255;cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if not cs:out.append([0]*n);continue
        c=max(cs,key=cv2.contourArea).reshape(-1,2);z=c[:,0]+1j*c[:,1];F=np.fft.fft(z-z.mean());v=np.abs(F[1:n+1]);v=v/(v[0]+1e-12) if len(v) else v;out.append(np.pad(v,(0,n-len(v))))
    return FeatureOutput(np.asarray(out),tuple(f'fd_{i:02d}' for i in range(1,n+1)))

@_reg('image.shape.zernike','shape',output_kind='vector',cost=Cost.HIGH,tags=('zernike',))
def zernike(x,orders=((0,0),(1,1),(2,0),(2,2),(3,1),(3,3),(4,0),(4,2),(4,4)),**kwargs):
    # Rotation-aware low-order Zernike magnitudes on a unit disk.
    def radial(n,m,r):
        R=np.zeros_like(r,float)
        for s in range((n-m)//2+1):
            c=(-1)**s*math.factorial(n-s)/(math.factorial(s)*math.factorial((n+m)//2-s)*math.factorial((n-m)//2-s));R+=c*r**(n-2*s)
        return R
    out=[]
    for g in gray_images(x):
        u=_gray_uint8(g)/255.0;h,w=u.shape;y,xs=np.indices(u.shape);xx=2*(xs-(w-1)/2)/max(w-1,1);yy=2*(y-(h-1)/2)/max(h-1,1);rr=np.sqrt(xx*xx+yy*yy);th=np.arctan2(yy,xx);mask=rr<=1;vals=[]
        for n,m in orders:
            V=radial(n,m,rr)*np.exp(-1j*m*th);z=(n+1)/np.pi*np.sum(u[mask]*V[mask]);vals.append(abs(z))
        out.append(vals)
    return FeatureOutput(np.asarray(out),tuple(f'n{n}_m{m}' for n,m in orders))

@_reg('image.morphology.components','morphology',output_kind='vector',cost=Cost.LOW,tags=('components',),requires=('scikit-image',))
def components(x,**kwargs):
    from skimage.measure import label,regionprops
    out=[]
    for g in gray_images(x):
        m=_gray_uint8(g)>np.mean(_gray_uint8(g));ps=regionprops(label(m));areas=np.array([p.area for p in ps]);out.append([len(ps),areas.mean() if len(areas) else 0,areas.max() if len(areas) else 0])
    return FeatureOutput(np.asarray(out),('count','area_mean','area_max'))

# Keypoints.
def _kp(x,kind):
    import cv2;out=[]
    for g in gray_images(x):
        u=_gray_uint8(g)
        if kind=='sift':det=cv2.SIFT_create()
        elif kind=='orb':det=cv2.ORB_create()
        else:det=cv2.FastFeatureDetector_create()
        kp=det.detect(u,None);resp=[k.response for k in kp];size=[k.size for k in kp];out.append([len(kp),np.mean(resp) if resp else 0,np.mean(size) if size else 0])
    return FeatureOutput(np.asarray(out),('count','response_mean','size_mean'))
for _kind in ('sift','orb','fast'):
    @_reg(f'image.keypoints.{_kind}','keypoints',output_kind='vector',cost=Cost.MEDIUM,tags=('keypoints',_kind),requires=('opencv-python',))
    def _key(x,_kind=_kind,**kwargs):return _kp(x,_kind)

# Topology of thresholded images.
@_reg('image.topology.euler_number','topology',cost=Cost.LOW,tags=('euler',),requires=('scikit-image',))
def euler(x,**kwargs):
    from skimage.measure import euler_number
    return np.asarray([euler_number(_gray_uint8(g)>np.mean(_gray_uint8(g)),connectivity=2) for g in gray_images(x)],float)

@_reg('image.topology.holes','topology',cost=Cost.LOW,tags=('holes',),requires=('scikit-image',))
def holes(x,**kwargs):
    from skimage.measure import label,euler_number
    out=[]
    for g in gray_images(x):
        m=_gray_uint8(g)>np.mean(_gray_uint8(g));comp=label(m,connectivity=2).max();eu=euler_number(m,connectivity=2);out.append(comp-eu)
    return np.asarray(out,float)

@_reg('image.topology.persistent_homology','topology',output_kind='vector',cost=Cost.HIGH,tags=('persistent-homology','filtration'),requires=('scikit-image',))
def image_ph(x,thresholds=8,**kwargs):
    from skimage.measure import label,euler_number
    out=[]
    for g in gray_images(x):
        u=_gray_uint8(g);ths=np.linspace(u.min(),u.max(),thresholds);vals=[]
        for t in ths:
            m=u>=t;c=label(m,connectivity=2).max();h=c-euler_number(m,connectivity=2);vals.extend([c,h])
        out.append(vals)
    return FeatureOutput(np.asarray(out),tuple(v for i in range(thresholds) for v in (f'beta0_{i:02d}',f'beta1_{i:02d}')))

# Saliency via spectral residual.
@_reg('image.saliency.statistics','saliency',output_kind='vector',cost=Cost.MEDIUM,tags=('saliency',),requires=('scipy',))
def saliency(x,**kwargs):
    from scipy.ndimage import gaussian_filter
    out=[]
    for g in gray_images(x):
        F=np.fft.fft2(np.nan_to_num(g));log=np.log(np.abs(F)+1e-12);phase=np.angle(F);avg=gaussian_filter(log,3);res=log-avg;S=np.abs(np.fft.ifft2(np.exp(res+1j*phase)))**2;S=gaussian_filter(S,2);out.append([S.mean(),S.std(),S.max(),np.quantile(S,.95)])
    return FeatureOutput(np.asarray(out),('mean','std','max','q95'))

# Learned image embeddings using timm when explicitly selected.
def _timm_embed(x,model_name,pretrained=True):
    try:
        import torch, timm
    except ImportError as exc:raise ImportError("Install 'timm' and 'torch' for image embedding features") from exc
    try:
        model=timm.create_model(model_name,pretrained=pretrained,num_classes=0,global_pool='avg')
    except RuntimeError as exc:
        # Some checkpoints (e.g. DINOv2's vit_*_dinov2.* tags) ship no
        # fc_norm weights because they were trained for CLS-token pooling,
        # not spatial-average pooling -- global_pool='avg' asks timm to add
        # a fresh fc_norm layer and then (on this timm version) tries to
        # strictly match it against the checkpoint's state_dict anyway,
        # which always fails for such a checkpoint, not intermittently.
        # Retry with token (CLS) pooling, which is what these checkpoints
        # actually expect, before giving up -- narrowly scoped to the
        # state_dict-shape failure so an unrelated bad model_name still
        # raises immediately instead of silently retrying forever.
        if "state_dict" not in str(exc) and "fc_norm" not in str(exc):
            raise
        model=timm.create_model(model_name,pretrained=pretrained,num_classes=0,global_pool='token')
    model.eval();a=as_images(x)
    if a.ndim==3:a=np.repeat(a[...,None],3,axis=-1)
    # Resize to each model's own expected input resolution (read from its
    # pretrained_cfg) rather than a hardcoded 224 -- some checkpoints (e.g.
    # DINOv2's patch14 variant, native 518x518) are not 224-native and
    # silently produce degraded/incorrect features (or, on this timm
    # version, a hard AssertionError from patch_embed) if force-fed 224.
    cfg_size=getattr(model,'pretrained_cfg',{}) or {}
    input_size=cfg_size.get('input_size')
    target_hw=tuple(input_size[-2:]) if input_size else (224,224)
    rows=[]
    for im in a:
        t=torch.tensor(im[...,:3].transpose(2,0,1),dtype=torch.float32)[None]
        if t.max()>1.5:t=t/255.0
        t=torch.nn.functional.interpolate(t,size=target_hw,mode='bilinear',align_corners=False)
        with torch.no_grad():v=model(t).squeeze(0).cpu().numpy()
        rows.append(v)
    return FeatureOutput(np.asarray(rows),tuple(f'dim_{i:04d}' for i in range(len(rows[0]))))

for _name,_model in [('resnet','resnet50'),('efficientnet','efficientnet_b0'),('convnext','convnext_tiny'),('vit','vit_small_patch16_224'),('dino','vit_small_patch14_dinov2.lvd142m'),('timm','resnet18')]:
    @_reg(f'image.embeddings.{_name}','embeddings',output_kind='vector',cost=Cost.EXTREME,tags=('embedding','pretrained'),requires=('torch','timm'))
    def _emb(x,model_name=_model,**kwargs):return _timm_embed(x,model_name=model_name,**kwargs)
    _emb.__featm_parameters__ = forward_parameters(_timm_embed, positional=1, bound=['model_name'])

@_reg('image.embeddings.clip','embeddings',output_kind='vector',cost=Cost.EXTREME,tags=('clip','embedding'),requires=('torch','transformers'))
def clip_embedding(x,model_name='openai/clip-vit-base-patch32',**kwargs):
    try:
        import torch
        from transformers import CLIPModel,CLIPProcessor
        from PIL import Image
    except ImportError as exc:raise ImportError("Install torch, transformers, and pillow for CLIP embeddings") from exc
    model=CLIPModel.from_pretrained(model_name);proc=CLIPProcessor.from_pretrained(model_name);model.eval();rows=[]
    for im in as_images(x):
        u=np.nan_to_num(im)
        if u.max()<=1.5:u*=255
        if u.ndim==2:u=np.repeat(u[...,None],3,axis=-1)
        pil=Image.fromarray(np.clip(u[...,:3],0,255).astype(np.uint8));inp=proc(images=pil,return_tensors='pt')
        with torch.no_grad():v=model.get_image_features(**inp).squeeze(0).cpu().numpy()
        rows.append(v)
    return FeatureOutput(np.asarray(rows),tuple(f'dim_{i:04d}' for i in range(len(rows[0]))))
