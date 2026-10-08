from __future__ import annotations

import numpy as np

from ...registry import BatchSemantics, Cost, register_feature


def _images(x) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim not in (3, 4):
        raise ValueError("images must be (n,h,w) or (n,h,w,c)")
    return a.astype(np.float64, copy=False)


@register_feature(
    name="image.intensity.mean",
    family="intensity",
    modality="image",
    input_types=("image",),
    cost=Cost.TINY,
    streamable=True,
    description="Per-image mean intensity over all pixels/channels.",
    tags=("intensity", "basic"),
    batch_semantics=BatchSemantics.OPAQUE_OBJECT_SAMPLE,
)
def image_mean(images, **kwargs):
    a = _images(images)
    axes = tuple(range(1, a.ndim))
    return np.nanmean(a, axis=axes)


@register_feature(
    name="image.intensity.std",
    family="intensity",
    modality="image",
    input_types=("image",),
    cost=Cost.TINY,
    streamable=True,
    description="Per-image intensity standard deviation.",
    tags=("intensity", "contrast"),
    batch_semantics=BatchSemantics.OPAQUE_OBJECT_SAMPLE,
)
def image_std(images, **kwargs):
    a = _images(images)
    axes = tuple(range(1, a.ndim))
    return np.nanstd(a, axis=axes)


@register_feature(
    name="image.edges.gradient_energy",
    family="edges",
    modality="image",
    input_types=("image",),
    cost=Cost.LOW,
    streamable=True,
    description="Mean finite-difference gradient magnitude per image.",
    tags=("edges", "gradient", "sharpness"),
    batch_semantics=BatchSemantics.OPAQUE_OBJECT_SAMPLE,
)
def gradient_energy(images, **kwargs):
    a = _images(images)
    if a.ndim == 4:
        a = a.mean(axis=-1)
    gy = np.diff(a, axis=1, append=a[:, -1:, :])
    gx = np.diff(a, axis=2, append=a[:, :, -1:])
    return np.sqrt(gx * gx + gy * gy).mean(axis=(1, 2))
