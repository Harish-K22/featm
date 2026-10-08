"""Vision extra: synthetic image batch or an explicitly prepared CIFAR grayscale .npy."""
import argparse
from pathlib import Path
import numpy as np
import featm

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path)
args = parser.parse_args()
if args.input:
    X = np.load(args.input, allow_pickle=False)
else:
    X = np.random.default_rng(7).integers(0, 256, size=(2, 32, 32), dtype=np.uint8)
if X.ndim != 3:
    raise ValueError("This example expects (n_samples, height, width) grayscale arrays")
names = ["image.texture.glcm", "image.texture.lbp"]
result = featm.extract(X, features=names, errors="raise")
assert len(result) == len(X)
print(result.shape, result.columns.tolist())
