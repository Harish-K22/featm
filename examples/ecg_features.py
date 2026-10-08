"""Base-only ECG-style features; optional UCR whitespace/comma label-first file."""
import argparse
from pathlib import Path
import numpy as np
import featm

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path)
parser.add_argument("--delimiter", default=None)
args = parser.parse_args()
if args.input:
    table = np.loadtxt(args.input, delimiter=args.delimiter, ndmin=2)
    labels, X = table[:, 0], table[:, 1:]
    if not X.shape[1]:
        raise ValueError("Expected class label followed by waveform observations")
else:
    X = np.stack([np.sin(np.linspace(0, 4*np.pi, 140)), np.cos(np.linspace(0, 4*np.pi, 140))])
names = ["timeseries.statistical.mean", "timeseries.statistical.std", "timeseries.spectral.spectral_entropy"]
result = featm.extract(X, features=names, errors="raise")
assert result.shape == (len(X), 3)
print(result)
