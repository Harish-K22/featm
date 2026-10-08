# Practical extraction examples

Start with the [small synthetic scripts](../examples/README.md). They require
no datasets or model downloads, and illustrate explicit method choice rather
than automatic selection. Real-data recipes below accept files you obtain
yourself. Keep datasets outside the repository and retain source checksums,
version, original labels/splits and attribution. No predictive scores are
claimed by these examples.

## ESC-10 audio

Use Karol J. Piczak's [official ESC-50 repository](https://github.com/karolpiczak/ESC-50)
and its download instructions. ESC-10 is the subset selected by the `esc10`
column in `meta/esc50.csv`; retain the supplied folds and clip attribution.
The provider distinguishes ESC-10's CC BY terms from the full dataset's
CC BY-NC terms; consult its LICENSE for individual recordings.

With the audio extra installed, pass the extracted root containing `audio/`
and `meta/`:

```bash
python examples/audio_features.py --esc-root "../datasets/ESC-50-master" --limit 8
```

The script preserves the loaded sample rate, selects mono/equal-length clips,
and explicitly requests 13 MFCC columns plus RMS and spectral centroid: 15
illustrative columns, not the original end-user 21-column experiment.
It does not train a classifier or reproduce that experiment's macro-F1.
The known short harmonic/tonnetz limitation remains [documented](COMPATIBILITY.md).

## ECG5000 time series

Obtain ECG5000 through the [UCR/UEA dataset page](https://www.timeseriesclassification.com/description.php?Dataset=ECG5000).
Retain original train/test partitions and provider/PhysioNet provenance.
The inspected dataset page does not establish a dataset redistribution license;
check provider terms rather than applying FEATM's MIT notice to the data.

For a label-first whitespace text file, run:

```bash
python examples/ecg_features.py --input "../datasets/ECG5000/ECG5000_TRAIN.txt"
```

For a comma-delimited variant, add `--delimiter ","`. Labels are separated
from waveform columns before extraction; each row remains one heartbeat.
The script computes three illustrative statistics, not the original ten-feature
classifier experiment. A compact summary can lose information in a waveform;
feature extraction is not a clinical interpretation or a guarantee of gain.

## CIFAR-10 images

Download CIFAR-10 using the [dataset author's page](https://cave.cs.toronto.edu/kriz/cifar.html)
and retain the dataset citation/source checksum. That inspected page did not
state a redistribution license; check its provider terms before redistribution.
Prepare grayscale arrays explicitly as `(n_samples, height, width)` in a local
`.npy` file. For channel-last RGB arrays, a documented preprocessing choice is
`gray = np.round(rgb.mean(axis=-1)).astype(np.uint8)`; retain that choice in any
experiment record instead of assuming FEATM makes the conversion silently.

```bash
python examples/image_features.py --input "../datasets/cifar10_gray.npy"
```

GLCM/LBP produce multiple named columns. This is extraction only, not a
reproduction of the original 25-column experiment. No CIFAR performance
comparison is published: the supplied raw/combined SGD runs had convergence
warnings, so a converged, reproducible baseline is needed first.

## Tabular health-category example

The [training-only frequency example](../examples/stateful_categories.py)
demonstrates six training observations without private health data.
Frequency encodings can be redundant with existing categorical representations;
do not infer predictive improvement from adding columns. No Student Health Risk
dataset URL, license or numerical result is published because its exact source
and reproducible experiment were not established here.
