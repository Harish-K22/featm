# Evidence and benchmark interpretation

Feature extraction is a representation choice, not a guarantee of predictive
improvement. The feature list, output dimensions, classifier, split protocol,
preprocessing and fitting boundaries all affect a comparison. A larger
catalogue does not establish superiority over another library.

## Release validation

The accepted v0.5.7 RC completed installed-wheel regressions with 802 passed/
9 skipped on Windows and 798 passed/13 skipped on Ubuntu/WSL, with no failures.
It retained 552 entries, 18 modalities, 182 modality/family groups, seven
stateful entries and 22 GPU-capable methods. These are correctness/inventory
checks, not benchmark scores. Real Windows CUDA and multiprocess Dask checks
also ran within the [documented support scope](COMPATIBILITY.md).

Historical system, predictive-utility and competitor investigations predate
the public preparation. Their internal handoffs are not shipped as public
evidence links here. No new competitor study was run for this release, and no
unverifiable headline performance number is promoted in the README.

## Practical datasets

[Examples](PRACTICAL_EXAMPLES.md) show how to construct and extract samples.
They make no performance claims. Local end-user observations informed the
choice of examples but are not a reproducible public benchmark package:
manually chosen audio features can help one task; ECG summary features can add
little to raw waveforms; categorical frequency transforms can be redundant.
The supplied CIFAR-10 model comparison had convergence warnings, so it is not
presented as a performance victory. No numerical comparison is published here.

To publish a future comparison, retain data provenance/checksums and license,
exact ordered feature lists and dimensions, package versions, model parameters,
seeds, train/test or fold IDs, training-only fitted state, convergence checks,
per-fold scores and aggregation. Compare on identical held-out splits. Report
negative results and the strengths of task-specific alternatives; runtime,
predictive quality and breadth answer different questions. In-sample feature
importance is not cross-validated predictive gain.
