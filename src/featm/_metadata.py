'Curated built-in metadata; no imports of optional backends or feature modules.\n\nRequirements are descriptive distribution names, not eager execution checks.\nConditional alternatives are documented in descriptions/docs, not enforced here.'

REQUIREMENTS = {'audio.perceptual.gfcc': ('scipy',),
 'graph.community.leiden': ('igraph', 'leidenalg', 'networkx'),
 'graph.embeddings.deepwalk': ('gensim', 'networkx'),
 'graph.embeddings.gnn': ('user-model', 'networkx'),
 'graph.embeddings.node2vec': ('node2vec', 'networkx'),
 'image.embeddings.clip': ('torch', 'transformers', 'Pillow'),
 'pointcloud.geometry.curvature': ('scipy',),
 'pointcloud.geometry.density': ('scipy',),
 'pointcloud.geometry.normals': ('scipy',),
 'pointcloud.geometry.roughness': ('scipy',),
 'pointcloud.local_geometry.anisotropy': ('scipy',),
 'pointcloud.local_geometry.eigenentropy': ('scipy',),
 'pointcloud.local_geometry.linearity': ('scipy',),
 'pointcloud.local_geometry.omnivariance': ('scipy',),
 'pointcloud.local_geometry.pca_linearity': ('scipy',),
 'pointcloud.local_geometry.pca_planarity': ('scipy',),
 'pointcloud.local_geometry.pca_sphericity': ('scipy',),
 'pointcloud.local_geometry.planarity': ('scipy',),
 'pointcloud.local_geometry.scattering': ('scipy',),
 'pointcloud.surface.surface_variation': ('scipy',),
 'pointcloud.topology.persistence': ('scipy',),
 'pointcloud.triangulation.delaunay': ('scipy',),
 'signal.envelope.hilbert': ('scipy',),
 'signal.modulation.amplitude_modulation': ('scipy',),
 'signal.nonlinear.lyapunov': ('scipy',),
 'signal.peaks.peak_statistics': ('scipy',),
 'signal.phase.instantaneous_phase': ('scipy',),
 'signal.synchronization.phase_locking': ('scipy',),
 'signal.time_frequency.stft': ('scipy',),
 'signal.wavelet.cwt': ('scipy',),
 'spatial.clustering.spatial_clustering': ('scipy', 'scikit-learn'),
 'spatial.density.kernel_density': ('scipy', 'scikit-learn'),
 'tabular.datetime.holidays': ('holidays',),
 'timeseries.cross_series.coherence': ('scipy',),
 'timeseries.cross_series.phase_synchrony': ('scipy',),
 'timeseries.dynamical.correlation_dimension': ('scipy',),
 'timeseries.dynamical.embedding_dimension': ('scipy',),
 'timeseries.dynamical.lyapunov': ('scipy',),
 'timeseries.recurrence.determinism': ('scipy',),
 'timeseries.recurrence.laminarity': ('scipy',),
 'timeseries.recurrence.recurrence_rate': ('scipy',),
 'timeseries.recurrence.trapping_time': ('scipy',),
 'timeseries.recurrence.recurrence_network': ('scipy',),
 'timeseries.statistical.kurtosis': ('scipy',),
 'timeseries.statistical.skewness': ('scipy',),
 'timeseries.topology.betti_curves': ('scipy',),
 'timeseries.topology.persistence': ('scipy',),
 'timeseries.topology.persistence_image': ('scipy',),
 'timeseries.topology.persistence_landscape': ('scipy',),
 'timeseries.trend.robust': ('scipy',),
 'timeseries.wavelet.cwt': ('scipy',),
 'timeseries.wavelet.scattering': ('scipy',),
 'topology.local_topology.local_persistence': ('scipy',),
 'topology.persistence.entropy': ('scipy',),
 'topology.persistence.image': ('scipy',),
 'topology.persistence.landscape': ('scipy',)}

DESCRIPTIONS = {'audio.perceptual.gfcc': 'ERB-spaced triangular-filterbank log energies transformed by SciPy '
                          'orthonormal type-II DCT; sr=22050, n_bands=32, first n_gfcc=13 '
                          'coefficients per waveform.',
 'audio.perceptual.loudness': '20*log10(max(waveform RMS,1e-12)) per row; amplitude-relative '
                              'decibels, not calibrated sound-pressure level.',
 'audio.perceptual.mfcc': 'Temporal mean of librosa MFCC coefficients per waveform; sr=22050 and '
                          'n_mfcc=13 give 13 output values per row.',
 'audio.spectral.centroid': 'Mean over frames of librosa spectral centroid in Hz per waveform; '
                            'sr=22050 controls sample rate. One scalar per waveform.',
 'audio.spectral.centroid_normalized': 'Mean librosa spectral centroid divided by Nyquist '
                                       'frequency (sr/2), one scalar per waveform; sr=22050.',
 'audio.spectral.rolloff': 'Mean over frames of librosa spectral rolloff at its default 85% energy '
                           'fraction, in Hz per waveform; sr=22050.',
 'audio.waveform.crest_factor': 'Peak absolute waveform amplitude divided by RMS per row; where '
                                'RMS is not positive the output is zero. One scalar per waveform.',
 'audio.waveform.rms': 'Square root of mean squared waveform samples per row (1D input is one '
                       'waveform); one scalar. NaNs propagate.',
 'audio.waveform.zero_crossing_rate': 'Fraction of adjacent waveform sign-bit changes per row; '
                                      'zero is nonnegative, NaNs are not filtered. One scalar per '
                                      'waveform.',
 'image.intensity.mean': 'NaN-ignoring mean over all pixels and channels per image; input (N,H,W) '
                         'or (N,H,W,C), one scalar per image. A bare RGB image needs a batch axis.',
 'image.intensity.std': 'NaN-ignoring population intensity standard deviation (ddof=0) over all '
                        'pixels/channels per image; input (N,H,W) or (N,H,W,C), one scalar per '
                        'image.',
 'timeseries.autocorrelation.acf': 'Pearson correlations of each finite-filtered series with its '
                                   'lagged copies at lags 1..max_lag (default 10); one vector per '
                                   'row. Short or constant slices yield NaN.',
 'timeseries.autocorrelation.autocovariance': 'Mean lagged products after subtracting the row '
                                              'nanmean and replacing NaNs by zero; max_lag=8, one '
                                              'vector per row. Insufficient length yields NaN.',
 'timeseries.autocorrelation.lag1': 'Pearson correlation of adjacent samples after filtering '
                                    'nonfinite values per series row; one scalar. Short or '
                                    'constant slices yield NaN.',
 'timeseries.basic.differences': 'Three statistics of adjacent first differences per series row: '
                                 'NaN-ignoring mean, population standard deviation and median.',
 'timeseries.basic.lags': 'Mean raw lagged products at fixed lags 1, 2, 4 and 8 per series row, '
                          'ignoring NaNs; not normalized autocorrelation. Too-short lags yield '
                          'NaN.',
 'timeseries.basic.rolling': 'Four summaries per series row of sliding means and population '
                             'standard deviations: mean/std of means, mean/max of stds. window=8; '
                             'shorter rows yield four NaNs.',
 'timeseries.complexity.approximate_entropy': 'Approximate entropy per finite-filtered series row '
                                              'with fixed m=2 and r=0.2*population std; includes '
                                              'self matches and uses natural logarithms. Constant '
                                              'rows return zero; short rows NaN.',
 'timeseries.complexity.permutation_entropy': 'Normalized ordinal-pattern Shannon entropy per '
                                              'series row; fixed order=3 and delay=1, natural '
                                              'logarithms divided by log(3!), stable ties. '
                                              'Nonfinite windows are omitted; no valid windows '
                                              'yields NaN.',
 'timeseries.complexity.sample_entropy': 'Sample entropy per finite-filtered series row with fixed '
                                         'm=2 and r=0.2*population std; unordered Chebyshev '
                                         'matches exclude self. Constant rows return zero; short '
                                         'rows NaN; zero counts use implementation-specific '
                                         'infinity/log(b+1) handling.',
 'timeseries.complexity.zero_crossing_rate': 'Fraction of adjacent sign-bit changes per series '
                                             'row; zero has a nonnegative sign. One scalar per '
                                             'row; NaNs are not filtered.',
 'timeseries.fractal.dfa': 'Log-log slope of mean segment RMS after linear detrending of the '
                           'cumulative demeaned series; fixed log-spaced scales, NaNs '
                           'mean-imputed. Fewer than 16 samples or insufficient positive scales '
                           'yields NaN.',
 'timeseries.fractal.higuchi': 'Higuchi log-log length slope per series row with fixed kmax=8 and '
                               'NaNs median-imputed; rows shorter than 10 or without enough '
                               'positive lengths yield NaN.',
 'timeseries.fractal.hurst': 'Log-log slope of lagged-difference standard deviations per '
                             'finite-filtered series row; fewer than 20 samples or insufficient '
                             'positive scales yield NaN.',
 'timeseries.spectral.centroid': 'Magnitude-weighted centroid of the mean-centered one-sided FFT '
                                 'per series row, in cycles per sample; zero total magnitude '
                                 'yields zero.',
 'timeseries.spectral.dominant_frequency': 'Frequency of the largest mean-centered one-sided FFT '
                                           'power bin per series row, in cycles per sample; ties '
                                           'choose the first bin.',
 'timeseries.spectral.flatness': 'Geometric mean of power floored at 1e-15 divided by arithmetic '
                                 'mean power of the mean-centered one-sided FFT per series row; '
                                 'zero-power rows yield zero.',
 'timeseries.spectral.psd': 'Welch power spectral density per series row using SciPy defaults '
                            '(fs=1, nperseg=min(256,length)); first n_bins=16 values, padded with '
                            'NaNs when shorter.',
 'timeseries.spectral.spectral_entropy': 'Normalized natural-log Shannon entropy of uncentered '
                                         'one-sided FFT power per series row, divided by '
                                         'log(max(number of bins,2)); zero-power rows yield zero.',
 'timeseries.statistical.change_statistics': 'Four statistics of first differences per series row: '
                                             'mean, population standard deviation, mean absolute '
                                             'difference and maximum absolute difference, ignoring '
                                             'NaNs.',
 'timeseries.statistical.energy': 'Sum of squared values per series row, ignoring NaNs; one scalar '
                                  'per row. An all-NaN row yields zero.',
 'timeseries.statistical.extrema': 'Four values per series row: NaN-ignoring minimum, maximum and '
                                   'their zero-based first indices; NaNs are replaced by signed '
                                   'infinities for index searches.',
 'timeseries.statistical.kurtosis': 'Bias-corrected Fisher excess kurtosis per series row via '
                                    'SciPy (bias=False, nan_policy=omit); normal-reference value '
                                    'zero, constant rows NaN.',
 'timeseries.statistical.max': 'Maximum per series row, ignoring NaNs; one scalar per row. An '
                               'all-NaN row yields NaN.',
 'timeseries.statistical.mean': 'Arithmetic mean per input series (1D or rows of a 2D array), '
                                'ignoring NaNs; one scalar per row.',
 'timeseries.statistical.median': 'Median per series row, ignoring NaNs; one scalar per row.',
 'timeseries.statistical.min': 'Minimum per series row, ignoring NaNs; one scalar per row. An '
                               'all-NaN row yields NaN.',
 'timeseries.statistical.range': 'Maximum minus minimum per series row, ignoring NaNs; one scalar '
                                 'per row.',
 'timeseries.statistical.rms': 'Square root of the mean squared values per series row, ignoring '
                               'NaNs; one scalar per row.',
 'timeseries.statistical.skewness': 'Bias-corrected SciPy skewness per series row (bias=False, '
                                    'nan_policy=omit); one scalar per row. Constant rows yield '
                                    'NaN.',
 'timeseries.statistical.std': 'Population standard deviation (ddof=0) per series row, ignoring '
                               'NaNs; one scalar per row.',
 'timeseries.trend.linear': 'Least-squares slope, intercept and R-squared against zero-based '
                            'sample indices per series row; nonfinite values are excluded. '
                            'Constant rows have NaN R-squared.',
 'timeseries.trend.linear_slope': 'Least-squares slope against zero-based sample indices per '
                                  'series row, excluding nonfinite values; units are amplitude per '
                                  'sample. Fewer than two finite samples yield NaN.',
 'timeseries.trend.robust': 'SciPy Theil-Sen slope, intercept and default 95% slope interval per '
                            'series row against sample indices, excluding nonfinite values; '
                            'fitting exceptions yield four NaNs.',
 'topology.betti.numbers': 'topology.betti.numbers. Dependencies are conditional: GUDHI '
                           'persistence is attempted first; if unavailable or unsuccessful, SciPy '
                           'computes an H0-only MST fallback. The flat requires tuple cannot '
                           'encode this alternative.',
 'topology.euler.characteristic': 'topology.euler.characteristic. Dependencies are conditional: '
                                  'GUDHI persistence is attempted first; if unavailable or '
                                  'unsuccessful, SciPy computes an H0-only MST fallback. The flat '
                                  'requires tuple cannot encode this alternative.',
 'topology.persistent_homology.alpha_complex': 'topology.persistent_homology.alpha_complex. '
                                               'Dependencies are conditional: GUDHI persistence is '
                                               'attempted first; if unavailable or unsuccessful, '
                                               'SciPy computes an H0-only MST fallback. The flat '
                                               'requires tuple cannot encode this alternative.',
 'topology.persistent_homology.cech': 'topology.persistent_homology.cech. Dependencies are '
                                      'conditional: GUDHI persistence is attempted first; if '
                                      'unavailable or unsuccessful, SciPy computes an H0-only MST '
                                      'fallback. The flat requires tuple cannot encode this '
                                      'alternative.',
 'topology.persistent_homology.vietoris_rips': 'topology.persistent_homology.vietoris_rips. '
                                               'Dependencies are conditional: GUDHI persistence is '
                                               'attempted first; if unavailable or unsuccessful, '
                                               'SciPy computes an H0-only MST fallback. The flat '
                                               'requires tuple cannot encode this alternative.',
 'topology.persistent_homology.witness_complex': 'topology.persistent_homology.witness_complex. '
                                                 'Dependencies are conditional: GUDHI persistence '
                                                 'is attempted first; if unavailable or '
                                                 'unsuccessful, SciPy computes an H0-only MST '
                                                 'fallback. The flat requires tuple cannot encode '
                                                 'this alternative.'}

def builtin_metadata(func, metadata):
    """Apply reviewed metadata only to FEATM built-ins, leaving plugins alone."""
    if not (getattr(func, "__module__", "") or "").startswith("featm.features."):
        return metadata
    name = metadata.get("name")
    result = dict(metadata)
    if name in REQUIREMENTS:
        result["requires"] = REQUIREMENTS[name]
    if name in DESCRIPTIONS:
        result["description"] = DESCRIPTIONS[name]
    return result
