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

DESCRIPTIONS = {'audio.embeddings.clap': 'Audio-tower embedding from default laion/clap-htsat-unfused via Hugging '
                          "Face get_audio_features; generic sr=16000 defers to the processor's "
                          'native rate without resampling input. Requires model weights.',
 'audio.embeddings.hubert': 'Hugging Face AutoModel waveform embedding, default '
                            'facebook/hubert-base-ls960; processor falls back to '
                            'AutoFeatureExtractor, temporal hidden states are mean-pooled. '
                            'sr=16000 labels input without resampling; requires model weights.',
 'audio.embeddings.wav2vec': 'Hugging Face AutoModel waveform embedding, default '
                             'facebook/wav2vec2-base-960h; mean-pools temporal hidden states per '
                             'waveform, sr=16000 labels input and does not resample it. Requires '
                             'model downloads or cache.',
 'audio.embeddings.whisper': 'Mean-pooled encoder hidden-state embedding from default '
                             'openai/whisper-tiny, using Hugging Face processing; sr=16000 labels '
                             'input without resampling, model weights must be available.',
 'audio.perceptual.bark': 'Log1p triangular-filterbank energies from centered FFT power on an '
                          'interpolated Bark scale; sr=22050, n_bands=24 gives 24 values per '
                          'waveform.',
 'audio.perceptual.gfcc': 'ERB-spaced triangular-filterbank log energies transformed by SciPy '
                          'orthonormal type-II DCT; sr=22050, n_bands=32, first n_gfcc=13 '
                          'coefficients per waveform.',
 'audio.perceptual.loudness': '20*log10(max(waveform RMS,1e-12)) per row; amplitude-relative '
                              'decibels, not calibrated sound-pressure level.',
 'audio.perceptual.mel': 'Temporal mean of log1p librosa mel-spectrogram values; sr=22050, '
                         'n_mels=40 yields 40 values per waveform, without decibel conversion.',
 'audio.perceptual.mfcc': 'Temporal mean of librosa MFCC coefficients per waveform; sr=22050 and '
                          'n_mfcc=13 give 13 output values per row.',
 'audio.rhythm.beats': 'librosa beat count, mean beat interval and population interval standard '
                       'deviation in seconds; sr=22050, fewer than two beats give NaN interval '
                       'summaries.',
 'audio.rhythm.onset': 'librosa detected-onset count plus mean and maximum of the complete '
                       'onset-strength envelope; sr=22050, empty envelopes give zero strengths.',
 'audio.rhythm.tempo': 'First librosa tempo estimate from the waveform onset-strength envelope, in '
                       'beats per minute; sr=22050, empty estimates yield NaN.',
 'audio.rhythm.tempogram': 'Time-averaged librosa onset tempogram sampled at n_bins=16 evenly '
                           'spaced lag indices; sr=22050, output indices are not calibrated BPM '
                           'bins.',
 'audio.spatial.ild': '20*log10((left RMS+1e-12)/(right RMS+1e-12)) per stereo waveform; input '
                      '(N,T,2) or (T,2), one relative level difference in decibels.',
 'audio.spatial.itd': 'Lag maximizing centered left/right cross-correlation within max_ms=2 '
                      'divided by sr=22050; returns seconds per stereo waveform without sub-sample '
                      'interpolation.',
 'audio.spectral.bandwidth': 'NaN-ignoring temporal mean of librosa spectral bandwidth in Hz per '
                             "waveform; sr=22050 and librosa's default bandwidth settings.",
 'audio.spectral.centroid': 'Mean over frames of librosa spectral centroid in Hz per waveform; '
                            'sr=22050 controls sample rate. One scalar per waveform.',
 'audio.spectral.centroid_normalized': 'Mean librosa spectral centroid divided by Nyquist '
                                       'frequency (sr/2), one scalar per waveform; sr=22050.',
 'audio.spectral.contrast': 'Temporal mean of librosa spectral contrast in each subband from STFT '
                            'magnitudes; sr=22050, n_bands=6 yields seven values per waveform.',
 'audio.spectral.flatness': 'NaN-ignoring temporal mean of librosa spectral flatness per mono '
                            "waveform, using librosa's default spectral settings.",
 'audio.spectral.flux': 'Mean Euclidean change between adjacent L2-normalized STFT magnitude '
                        'frames per waveform; normalization adds 1e-12 and a single frame yields '
                        'zero.',
 'audio.spectral.rolloff': 'Mean over frames of librosa spectral rolloff at its default 85% energy '
                           'fraction, in Hz per waveform; sr=22050.',
 'audio.tonal.chroma': 'Temporal mean of librosa STFT chroma, 12 pitch-class values per mono '
                       'waveform; sr=22050.',
 'audio.tonal.harmonicity': 'Largest positive-lag autocorrelation in a length-dependent lag window '
                            'divided by zero-lag energy after mean centering; short or zero-energy '
                            'waveforms return zero, no sample-rate parameter.',
 'audio.tonal.key': 'Index and strength of the largest time-averaged STFT chroma component; 12 '
                    'pitch classes at sr=22050, not a major/minor musical-key estimator.',
 'audio.tonal.pitch': 'Median librosa YIN frequency in Hz per waveform; sr=22050, fmin=50, '
                      'fmax=2000 capped below Nyquist; computation exceptions yield NaN.',
 'audio.tonal.tonnetz': 'Temporal mean of six librosa tonnetz coordinates after harmonic '
                        'separation; sr=22050. Exceptions during separation or tonnetz computation '
                        'return six NaNs.',
 'audio.voice.f0': 'Alias of median librosa YIN frequency in Hz; forwards sr=22050, fmin=50 and '
                   'fmax=2000 with a Nyquist cap, returning NaN on pitch-computation exceptions.',
 'audio.voice.formants': 'Sorted positive-angle LPC-root frequencies above 90 Hz and below '
                         'Nyquist, padded to n_formants=4 with NaNs; sr=22050, LPC order capped at '
                         '16, failures yield NaNs.',
 'audio.voice.hnr': "10*log10(h/(1-h)) from FEATM's autocorrelation harmonicity proxy, each ratio "
                    'term floored at 1e-12; decibel proxy rather than a calibrated voice-noise '
                    'measurement.',
 'audio.voice.jitter': 'Mean absolute difference between successive inverse-YIN-frequency frame '
                       'values divided by their mean; sr=22050 and YIN limits 50 to '
                       'min(1000,sr/2-1) Hz, not cycle-segmented jitter.',
 'audio.voice.shimmer': 'Mean absolute change in frame RMS divided by mean RMS; frame=512, hop=256 '
                        'samples. Insufficient frames or nonpositive mean RMS give NaN; not '
                        'cycle-segmented shimmer.',
 'audio.waveform.crest_factor': 'Peak absolute waveform amplitude divided by RMS per row; where '
                                'RMS is not positive the output is zero. One scalar per waveform.',
 'audio.waveform.dynamic_range': '20*log10((95th percentile absolute amplitude+1e-12)/(10th '
                                 'percentile+1e-12)) per mono waveform; an amplitude-relative '
                                 'percentile range in decibels.',
 'audio.waveform.rms': 'Square root of mean squared waveform samples per row (1D input is one '
                       'waveform); one scalar. NaNs propagate.',
 'audio.waveform.zero_crossing_rate': 'Fraction of adjacent waveform sign-bit changes per row; '
                                      'zero is nonnegative, NaNs are not filtered. One scalar per '
                                      'waveform.',
 'event.point_process.temporal_point_process': 'Four per-sequence timing summaries: event rate, '
                                               'gap burstiness, mean interarrival gap and '
                                               'population gap standard deviation; this does not '
                                               'fit a point-process model.',
 'event.process.hawkes': 'SciPy L-BFGS-B fit of positive mu, alpha and beta for an exponential '
                         'univariate Hawkes intensity mu+alpha*sum(exp(-beta*lag)); three outputs, '
                         'NaNs for fewer than three finite events or unsuccessful optimization.',
 'event.sessions.session': 'Session count, mean and maximum session size after splitting gaps '
                           'exceeding gap, default five times the median finite-event gap; an '
                           'empty sequence yields one zero-size session.',
 'event.survival.survival': 'Per-cohort final Kaplan-Meier survival, first time survival reaches '
                            '0.5 and integrated survival through the last duration; accepts a '
                            'durations/observed dict, observed defaults true, missing median is '
                            'NaN.',
 'event.timing.burstiness': '(gap standard deviation-gap mean)/(their sum+1e-15) using finite '
                            'event times; no gaps give zero. List inputs are sorted, bare 1D '
                            'ndarrays retain order.',
 'event.timing.event_rate': '(event count-1)/(last-first time) per event sequence; returns zero '
                            'with fewer than two events or nonpositive span. Lists are sorted; a '
                            'bare 1D ndarray is not.',
 'event.timing.interarrival_time': 'Mean, population standard deviation, minimum and maximum of '
                                   'gaps between finite event times per sequence; no gaps yield '
                                   'four NaNs. Lists are sorted, but a bare 1D ndarray retains its '
                                   'order.',
 'event.timing.recency': "Reference time minus each sequence's last event; empty sequences give "
                         'NaN. reference_time defaults to the largest last-event time in the call '
                         'and must be explicit for multibatch execution.',
 'geometry.alpha_shapes.alpha_shape': '2D per-point boundary flag and degree in retained Delaunay '
                                      'triangles; circumradius threshold is median radius by '
                                      'default or 1/max(alpha,1e-12), no polygon is returned.',
 'geometry.angles.local_angles': 'Mean and population standard deviation of angles in radians '
                                 'between pairs of vectors to k=6 nearest neighbors per point; no '
                                 'vector pairs produce two NaNs.',
 'geometry.distances.euclidean': "Euclidean distance of each point from the whole point set's "
                                 'centroid, in coordinate units; no reference-point parameter.',
 'geometry.distances.mahalanobis': "Distance from the whole point set's centroid using the "
                                   'pseudoinverse of its sample covariance matrix; one scalar per '
                                   'point.',
 'geometry.hull.convex_hull': 'Convex-hull vertex flag and inward distance to the nearest hull '
                              'facet plane per input point; two outputs, with no hull mesh '
                              'returned.',
 'geometry.hull.hull_area': '2D convex-hull enclosed area or, in higher dimensions, SciPy '
                            'ConvexHull surface area, repeated for every input point.',
 'geometry.hull.hull_distance': 'Negative largest signed hull-facet plane equation at each input '
                                'point; inward distance to the nearest supporting plane, zero at '
                                'hull vertices up to numerical error.',
 'geometry.hull.hull_volume': 'SciPy convex-hull volume repeated per input point; in 2D this is '
                              'enclosed area rather than a 3D volume.',
 'geometry.hull.solidity': 'Convex-hull volume divided by axis-aligned bounding-box hypervolume '
                           'plus 1e-15, repeated per point; not an object-mask solidity measure.',
 'geometry.projections.projection': 'Coordinates along leading right singular vectors of the '
                                    'mean-centered whole point set; n_components=3 capped by '
                                    'available vectors, no variance normalization.',
 'geometry.proximity.gabriel_graph': 'Per-point degree, mean and maximum incident edge length in '
                                     'the Gabriel graph; another point strictly inside the '
                                     'diameter ball excludes an edge, with 1e-12 tolerance.',
 'geometry.proximity.minimum_spanning_tree': 'Per-point degree, mean and maximum incident edge '
                                             "length of SciPy's Euclidean minimum spanning tree "
                                             'after symmetrization; absent incident edges yield '
                                             'NaN lengths.',
 'geometry.proximity.relative_neighborhood_graph': 'Per-point degree, mean and maximum incident '
                                                   'edge length in the relative-neighborhood '
                                                   'graph; excludes a pair when another point is '
                                                   'closer to both endpoints, with 1e-12 '
                                                   'tolerance.',
 'geometry.stochastic_geometry.point_process': 'Nearest-neighbor distance, k-neighbor radius and '
                                               'min(k,n-1)/radius^dimension per point; k=6 and '
                                               'radius floor 1e-12, no unit-ball volume factor.',
 'geometry.triangulation.aspect_ratio': 'NaN-ignoring mean of longest/(shortest+1e-15) edge length '
                                        'across incident Delaunay simplices per point.',
 'geometry.triangulation.circumradius': 'NaN-ignoring mean circumradius of incident Delaunay '
                                        'simplices per point, obtained by a linear circumcenter '
                                        'solve; failed solves contribute NaN.',
 'geometry.triangulation.delaunay_degree': 'Number of distinct Delaunay edge neighbors per input '
                                           'point; one value per coordinate row, triangulating the '
                                           'whole point set with SciPy.',
 'geometry.triangulation.edge_length': 'Alias of mean incident Delaunay edge length per point; '
                                       'returns a local mean, not a vector of individual edge '
                                       'lengths.',
 'geometry.triangulation.inradius': 'NaN-ignoring mean of d*simplex_volume/summed_face_volume over '
                                    'incident Delaunay simplices per point; nonpositive face sums '
                                    'give NaN.',
 'geometry.triangulation.max_edge_length': 'Maximum Euclidean distance to Delaunay edge neighbors '
                                           'per point; no neighbors yield NaN.',
 'geometry.triangulation.mean_edge_length': 'Mean Euclidean distance to Delaunay edge neighbors '
                                            'per point; no incident neighbors yield NaN.',
 'geometry.triangulation.mean_simplex_volume': 'NaN-ignoring mean of incident Delaunay d-simplex '
                                               'volumes per point, using abs(det)/d!; same '
                                               'computation as simplex_volume.',
 'geometry.triangulation.min_edge_length': 'Minimum Euclidean distance to Delaunay edge neighbors '
                                           'per point; no neighbors yield NaN.',
 'geometry.triangulation.simplex_quality': 'NaN-ignoring mean of '
                                           'd*volume^(1/d)/(longest_edge+1e-15) across incident '
                                           'Delaunay simplices per point; this formula is not '
                                           'normalized to unit equilateral quality.',
 'geometry.triangulation.simplex_volume': 'NaN-ignoring mean of incident Delaunay d-simplex '
                                          'volumes abs(det)/d! per point; returns an aggregate, '
                                          'not one value per simplex.',
 'geometry.triangulation.std_edge_length': 'Population standard deviation of Euclidean Delaunay '
                                           'neighbor distances per point; no neighbors yield NaN.',
 'geometry.triangulation.triangle_angles': 'Mean, population std, minimum and maximum in radians '
                                           'of all three angles of every incident Delaunay '
                                           'triangle per point; requires 2D coordinates.',
 'geometry.triangulation.triangle_area': 'Mean area of incident Delaunay triangles per point; '
                                         'requires 2D coordinates and returns NaN when a point has '
                                         'no incident simplex.',
 'geometry.voronoi.boundary_length': 'Polygon perimeter of each finite 2D Voronoi cell; empty or '
                                     'unbounded regions return NaN, non-2D coordinates are '
                                     'rejected.',
 'geometry.voronoi.cell_area': 'Shoelace area of each finite 2D Voronoi cell; empty or unbounded '
                               'regions return NaN, non-2D coordinates are rejected.',
 'geometry.voronoi.cell_volume': "ConvexHull volume of each finite Voronoi region's vertices; "
                                 'unbounded or failed regions return NaN, and in 2D the value is '
                                 'area.',
 'geometry.voronoi.neighbor_count': 'Count of distinct sites sharing a SciPy Voronoi ridge with '
                                    'each input site, including sites with unbounded cells.',
 'graph.assortativity.degree': 'NetworkX degree assortativity coefficient for each graph in its '
                               'original directionality; no edges return NaN.',
 'graph.centrality.betweenness': 'Mean, population std, maximum and minimum of NetworkX default '
                                 'betweenness centrality across nodes; four outputs per graph, no '
                                 'sampling parameter is exposed.',
 'graph.centrality.closeness': 'Mean, population std, maximum and minimum of NetworkX default '
                               'closeness centrality across nodes; four outputs per graph.',
 'graph.centrality.current_flow': 'Mean, population std, maximum and minimum of NetworkX '
                                  'current-flow betweenness after undirected conversion; nonempty '
                                  'disconnected graphs, graphs with fewer than three nodes and '
                                  'solver failures yield NaNs.',
 'graph.centrality.degree': 'Mean, population std, maximum and minimum of NetworkX degree '
                            'centrality across nodes; four outputs per graph, empty distributions '
                            'give zeros.',
 'graph.centrality.eigenvector': 'Mean, population std, maximum and minimum of NetworkX NumPy '
                                 'eigenvector centrality across nodes; graphs with at most one '
                                 'node use zero node scores.',
 'graph.centrality.harmonic': 'Mean, population std, maximum and minimum of NetworkX default '
                              'harmonic centrality across nodes; four outputs per graph.',
 'graph.centrality.katz': 'Mean, population std, maximum and minimum of NetworkX NumPy Katz '
                          'centrality with fixed alpha=0.01, beta=1; at most one node gives zero '
                          'node scores.',
 'graph.centrality.pagerank': 'Mean, population std, maximum and minimum of NetworkX default '
                              'PageRank values across nodes; four outputs per graph, backend '
                              'failures propagate.',
 'graph.cliques.cliques': 'Maximal-clique count, largest clique size and mean maximal-clique size '
                          'of the undirected graph; empty clique collections give zeros.',
 'graph.clustering.average_clustering': 'NetworkX mean node clustering coefficient after '
                                        'undirected conversion, one scalar per graph; empty graphs '
                                        'give zero.',
 'graph.clustering.coefficient': 'Mean, population std, maximum and minimum of node clustering '
                                 'coefficients after undirected conversion; four outputs per '
                                 'graph.',
 'graph.clustering.transitivity': 'NetworkX transitivity of the undirected graph: global '
                                  'closed-triplet proportion, one scalar per graph.',
 'graph.clustering.triangles': 'Mean, population std, maximum and minimum of incident triangle '
                               'counts per node after undirected conversion; not the total number '
                               'of distinct triangles.',
 'graph.community.leiden': 'Community count, mean/max size and partition quality from Leiden '
                           'ModularityVertexPartition on an igraph conversion; directions are '
                           'retained but edge attributes are not copied, no seed is set.',
 'graph.community.louvain': 'Count, mean size and maximum size of NetworkX Louvain communities '
                            'after undirected conversion with seed=0; empty graphs give zeros.',
 'graph.community.modularity': 'NetworkX modularity of the seed=0 Louvain partition on the '
                               'undirected graph; empty graphs or graphs without edges return '
                               'zero.',
 'graph.connectivity.articulation': 'Number of articulation vertices in the undirected conversion '
                                    'of each graph.',
 'graph.connectivity.bridges': 'Number of bridge edges whose removal disconnects the undirected '
                               'conversion of each graph.',
 'graph.connectivity.components': 'Number of connected components for undirected graphs or weakly '
                                  'connected components for directed graphs.',
 'graph.connectivity.density': 'NetworkX graph density per input graph, using directed or '
                               'undirected normalization according to graph type.',
 'graph.cores.k_core': 'Mean, population std, maximum and minimum of undirected node core numbers; '
                       'graphs without edges use zero node scores, no subgraph is returned.',
 'graph.cores.k_shell': 'Alias of the four undirected node-core-number distribution summaries; '
                        'this feature returns scores rather than a k-shell subgraph.',
 'graph.cores.k_truss': 'Largest tested nonempty k-truss and its edge count, scanning k=3 through '
                        'max_k=8 on the undirected graph; no passing level returns (2,0).',
 'graph.cores.onion_layers': 'Mean, population std, maximum and minimum of NetworkX onion-layer '
                             'node indices after undirected conversion; exceptions yield four '
                             'NaNs.',
 'graph.cycles.cycle_count': "Number of cycles in NetworkX's undirected cycle basis, rather than a "
                             'count of every simple cycle.',
 'graph.cycles.girth': 'NetworkX girth of the undirected graph, one scalar per graph; backend '
                       'exceptions return NaN and acyclic graphs may return infinity.',
 'graph.degree.average_degree': 'Mean NetworkX node degree per graph; directed graphs use total '
                                'degree and empty graphs return zero.',
 'graph.embeddings.deepwalk': 'Mean gensim skip-gram node vector per graph from content-seeded '
                              'random walks; dimensions=32, walk_length=20, num_walks=20, Word2Vec '
                              'window=5/workers=1/seed=0.',
 'graph.embeddings.gnn': 'Flattened fixed-width embedding per graph from required user callable '
                         'model(graph); FEATM supplies no GNN architecture, graph training or '
                         'pretrained weights.',
 'graph.embeddings.node2vec': 'Mean node embedding per graph from Node2Vec with dimensions=32, '
                              'walk_length=20, num_walks=20, workers=1 and seed=0, fitted with '
                              'window=5; requires the optional node2vec backend.',
 'graph.graphlets.graphlets': 'Undirected edge, open-wedge, triangle, four-clique and '
                              'chord-agnostic four-cycle counts; five raw counts, without '
                              'graph-size normalization.',
 'graph.homophily.homophily': 'Fraction of edges whose endpoint node labels match, considering '
                              "only edges with label_attr='label' present at both ends; no "
                              'eligible edges give NaN.',
 'graph.link_prediction.common_neighbors': 'Mean, population std and maximum common-neighbor count '
                                           'over the first max_pairs=5000 undirected nonedges in '
                                           'NetworkX iteration order; no nonedges yield zeros.',
 'graph.mixing.mixing': "Diagonal mass, Shannon entropy in nats and category count of NetworkX's "
                        "normalized attribute-mixing matrix; attribute='label', exceptions give "
                        'three NaNs.',
 'graph.motifs.motifs': 'NetworkX 16-category triadic census for directed graphs; undirected '
                        'graphs instead place induced empty/one-edge/wedge/triangle counts in the '
                        'first four slots and pad twelve zeros.',
 'graph.paths.diameter': 'NetworkX unweighted diameter of the largest connected component after '
                         'undirected conversion; at most one node returns zero and other '
                         'components are excluded.',
 'graph.paths.eccentricity': 'Mean, population std, maximum and minimum of unweighted node '
                             'eccentricities in the largest undirected connected component; empty '
                             'graphs give zeros.',
 'graph.paths.efficiency': 'NetworkX global efficiency of the largest connected component after '
                           'undirected conversion; other components are excluded and at most one '
                           'node gives zero.',
 'graph.paths.radius': 'NetworkX unweighted radius of the largest connected component after '
                       'undirected conversion; at most one node returns zero and other components '
                       'are excluded.',
 'graph.paths.shortest_paths': 'Mean, population std and maximum of positive all-pairs unweighted '
                               'path lengths within the largest undirected connected component; no '
                               'positive paths gives zeros.',
 'graph.reciprocity.reciprocity': 'NetworkX reciprocity for directed graphs with edges; undirected '
                                  'nonempty-edge graphs return one, and graphs without edges '
                                  'return zero.',
 'graph.size.n_edges': 'Number of edges in each NetworkX input graph, one scalar per graph using '
                       "that graph type's edge-count convention.",
 'graph.size.n_nodes': 'Number of nodes in each NetworkX input graph, one scalar per graph.',
 'graph.spectral.algebraic_connectivity': 'Second-smallest undirected combinatorial-Laplacian '
                                          'eigenvalue, or zero for fewer than two eigenvalues; '
                                          'includes all components.',
 'graph.spectral.eigenvalues': 'Smallest k=8 eigenvalues of the dense combinatorial Laplacian '
                               'after undirected conversion, padded with NaN for smaller graphs; '
                               'not normalized Laplacian eigenvalues.',
 'graph.spectral.laplacian': 'Alias of the smallest k=8 undirected combinatorial-Laplacian '
                             'eigenvalues, padded with NaNs; returns a spectrum rather than the '
                             'Laplacian matrix.',
 'graph.spectral.spectral_gap': 'Difference between the two largest absolute eigenvalues of the '
                                'undirected adjacency matrix; fewer than two eigenvalues give '
                                'zero, not a Laplacian gap.',
 'graph.topology.clique_complex': 'Counts of undirected cliques by simplex dimension zero through '
                                  'max_dim=3; returns counts rather than a complex object or '
                                  'homology groups.',
 'graph.topology.persistent_homology': 'Undirected component count, graph cycle rank E-V+C and '
                                       'distinct triangle count; three static graph summaries, '
                                       'with no filtration or persistence intervals.',
 'image.color.blue_mean': 'NaN-ignoring spatial mean of channel 2 per image, capped to the last '
                          'available channel; grayscale input uses its intensity mean.',
 'image.color.colorfulness': 'sqrt(var(R-G)+var((R+G)/2-B))+0.3*sqrt(mean(R-G)^2+mean((R+G)/2-B)^2); '
                             'one value in input intensity units, grayscale or fewer than three '
                             'channels returns zero.',
 'image.color.green_mean': 'NaN-ignoring spatial mean of channel 1 per image, capped to the last '
                           'available channel; grayscale input uses its intensity mean.',
 'image.color.hsv_histogram': 'OpenCV uint8 RGB-to-HSV followed by three normalized histograms, '
                              'bins=8 each over [0,256], including hue; low-range input max<=1.5 '
                              'is scaled by 255 first.',
 'image.color.lab_histogram': 'OpenCV uint8 RGB-to-Lab followed by three normalized histograms, '
                              'bins=8 each over [0,256]; low-range input max<=1.5 is scaled by 255 '
                              'first, not physical Lab units.',
 'image.color.red_mean': 'NaN-ignoring spatial mean of channel 0 per image; grayscale input uses '
                         'its intensity mean, channels are not rescaled.',
 'image.color.rgb_histogram': 'Concatenate three independently normalized RGB histograms, bins=8 '
                              'per channel over [0,256]; input max<=1.5 is multiplied by 255, '
                              'clipped to uint8, grayscale replicated to three channels.',
 'image.edges.edge_density': 'Fraction of pixels marked by scikit-image Canny with defaults after '
                             'per-image grayscale range conversion to uint8 then division by 255; '
                             'one scalar per image.',
 'image.embeddings.clip': 'Hugging Face CLIP image-tower vector, default '
                          'openai/clip-vit-base-patch32; converts to uint8 PIL RGB, uses '
                          'CLIPProcessor preprocessing and available model weights, no explicit '
                          'output L2 normalization.',
 'image.embeddings.convnext': 'timm embedding per image, default convnext_tiny with '
                              'pretrained=True and average pooling; checkpoint-size bilinear '
                              'resize and conditional /255 scaling, no channel mean/std '
                              'normalization.',
 'image.embeddings.dino': 'timm embedding per image, default vit_small_patch14_dinov2.lvd142m with '
                          'pretrained=True; checkpoint-size resize, average pooling with token '
                          'retry for specific load errors, no channel mean/std normalization.',
 'image.embeddings.efficientnet': 'timm embedding per image, default efficientnet_b0 with '
                                  'pretrained=True and average pooling; resizes to checkpoint '
                                  'size, scales values above 1.5 by 255, no channel mean/std '
                                  'normalization.',
 'image.embeddings.resnet': 'timm embedding per image, default resnet50 with pretrained=True and '
                            'average pooling; resizes to checkpoint input size and scales values '
                            'above 1.5 by 255, without channel mean/std normalization.',
 'image.embeddings.timm': 'timm embedding per image with configurable model_name, default '
                          'resnet18/pretrained=True; checkpoint-size resize, average pooling with '
                          'token retry for specific load errors, conditional /255 without channel '
                          'normalization.',
 'image.embeddings.vit': 'timm embedding per image, default vit_small_patch16_224 with '
                         'pretrained=True; average pooling with token-pooling retry for specific '
                         'checkpoint-load errors, resize/conditional /255 only, no channel '
                         'mean/std normalization.',
 'image.frequency.dct': 'Flattened top-left size=4 square of orthonormal SciPy DCT coefficients '
                        'after grayscale nan_to_num; 16 outputs when the image is at least 4x4, no '
                        'centering.',
 'image.frequency.fft2': 'Mean centered-image FFT2 power in bins=12 radial pixel-frequency bands '
                         'after fftshift; nan_to_num after centering, unnormalized power without '
                         'physical spatial-frequency units.',
 'image.frequency.wavelet': 'Mean squared LH/HL/HH Haar detail coefficients for levels=3 per '
                            'grayscale image; odd tails dropped, nan_to_num, unavailable levels '
                            'give three NaNs, approximation omitted.',
 'image.gradient.hog': 'Spatial mean of scikit-image HOG cell histograms after grayscale uint8 '
                       'conversion; orientations=9, 8x8 pixels per cell, 1x1 cells per block, '
                       'fixed orientation vector rather than full spatial HOG.',
 'image.gradient.structure_tensor': 'Mean two structure-tensor eigenvalues and mean/std of '
                                    '(lambda1-lambda2)/(lambda1+lambda2+1e-12); grayscale '
                                    'uint8/255 input, sigma=1, four values per image.',
 'image.intensity.entropy': 'Shannon entropy in nats of a bins=64 histogram of finite grayscale '
                            'intensities; histogram range is selected independently per image.',
 'image.intensity.mean': 'NaN-ignoring mean over all pixels and channels per image; input (N,H,W) '
                         'or (N,H,W,C), one scalar per image. A bare RGB image needs a batch axis.',
 'image.intensity.std': 'NaN-ignoring population intensity standard deviation (ddof=0) over all '
                        'pixels/channels per image; input (N,H,W) or (N,H,W,C), one scalar per '
                        'image.',
 'image.keypoints.fast': 'OpenCV FAST detected-keypoint count, mean response and mean size per '
                         'grayscale uint8 image; detector defaults, no keypoint coordinates are '
                         'returned.',
 'image.keypoints.orb': 'OpenCV ORB detected-keypoint count, mean response and mean size per '
                        'grayscale uint8 image; detector defaults, no binary descriptors are '
                        'returned.',
 'image.keypoints.sift': 'OpenCV SIFT detected-keypoint count, mean response and mean size per '
                         'grayscale uint8 image; detector defaults, no keypoint descriptors are '
                         'returned.',
 'image.morphology.components': 'Connected-component count and mean/max pixel area of all '
                                'components above mean grayscale uint8 intensity; no foreground '
                                'gives three zeros.',
 'image.quality.blur': 'Population variance of the OpenCV float64 Laplacian of the grayscale uint8 '
                       'image; a sharpness proxy, with larger response variance rather than a '
                       'calibrated blur amount.',
 'image.quality.noise': 'Population std of grayscale intensity minus its sigma=1 Gaussian-smoothed '
                        'version; one residual-noise proxy per image in original intensity units.',
 'image.saliency.statistics': 'Mean, population std, maximum and 95th percentile of a '
                              'spectral-residual saliency map: log-FFT magnitude minus sigma=3 '
                              'Gaussian average, inverse transform squared magnitude, then sigma=2 '
                              'smoothing.',
 'image.shape.area': 'Pixel area of the largest connected component above the grayscale uint8 '
                     "image's mean intensity; no foreground component returns zero.",
 'image.shape.circularity': '4*pi*area/(perimeter^2+1e-12) for the largest above-mean grayscale '
                            'uint8 component; no foreground returns zero, raster perimeter may '
                            'give scores above one.',
 'image.shape.eccentricity': 'scikit-image ellipse eccentricity of the largest connected component '
                             'above mean grayscale uint8 intensity; no component returns zero.',
 'image.shape.fourier_descriptors': 'First n=10 non-DC Fourier magnitudes of the largest external '
                                    'threshold-mask contour, divided by its first magnitude+1e-12 '
                                    'and zero-padded; absent contours yield zeros.',
 'image.shape.hu_moments': 'Seven scikit-image Hu moments from normalized central moments of the '
                           'entire above-mean grayscale uint8 mask, not just its largest connected '
                           'component; no log transform.',
 'image.shape.perimeter': 'scikit-image perimeter of the largest connected component above mean '
                          'grayscale uint8 intensity; no foreground component returns zero.',
 'image.shape.solidity': 'scikit-image area/convex-area solidity of the largest connected '
                         'component above mean grayscale uint8 intensity; no component returns '
                         'zero.',
 'image.shape.zernike': 'Magnitudes of nine default low-order Zernike sums on a normalized '
                        'coordinate disk after uint8 grayscale scaling; (n+1)/pi weighting without '
                        'pixel-area normalization, configurable orders.',
 'image.symmetry.horizontal': '1 minus mean absolute top/bottom mirrored-half difference divided '
                              'by image population std+1e-12; grayscale, middle row omitted for '
                              'odd height, score is not clipped to [0,1].',
 'image.symmetry.vertical': '1 minus mean absolute left/right mirrored-half difference divided by '
                            'image population std+1e-12; grayscale, middle column omitted for odd '
                            'width, score is not clipped to [0,1].',
 'image.texture.fractal': 'Negative log-log slope of occupied-box count against power-of-two box '
                          'size for a mean-thresholded uint8 image; cropped full boxes only, fewer '
                          'than two positive scales gives NaN.',
 'image.texture.gabor': 'Mean scikit-image Gabor response magnitudes at frequencies 0.1/0.2/0.3 '
                        'and angles 0, pi/4, pi/2 and 3*pi/4; twelve values after grayscale '
                        'uint8/255 conversion.',
 'image.texture.glcm': 'Six averaged GLCM properties: contrast, dissimilarity, homogeneity, ASM, '
                       'energy and correlation; 16 grayscale levels, distance 1, four angles, '
                       'symmetric normalized co-occurrence matrices.',
 'image.texture.haralick': 'Six averaged GLCM properties plus entropy in nats and maximum '
                           'probability of the angle-averaged matrix; 16 levels, distance 1 and '
                           'four angles, eight outputs rather than all classical Haralick '
                           'statistics.',
 'image.texture.laws': 'Mean absolute responses of six fixed 5x5 Laws outer-product filters '
                       '(L5E5,E5E5,S5S5,R5R5,E5S5,L5S5); grayscale uint8, symmetric boundary '
                       'convolution, no local-mean subtraction.',
 'image.texture.lbp': 'Normalized uniform local-binary-pattern histogram after grayscale uint8 '
                      'conversion; P=8, R=1, bins=10 over [0,P+2], one fixed vector per image.',
 'image.topology.euler_number': 'scikit-image Euler number of the above-mean grayscale uint8 mask '
                                'with 8-connectivity; one components-minus-holes value per image.',
 'image.topology.holes': '8-connected component count minus Euler number for the above-mean '
                         'grayscale uint8 mask; one inferred hole count per image.',
 'image.topology.persistent_homology': 'Component and hole counts at thresholds=8 evenly spaced '
                                       'grayscale uint8 thresholds using >= and 8-connectivity; 16 '
                                       'default outputs, threshold summaries without persistence '
                                       'pairing or intervals.',
 'information.dependence.conditional_mi': 'Weighted within-Z-bin empirical mutual information in '
                                          'nats; bins=8, X/Y/Z are quantile-binned then X/Y '
                                          'rebinned inside each Z group, groups smaller than two '
                                          'are skipped.',
 'information.dependence.interaction_information': "Binned I(X;Y) minus the helper's within-Z "
                                                   'conditional-MI estimate, in nats; bins=8, '
                                                   'three supplied series per sample.',
 'information.dependence.mutual_information': 'Empirical mutual information in nats after separate '
                                              'quantile discretization of paired series; bins=16, '
                                              'unequal lengths truncate to their common length, '
                                              'empty pairs return zero.',
 'information.dependence.transfer_entropy': 'One-step X-to-Y conditional-MI estimate from Y[1:], '
                                            'X[:-1] and Y[:-1], using the quantile/rebinning '
                                            'helper with bins=8; one value per supplied series '
                                            'pair.',
 'information.divergence.jensen_shannon': 'Half KL(P||M) plus half KL(Q||M), M=(P+Q)/2, per '
                                          'normalized distribution pair in nats; returns '
                                          'divergence, not its square-root distance.',
 'information.divergence.kl': 'Sum p*log(p/q) per supplied distribution pair after clipping '
                              'negative counts and row normalization; log arguments use a 1e-15 '
                              'floor, natural logarithms.',
 'information.divergence.variation_of_information': 'H(X)+H(Y)-2*MI for paired series after '
                                                    'quantile binning; bins=16, MI helper rebins '
                                                    'the discrete labels, one scalar in nats per '
                                                    'pair.',
 'information.entropy.conditional': 'Joint quantile-binned entropy H(X,Y) minus binned H(Y), in '
                                    'nats per series pair; bins=16, no estimator bias correction.',
 'information.entropy.joint': 'Empirical joint entropy in nats of paired series after separate '
                              'quantile discretization with bins=16; paired symbols are counted '
                              'jointly, NaNs are replaced by finite medians during binning.',
 'information.entropy.renyi': 'Renyi entropy log(sum(p**alpha))/(1-alpha) per normalized '
                              'nonnegative row; alpha=2, alpha=1 delegates to Shannon, power sum '
                              'floored at 1e-15.',
 'information.entropy.shannon': 'Shannon entropy in nats per probability/count row after clipping '
                                'negatives to zero and sum normalization; all-zero rows yield '
                                'zero.',
 'information.entropy.tsallis': 'Tsallis entropy (1-sum(p**q))/(q-1) per normalized nonnegative '
                                'row; q=2, q=1 delegates to Shannon entropy in nats.',
 'information.geometry.information_geometry': 'Fisher-Rao angle 2*acos(sum(sqrt(p*u))), Hellinger '
                                              'distance to uniform u and Shannon entropy in nats '
                                              'per clipped, normalized probability row; three '
                                              'outputs.',
 'information.multivariate.dual_total_correlation': 'Sum of leave-one-variable-out joint entropies '
                                                    'minus (variable count-1)*full joint entropy, '
                                                    'in nats per dataset; bins=8 quantile '
                                                    'discretization.',
 'information.multivariate.total_correlation': 'Sum of marginal quantile-binned entropies minus '
                                               'joint entropy in nats per observation-by-variable '
                                               'dataset; bins=8, a plain 2D array is one dataset.',
 'information.pid.redundancy': "Target-weighted minimum of two predictors' specific-information "
                               'estimates, in nats; bins=8 quantile discretization, one scalar per '
                               'predictor1/predictor2/target triple.',
 'information.pid.synergy': 'Joint-coded-predictor MI minus redundancy and both unique components; '
                            'bins=8, joint codes are rebinned by the MI helper, so this is the '
                            'implemented discrete PID-style estimate.',
 'information.pid.unique_information': "First predictor's binned mutual information with target "
                                       'minus the two-predictor minimum-specific-information '
                                       "redundancy; bins=8, reports only predictor1's unique "
                                       'component.',
 'manifold.curvature.curvature': 'Smallest local covariance eigenvalue divided by the sum of the '
                                 'first three descending eigenvalues, with a 1e-15 denominator '
                                 'floor; k=12 neighbors, one value per point.',
 'manifold.dimension.intrinsic_dimension': 'Median of the local neighbor-distance dimension '
                                           'estimates, repeated for every point; forwards k=10 to '
                                           'the local estimator.',
 'manifold.dimension.local_dimension': 'Per-point neighbor-distance dimension estimate '
                                       '(m-1)/sum(log(r_max/r_j)), excluding the farthest distance '
                                       'from the sum; k=10, distances floored at 1e-12.',
 'manifold.distance.geodesic': 'Undirected shortest-path distance from the point nearest the cloud '
                               'centroid on a symmetrized k=8 neighbor-distance graph; '
                               'disconnected points can have infinite distance.',
 'manifold.embedding.diffusion_maps': 'Eigenvectors of the row-normalized RBF affinity matrix '
                                      'multiplied by their eigenvalues, taking n_components=2 '
                                      "after the leading eigenpair; gamma=None uses sklearn's "
                                      'kernel default and imaginary parts are discarded.',
 'manifold.embedding.isomap': 'scikit-learn Isomap coordinates fitted to the whole cloud; '
                              'n_components=2 and n_neighbors=8 are capped by input size, one '
                              'vector per point.',
 'manifold.embedding.laplacian_eigenmaps': 'scikit-learn SpectralEmbedding coordinates fitted to '
                                           'the whole cloud with random_state=0; n_components=2, '
                                           'n_neighbors=8, capped by sample count.',
 'manifold.residual.local_manifold_residual': 'Mean neighborhood reconstruction distance after '
                                              'projection onto components=2 local SVD directions; '
                                              'k=12, neighborhoods are centered on the query '
                                              'point.',
 'manifold.tangent.local_tangent_space': 'Leading right singular vector of each point-centered '
                                         'neighborhood, including the center; k=12, one '
                                         'ambient-dimensional vector per point with arbitrary '
                                         'sign.',
 'matrix.algebra.condition_number': 'NumPy 2-norm condition number per matrix; ill-conditioned or '
                                    'singular inputs can produce very large values or infinity.',
 'matrix.algebra.determinant': 'Determinant of each square input matrix; rectangular matrices '
                               'return NaN. A plain 2D array is one matrix.',
 'matrix.algebra.eigenvalues': 'Descending absolute eigenvalues for square matrices, or '
                               'eigenvalues of M.T@M for rectangular matrices; k=8 values, padded '
                               'with NaN. Signs and complex phases are discarded.',
 'matrix.algebra.frobenius_norm': 'Square root of the sum of squared matrix entries, one Frobenius '
                                  'norm per input matrix.',
 'matrix.algebra.norms': 'Four matrix norms per sample: Frobenius, induced one-norm, induced '
                         'infinity-norm and spectral 2-norm.',
 'matrix.algebra.rank': 'NumPy numerical matrix rank per input matrix, using its default '
                        'singular-value tolerance.',
 'matrix.algebra.singular_values': 'Leading k=8 singular values per matrix, in descending order; '
                                   'shorter spectra are padded with NaN.',
 'matrix.algebra.sparsity': 'Fraction of matrix entries close to zero under NumPy isclose '
                            'defaults, rather than the fraction exactly equal to zero.',
 'matrix.algebra.spectral_norm': 'Largest singular value of each input matrix, computed with '
                                 "NumPy's matrix 2-norm.",
 'matrix.algebra.trace': 'Sum of diagonal entries of each input matrix, including rectangular '
                         'matrices; a plain 2D array is one matrix.',
 'matrix.information.matrix_entropy': 'Shannon entropy in nats of the clipped nonnegative '
                                      'eigenvalues of M@M.T after sum normalization; an all-zero '
                                      'spectrum gives zero.',
 'matrix.spectral.spectral_statistics': 'Five singular-spectrum summaries per matrix: maximum, '
                                        'mean, population standard deviation, entropy of '
                                        'normalized squared singular values in nats, and count '
                                        'above 1e-12.',
 'multimodal.alignment.audio_text': 'Cosine similarity and Euclidean distance between supplied '
                                    'audio/text embeddings per row, truncated to common width; '
                                    'dict keys audio/text select the pair, otherwise pair-input '
                                    'conventions apply.',
 'multimodal.alignment.image_text': 'Cosine similarity and Euclidean distance between supplied '
                                    'image/text embeddings per row, truncated to common width; '
                                    'dict keys image/text select the pair, otherwise pair-input '
                                    'conventions apply.',
 'multimodal.alignment.video_text': 'Cosine similarity and Euclidean distance between supplied '
                                    'video/text embeddings per row, truncated to common width; '
                                    'dict keys video/text select the pair, otherwise pair-input '
                                    'conventions apply.',
 'multimodal.embedding_distances.cross_modal': 'Rowwise Euclidean, Manhattan and 1-minus-cosine '
                                               'distances between supplied embeddings, truncated '
                                               'to the smaller width; cosine denominator includes '
                                               '1e-15.',
 'multimodal.fusion.image_tabular': 'Concatenate supplied image and tabular embedding rows without '
                                    'scaling or fitting; dict keys image/tabular select inputs, '
                                    'otherwise pair-input conventions apply.',
 'multimodal.fusion.late_fusion': 'Weighted sum of two supplied embedding rows truncated to their '
                                  'common width; weights default to (0.5,0.5), with no weight '
                                  'normalization.',
 'multimodal.fusion.temporal_tabular': 'Concatenate supplied temporal and tabular embedding rows '
                                       'without scaling or fitting; dict keys temporal/tabular '
                                       'select inputs, otherwise pair-input conventions apply.',
 'multimodal.similarity.cross_modal': 'Rowwise cosine similarity between two supplied embedding '
                                      'matrices, truncated to their common width; denominator '
                                      'includes 1e-15. Does not compute embeddings.',
 'pointcloud.descriptors.fpfh': "Mean of Open3D's 33-dimensional FPFH vectors over a 3D cloud; "
                                'normals use radius or 5% of bounding-box diagonal, FPFH uses '
                                'twice that radius and max_nn=100.',
 'pointcloud.descriptors.shape_context': 'Normalized radial/angular histogram about the cloud '
                                         'centroid in its first two coordinates; radial_bins=5 '
                                         'logarithmic radial intervals, angular_bins=12, 60 '
                                         'outputs by default, not per-point shape contexts.',
 'pointcloud.descriptors.shot': 'Unavailable SHOT adapter: this registration always raises '
                                'ImportError directing users to a PCL/pclpy plugin pack; '
                                'installing pclpy alone does not execute a descriptor here.',
 'pointcloud.descriptors.spin_images': 'Unavailable spin-image adapter: this registration always '
                                       'raises ImportError directing users to a PCL/pclpy plugin '
                                       'pack; no spin-image computation is implemented here.',
 'pointcloud.embeddings.point_transformer': 'Flattened fixed-width vector per cloud from required '
                                            'user callable model(points); no point-transformer '
                                            'architecture, training or pretrained weights are '
                                            'supplied.',
 'pointcloud.embeddings.pointnet': 'Flattened fixed-width vector per cloud from required user '
                                   'callable model(points); no PointNet architecture, training or '
                                   'pretrained weights are supplied.',
 'pointcloud.geometry.bbox_volume': 'Product of coordinate-wise ranges in each cloud: axis-aligned '
                                    'bounding-box hypervolume, with zero for any zero-span axis.',
 'pointcloud.geometry.centroid_radius_mean': 'Mean Euclidean distance of points from their cloud '
                                             'centroid, one value per cloud in coordinate units.',
 'pointcloud.geometry.centroid_radius_std': 'Population standard deviation of distances from the '
                                            'cloud centroid, one value per cloud in coordinate '
                                            'units.',
 'pointcloud.geometry.curvature': 'Alias of the surface-variation summaries: mean, population std '
                                  'and 90th percentile of the third local covariance eigenvalue '
                                  'divided by the first-three sum; k=16.',
 'pointcloud.geometry.density': 'Mean, population std and 90th percentile of k/max(kth-neighbor '
                                'radius,1e-12)^dimension; k=8 stays in numerator even for smaller '
                                'clouds, no unit-ball volume factor.',
 'pointcloud.geometry.normals': 'Per-axis mean absolute component and signed population standard '
                                'deviation of smallest-eigenvalue local covariance eigenvectors; '
                                'k=16, 2*d values per cloud, normals are not globally oriented.',
 'pointcloud.geometry.roughness': 'Mean, population std and maximum of local mean absolute '
                                  "displacement projected on each point's covariance normal; k=16, "
                                  "includes the center's zero displacement.",
 'pointcloud.local_geometry.anisotropy': 'Mean, population std and 90th percentile of '
                                         '(lambda1-lambda3)/max(lambda1,1e-15) over k=16 local '
                                         'covariance neighborhoods; descending eigenvalues, one '
                                         'triple per cloud.',
 'pointcloud.local_geometry.eigenentropy': 'Mean, population std and 90th percentile of Shannon '
                                           'entropy in nats of the three leading local covariance '
                                           'eigenvalues after sum normalization; k=16, zero '
                                           'spectrum gives zero entropy.',
 'pointcloud.local_geometry.linearity': 'Mean, population std and 90th percentile of '
                                        '(lambda1-lambda2)/max(lambda1,1e-15) over local '
                                        'covariance neighborhoods; k=16, descending eigenvalues, '
                                        'alias of pca_linearity.',
 'pointcloud.local_geometry.omnivariance': 'Mean, population std and 90th percentile of the cube '
                                           'root of the product of the first three local '
                                           'covariance eigenvalues; k=16, eigenvalues clipped '
                                           'nonnegative, zero-padded below 3D.',
 'pointcloud.local_geometry.pca_linearity': 'Mean, population std and 90th percentile of '
                                            '(lambda1-lambda2)/max(lambda1,1e-15) over local '
                                            'covariance neighborhoods; k=16, descending '
                                            'eigenvalues, one triple per cloud.',
 'pointcloud.local_geometry.pca_planarity': 'Mean, population std and 90th percentile of '
                                            '(lambda2-lambda3)/max(lambda1,1e-15) over local '
                                            'covariance neighborhoods; k=16, descending '
                                            'eigenvalues, zero padding below three dimensions.',
 'pointcloud.local_geometry.pca_sphericity': 'Mean, population std and 90th percentile of '
                                             'lambda3/max(lambda1,1e-15) over local covariance '
                                             'neighborhoods; k=16, descending eigenvalues, zero '
                                             'padding below three dimensions.',
 'pointcloud.local_geometry.planarity': 'Mean, population std and 90th percentile of '
                                        '(lambda2-lambda3)/max(lambda1,1e-15) over local '
                                        'covariance neighborhoods; k=16, alias of pca_planarity.',
 'pointcloud.local_geometry.scattering': 'Mean, population std and 90th percentile of '
                                         'lambda3/max(lambda1,1e-15) over local covariance '
                                         'neighborhoods; k=16, same local ratio as pca_sphericity.',
 'pointcloud.size.n_points': 'Number of coordinate rows in each point cloud; a plain 2D array '
                             'represents one cloud.',
 'pointcloud.surface.surface_variation': 'Mean, population std and 90th percentile of '
                                         'lambda3/max(lambda1+lambda2+lambda3,1e-15) across k=16 '
                                         'local covariance neighborhoods; one triple per cloud.',
 'pointcloud.topology.persistence': 'Largest n=8 positive minimum-spanning-tree edge weights, '
                                    'ascending within the retained tail and left-padded with '
                                    'zeros; at most 400 evenly indexed points, an H0-only summary '
                                    'per cloud.',
 'pointcloud.triangulation.delaunay': 'Mean and population std of Delaunay degree and incident '
                                      'mean edge length across each cloud; four outputs delegated '
                                      'to the geometry helpers.',
 'pointcloud.voxel.voxel_density': 'Occupied-cell fraction and mean/max points per occupied cell '
                                   'after axiswise range scaling into a resolution=16 grid; '
                                   'denominator is resolution^dimension.',
 'sequence.alignment.alignment': 'Global dynamic-programming alignment score and score divided by '
                                 'the larger sequence length; reference defaults to the first '
                                 'sequence, match=1, mismatch=-1, gap=-1, with integer score '
                                 'storage.',
 'sequence.basic.length': 'Number of elements per sequence; strings count characters, while a bare '
                          '1D ndarray is treated as one sequence.',
 'sequence.basic.unique_ratio': 'Distinct token count divided by sequence length, zero for an '
                                'empty sequence; string tokens are characters.',
 'sequence.compression.compression': 'zlib-compressed byte length divided by original byte length '
                                     'per sequence; strings use UTF-8, arrays use raw array bytes, '
                                     'empty input returns zero and ratios may exceed one.',
 'sequence.distance.edit_distance': 'Unit-cost Levenshtein edit distance to reference, defaulting '
                                    'to the first sequence in the call; an explicit reference is '
                                    'required for multibatch extraction.',
 'sequence.entropy.shannon': 'Shannon entropy in nats of empirical token counts per sequence; '
                             'string tokens are characters, and empty input has entropy zero.',
 'sequence.entropy.shannon_entropy': 'Alias of the sequence token-count Shannon entropy in nats; '
                                     'strings are character sequences and empty sequences return '
                                     'zero.',
 'sequence.markov.markov': 'Alias of the normalized adjacent-token matrix, not a fitted '
                           'row-stochastic Markov model; buckets=8, per-sequence sorted '
                           'vocabulary, flattened buckets squared outputs.',
 'sequence.motifs.motifs': 'For each contiguous gram length 2 through max_len=4, return distinct '
                           'gram count and largest occurrence count; overlapping occurrences are '
                           'counted.',
 'sequence.ngrams.ngrams': 'Normalized stable-hash bucket counts for contiguous n=2 token grams '
                           'per sequence; buckets=32, no grams produce zeros and collisions '
                           'combine counts.',
 'sequence.run_length.run_count': 'Count of contiguous runs of equal tokens per sequence; empty '
                                  'sequences have zero runs.',
 'sequence.run_length.runs': 'Run count, mean run length, population standard deviation and '
                             'maximum run length per sequence; all four are zero for empty input.',
 'sequence.subsequences.subsequences': 'Count, distinct count and largest frequency of contiguous '
                                       'window=3 token slices per sequence; no slices produce '
                                       'three zeros.',
 'sequence.symbolic_dynamics.symbolic_dynamics': 'Three per-sequence values: empirical token '
                                                 'Shannon entropy in nats, adjacent-token change '
                                                 'fraction and zlib compressed/original byte '
                                                 'ratio.',
 'sequence.transitions.transition_matrix': 'Flattened buckets-by-buckets adjacent-token count '
                                           'matrix normalized by total retained transitions; '
                                           'buckets=8, vocabulary is the first tokens sorted by '
                                           'str independently per sequence, excess tokens are '
                                           'ignored.',
 'sequence.transitions.transition_rate': 'Fraction of adjacent tokens that differ, one value per '
                                         'sequence; sequences shorter than two tokens return zero.',
 'signal.cepstral.cepstrum': 'First n_coeffs=12 available inverse-real-FFT coefficients of floored '
                             'log centered-FFT magnitudes; DC zero, adaptive magnitude floor, '
                             'output truncated to available coefficients.',
 'signal.decomposition.eemd': 'Mean squared amplitude of up to n_imfs=6 PyEMD EEMD components, '
                              'zero-padded; 20 trials, sequential execution and per-signal '
                              'content-derived noise seed.',
 'signal.decomposition.emd': 'Mean squared amplitude of up to n_imfs=6 components returned by '
                             'PyEMD EMD per signal, zero-padded; returns component energies, not '
                             'reconstructed modes.',
 'signal.decomposition.ica': 'Whole-batch sklearn FastICA coordinates with random_state=0 and '
                             'unit-variance whitening; n_components=4 capped by batch shape, '
                             'nan_to_num input, not a per-row decomposition or frozen train/test '
                             'transform.',
 'signal.decomposition.ssa': "Leading components=8 squared singular values of each signal's "
                             'trajectory matrix normalized by total squared singular values; '
                             'window defaults to min(max(4,length//3),64), unavailable components '
                             'zero-padded, too-short rows NaN.',
 'signal.decomposition.vmd': 'Mean squared amplitude of modes=5 vmdpy VMD modes per row; fixed '
                             'alpha=2000, tau=0, DC=0, init=1 and tolerance=1e-7.',
 'signal.entropy.sample_entropy': 'Finite-filtered sample-entropy helper with fixed m=2, '
                                  'r=0.2*population std and inclusive Chebyshev matching; short '
                                  'input gives NaN, constant gives zero, zero-match cases use '
                                  'special finite/infinite conventions.',
 'signal.entropy.spectral_entropy': 'Shannon entropy of centered real-FFT powers divided by '
                                    'log(number of bins); DC forced zero, nan_to_num after '
                                    'centering, zero-power rows yield zero.',
 'signal.envelope.hilbert': 'Mean, population std and maximum analytic-signal magnitude after '
                            'nan_to_num, using SciPy Hilbert transform per row; three envelope '
                            'summaries.',
 'signal.fractal.hurst': 'Log-log slope of finite-series increment standard deviation against '
                         'logarithmically spaced lags, one value per row; fewer than 20 finite '
                         'samples or fewer than two positive deviations give NaN, not '
                         'rescaled-range analysis.',
 'signal.frequency.fft': 'First n_coeffs=16 available real-FFT magnitudes after row nanmean '
                         'centering and nan_to_num; DC is forced to zero, no amplitude '
                         'normalization or phase output.',
 'signal.frequency.psd': 'First n_bins=16 available squared centered real-FFT magnitudes per row, '
                         'DC forced zero; unnormalized power bins, not Welch density.',
 'signal.higher_order_spectra.bispectrum': 'Magnitudes of diagonal products X[i]^2*conj(X[2i]) '
                                           'divided by row length after mean centering; n_freq=6, '
                                           'unavailable entries NaN-padded, not a full bispectral '
                                           'surface.',
 'signal.modulation.amplitude_modulation': 'Dominant non-DC FFT magnitude frequency of the Hilbert '
                                           'envelope in cycles/sample and its fraction of total '
                                           'envelope-spectrum magnitude; nan_to_num before '
                                           'Hilbert, two outputs per row.',
 'signal.nonlinear.lyapunov': "Delegates to FEATM's 3D delay-embedding nearest-neighbor "
                              'mean-log-divergence slope; delay chosen by autocorrelation, up to '
                              '11 evolution steps, fewer than 30 embedded points gives NaN, no '
                              'physical-time scaling.',
 'signal.oscillation.frequency': 'Dominant centered real-FFT bin frequency in cycles/sample and '
                                 'its fraction of total power; DC zero, denominator floored at '
                                 '1e-15, two outputs per row.',
 'signal.peaks.peak_statistics': 'Peak count, mean and maximum prominence from SciPy find_peaks '
                                 'after nan_to_num, prominence threshold max(0.5*row '
                                 'nanstd,1e-12); no peaks give zero summaries.',
 'signal.phase.instantaneous_phase': 'Mean, population std and mean absolute adjacent change in '
                                     'unwrapped Hilbert phase, in radians per sample step; '
                                     'nan_to_num before transform, returns summaries rather than a '
                                     'phase sequence.',
 'signal.synchronization.phase_locking': 'Magnitude of the mean complex phase difference between '
                                         'two Hilbert analytic signals; input (N,2,T) or (2,T), '
                                         'one phase-locking value per pair, no missing-value '
                                         'filtering.',
 'signal.time_domain.crest_factor': 'NaN-ignoring peak absolute sample divided by RMS with a 1e-15 '
                                    'denominator floor; one scalar per row.',
 'signal.time_domain.impulse_factor': 'NaN-ignoring peak absolute sample divided by mean absolute '
                                      'amplitude with a 1e-15 denominator floor; one scalar per '
                                      'row.',
 'signal.time_domain.line_length': 'NaN-ignoring sum of absolute adjacent-sample differences per '
                                   'row; no elapsed-time or length normalization.',
 'signal.time_domain.peak_to_peak': 'NaN-ignoring maximum minus minimum sample per signal row, in '
                                    'input amplitude units.',
 'signal.time_domain.rms': 'Square root of NaN-ignoring mean squared samples per signal row; one '
                           'scalar in input amplitude units.',
 'signal.time_frequency.stft': 'Time-averaged SciPy STFT squared magnitudes sampled at bins=16 '
                               'evenly spaced frequency indices; nan_to_num, segment length '
                               'min(128,row length), default sampling convention rather than Hz '
                               'calibration.',
 'signal.wavelet.cwt': 'Mean squared magnitude of same-length convolution with mean-subtracted '
                       'unnormalized complex Morlet-like kernels at scales=(2,4,8,16); nan_to_num, '
                       'one value per scale.',
 'signal.wavelet.dwt': 'Mean squared Haar details at levels=4 followed by final approximation '
                       'energy; odd tails are dropped, nan_to_num before decomposition, '
                       'unavailable detail levels yield NaN.',
 'spatial.autocorrelation.geary_c': 'Mean squared difference from symmetric k=8 neighbors divided '
                                    'by twice the global population value variance plus 1e-15; one '
                                    'local value per coordinate row.',
 'spatial.autocorrelation.moran_i': 'Local centered value times summed neighbor deviations, '
                                    'divided by degree times population variance with a 1e-15 '
                                    'floor; symmetric binary k=8 neighbor graph, requires '
                                    'coordinates and values.',
 'spatial.clustering.spatial_clustering': 'DBSCAN label and size of its label group per point, '
                                          'including a shared noise group labeled -1; '
                                          'min_samples=5, eps defaults to median neighbor radius '
                                          'with a 1e-12 floor.',
 'spatial.density.kernel_density': 'Exponentiated sklearn KernelDensity log-density evaluated on '
                                   'the fitted coordinates; bandwidth defaults to the median '
                                   'fifth-neighbor radius with a 1e-6 floor.',
 'spatial.entropy.spatial_entropy': 'Shannon entropy in nats of globally quantile-discretized '
                                    "values among each point's k=8 nearest other points; bins=8, "
                                    'requires coordinates and values.',
 'spatial.graph.spatial_graph': 'k-neighbor count, mean distance and maximum distance per point '
                                'from cKDTree; k=8 capped by cloud size, no separate graph object '
                                'is returned.',
 'spatial.hotspots.getis_ord': "Mean supplied value over each point's symmetric binary k=8 "
                               'neighbor set; a local neighbor average, not a standardized '
                               'Getis-Ord significance score.',
 'spatial.neighbors.nearest_neighbor': 'Nearest distance, mean k-neighbor distance and '
                                       'kth-neighbor radius per coordinate row; k=8 capped at '
                                       'available other points, using SciPy cKDTree.',
 'spatial.point_patterns.pair_correlation': 'Per-point fractions of all n distance slots falling '
                                            'in each pair-distance quantile shell; bins=4, self '
                                            'slot remains in denominator and upper edges are '
                                            'excluded. Not annular-area-normalized g(r).',
 'spatial.point_patterns.ripley_k': 'Per-point counts of other points within each radius divided '
                                    'by planar bounding-box density; default radii are '
                                    'pair-distance quantiles 0.1/0.25/0.5/0.75. No boundary '
                                    'correction is applied.',
 'spatial.tessellation.voronoi': 'Per-point Voronoi ridge-neighbor count and finite 2D cell area; '
                                 'unbounded cells have NaN area, and the delegated area helper '
                                 'rejects non-2D coordinates.',
 'spatial.triangulation.delaunay': 'Per-point Delaunay neighbor degree and mean incident edge '
                                   'length, delegated to geometry helpers; a coordinate matrix '
                                   'needs at least d+1 points in d dimensions, with d>=2 and '
                                   'backend degeneracies still able to raise.',
 'tabular.categorical.binary_encoding': 'Least-significant-first bits=16 of lexicographically '
                                        'fitted string-category codes; unseen codes are -1 and '
                                        "emit all ones, missing is '<NA>', excess high bits are "
                                        'discarded.',
 'tabular.categorical.cooccurrence': 'Fitted count of each pair of string categories concatenated '
                                     "with '|'; missing uses '<NA>', unseen keys map to zero, "
                                     'embedded separators can make different pairs share a key.',
 'tabular.categorical.count': 'Fitted category count per cell, optionally reusing a supplied '
                              "mapping; categories cast to strings with missing sentinel '<NA>', "
                              'unseen values map to zero.',
 'tabular.categorical.embeddings': 'Deterministic dim=8 standard-normal vector seeded by each '
                                   "string category's BLAKE2b hash; missing uses '<NA>', vectors "
                                   'are random encodings rather than learned semantic embeddings.',
 'tabular.categorical.entropy': 'Empirical Shannon entropy in nats of the entire string-category '
                                "column, repeated on every row; missing values are the '<NA>' "
                                'category, not omitted.',
 'tabular.categorical.hashing': 'Stable BLAKE2b 8-byte little-endian hash of each string category '
                                "modulo buckets=1024, returned as float; missing uses '<NA>', "
                                'collisions are possible.',
 'tabular.categorical.is_rare': 'Flag fitted string-category frequency below threshold=0.01; '
                                "missing uses '<NA>' category, unseen categories have frequency "
                                'zero and are flagged for positive thresholds.',
 'tabular.categorical.rarity': 'Negative log of fitted category frequency floored at 1/max(current '
                               "row count,1); string categories with '<NA>' sentinel, unseen "
                               'categories use that floor.',
 'tabular.categorical.string_length': 'Character count of each string-cast category, with missing '
                                      "values replaced by literal '<NA>' and therefore counted as "
                                      'four characters.',
 'tabular.categorical.transition': 'Flag each string category that differs from the preceding row; '
                                   "missing uses '<NA>', first row compares against a missing "
                                   'shifted predecessor and can yield a missing result.',
 'tabular.datetime.astronomical': 'Solar-declination radians and equation-of-time minutes from '
                                  'fixed Fourier polynomials of day-of-year with a 365.2425-day '
                                  'period; no latitude, longitude or time-of-day input.',
 'tabular.datetime.business_calendar': 'Flag dates whose weekday is not Saturday or Sunday; no '
                                       'holiday calendar is used, and invalid timestamps also '
                                       'receive one under the current membership rule.',
 'tabular.datetime.calendar': 'Seven timestamp components per cell: year, month, day, weekday '
                              '(Monday=0), day-of-year, quarter and hour; pandas parse failures '
                              'yield missing components.',
 'tabular.datetime.cyclical': 'Sine/cosine pairs for month/12, weekday/7 and hour/24 times 2*pi; '
                              'month starts at one, weekday Monday=0, six outputs, invalid dates '
                              'yield NaNs.',
 'tabular.datetime.dayofyear_cos': 'Cosine of 2*pi*day_of_year/365.2425 using day indices starting '
                                   'at one; parse failures yield NaN, fixed year period including '
                                   'leap years.',
 'tabular.datetime.dayofyear_sin': 'Sine of 2*pi*day_of_year/365.2425 using day indices starting '
                                   'at one; parse failures yield NaN, fixed year period including '
                                   'leap years.',
 'tabular.datetime.elapsed_time': 'Seconds since the minimum timestamp fitted from the input '
                                  'column or supplied as state; parse failures yield NaN, no '
                                  'per-entity grouping.',
 'tabular.datetime.holidays': "Holiday-membership flag from the holidays package's country "
                              "calendar, default country='US', populated for input years; invalid "
                              'dates yield NaN.',
 'tabular.datetime.hour': 'Integer hour component 0..23 per pandas-parsed timestamp, returned as '
                          'float; invalid timestamps yield NaN.',
 'tabular.datetime.hour_cos': 'Cosine of 2*pi*hour/24 using integer hours 0..23; datetime parse '
                              'failures yield NaN, minutes and seconds do not affect phase.',
 'tabular.datetime.hour_sin': 'Sine of 2*pi*hour/24 using integer hours 0..23; datetime parse '
                              'failures yield NaN, minutes and seconds do not affect phase.',
 'tabular.datetime.month_cos': 'Cosine of 2*pi*calendar_month/12 using months 1..12; datetime '
                               'parse failures yield NaN.',
 'tabular.datetime.month_sin': 'Sine of 2*pi*calendar_month/12 using months 1..12; datetime parse '
                               'failures yield NaN, January is not phase zero.',
 'tabular.geographic.unit_x': 'cos(latitude)*cos(longitude) unit-sphere Cartesian coordinate per '
                              'row; first two input columns are latitude/longitude degrees, no '
                              'Earth-radius scaling.',
 'tabular.geographic.unit_y': 'cos(latitude)*sin(longitude) unit-sphere Cartesian coordinate per '
                              'row; first two input columns are latitude/longitude degrees, no '
                              'Earth-radius scaling.',
 'tabular.geographic.unit_z': 'sin(latitude) unit-sphere Cartesian coordinate per row; first input '
                              'column is latitude degrees, at least two coordinate columns '
                              'required.',
 'tabular.interactions.abs_difference': 'Absolute elementwise difference of two numeric-coerced '
                                        'columns with pandas index alignment; missing values '
                                        'propagate.',
 'tabular.interactions.difference': 'First numeric-coerced column minus the second, with pandas '
                                    'index alignment; unparseable cells become NaN.',
 'tabular.interactions.logical': 'AND, OR and XOR after testing each numeric-coerced value !=0; '
                                 'three float flags per row, NaN compares unequal to zero and is '
                                 'treated as true.',
 'tabular.interactions.pairwise': 'Per-row sum, first-minus-second difference, product, ratio and '
                                  'absolute difference of two numeric columns; zero denominator '
                                  'gives NaN, array positions are paired.',
 'tabular.interactions.polynomial': 'Five per-row products of two numeric columns: xy, x squared, '
                                    'y squared, x squared*y and x*y squared; no fitting or '
                                    'centering.',
 'tabular.interactions.sum': 'Elementwise sum of two numeric-coerced columns with pandas index '
                             'alignment; unparseable cells become NaN.',
 'tabular.interactions.three_way': 'Product of three numeric-coerced columns using pandas '
                                   'alignment; one value per row, missing values propagate.',
 'tabular.missingness.co_missingness': 'One where both aligned input cells are pandas-missing, '
                                       'zero otherwise; two-column joint missingness indicator.',
 'tabular.missingness.entropy': "Binary Shannon entropy in nats of the column's missing fraction, "
                                'repeated for every row; entirely present or entirely missing '
                                'columns give zero. Source extraction fits mergeable '
                                'whole-column counts before transforming batches.',
 'tabular.missingness.indicator_categorical': 'One for a pandas-missing categorical cell and zero '
                                              'otherwise; literal missing-looking strings are not '
                                              'treated as missing.',
 'tabular.missingness.indicator_numeric': 'One for a pandas-missing input cell and zero otherwise; '
                                          'tests original missingness without attempting numeric '
                                          'coercion.',
 'tabular.missingness.patterns': 'Alias-style single-column pandas missingness indicator, one per '
                                 'missing cell; does not compute multicolumn pattern frequencies.',
 'tabular.numeric.abs': 'Absolute value of each numeric-coerced cell; unparseable values become '
                        'NaN, no whole-column fitting.',
 'tabular.numeric.arithmetic': 'Original numeric-coerced value, its negation and its absolute '
                               'value; three outputs per cell.',
 'tabular.numeric.binning': 'Index from digitize against unique finite-value quantile edges fitted '
                            'with bins=10, optionally reusing supplied state; missing inputs are '
                            'digitized rather than kept as NaN.',
 'tabular.numeric.cube_signed': 'Third power of each numeric-coerced cell, preserving negative '
                                'signs; unparseable values become NaN.',
 'tabular.numeric.ecdf': 'Whole-column maximum-tie rank divided by nonmissing count, the empirical '
                         'cumulative fraction at each numeric value; NaNs remain missing.',
 'tabular.numeric.exponential': 'Exponential of each numeric-coerced value clipped to [-50,50] '
                                'before exponentiation; missing/unparseable cells remain NaN.',
 'tabular.numeric.hyperbolic': 'Sinh, cosh and tanh per numeric-coerced cell after clipping all '
                               'three inputs to [-20,20]; three outputs.',
 'tabular.numeric.logarithmic': 'Three per-cell transforms: sign(x)*log1p(abs(x)), log1p(max(x,0)) '
                                'and log10(1+abs(x)); numeric coercion, no fitting.',
 'tabular.numeric.minmax': '(x-column minimum)/(column maximum-minimum), optionally reusing '
                           'supplied state; constant columns produce NaN, legacy single-shot '
                           'transform rather than mergeable scaling.minmax.',
 'tabular.numeric.percentile': 'Whole-column average-tie rank divided by nonmissing count using '
                               'pandas pct=True; NaNs remain missing, not an interpolated '
                               'percentile-value transform.',
 'tabular.numeric.piecewise': 'Clip each numeric-coerced cell to [-1,1], leaving values inside '
                              'unchanged and missing values NaN.',
 'tabular.numeric.powers': 'Second, third and fourth powers per numeric-coerced cell, without '
                           'clipping or normalization; three outputs.',
 'tabular.numeric.quantization': 'Round each numeric-coerced cell to the nearest integer with '
                                 "NumPy's ties-to-even rule; values stay numeric, missing values "
                                 'remain NaN.',
 'tabular.numeric.rank': 'Whole-column ascending average-tie rank per numeric cell; NaNs retain '
                         'missing ranks, multibatch recomputation is rejected by the legacy '
                         'execution guard.',
 'tabular.numeric.ratios': 'Reciprocal and x/(1+abs(x)) per numeric cell; zero reciprocal is NaN, '
                           'no whole-column fitting.',
 'tabular.numeric.roots': 'Square root of absolute value, signed cube root and fourth root of '
                          'absolute value per numeric cell; three outputs, coercion failures '
                          'remain NaN.',
 'tabular.numeric.sigmoid': 'Logistic 1/(1+exp(-x)) per numeric cell after clipping x to [-50,50]; '
                            'missing/unparseable cells remain NaN.',
 'tabular.numeric.trigonometric': 'Sine and cosine of each numeric value in radians plus tangent '
                                  'after clipping to [-1.4,1.4]; three outputs.',
 'tabular.numeric.zscore': '(x-column mean)/population std, optionally reusing supplied state; '
                           'zero std gives NaN, legacy single-shot transform rather than mergeable '
                           'scaling.zscore.',
 'tabular.ordinal.cumulative': 'Sorted nonmissing category rank divided by max(number of fitted '
                               'levels-1,1); supplied state may be reused, missing or unseen '
                               'values yield NaN.',
 'tabular.ordinal.threshold': 'Flags code>=1 through the last fitted level using at most '
                              'max_levels=8 sorted categories; missing/unseen values emit zeros, '
                              'at most one fitted level emits one zero column.',
 'tabular.relational.entity_counts': 'Alias of fitted string-category counts per cell; missing '
                                     "uses '<NA>', supplied count mapping may be reused and unseen "
                                     'categories map to zero.',
 'tabular.relational.entity_graph': 'Left-category count, right-category count and '
                                    'concatenated-pair multiplicity per relation row; computes '
                                    'column counts rather than distinct-neighbor graph degrees, '
                                    "missing uses '<NA>'.",
 'tabular.relational.groupby': 'Per-row group mean, sample std (ddof=1), median and nonmissing '
                               'count of a numeric column grouped by string categories; recomputes '
                               'on the current call even when state is supplied.',
 'tabular.relational.recency': 'Seconds from each timestamp to the fitted column maximum or '
                               'supplied reference state; parse failures give NaN, no per-entity '
                               'grouping.',
 'tabular.statistical.moments': "First through fourth powers of each value's "
                                'population-standardized z-score, four outputs per row; fitted '
                                'mean/std can be reused, zero std gives NaNs, does not return four '
                                'aggregate moments.',
 'tabular.statistical.order_statistics': 'Whole-column average-tie rank and rank divided by total '
                                         'row count, including missing rows in the divisor; NaN '
                                         'ranks remain missing.',
 'tabular.statistical.outlier_scores': 'Absolute population z-score and absolute median/MAD score '
                                       'scaled by 1.4826 per row; fitted statistics reusable, zero '
                                       'scales yield NaN.',
 'tabular.statistical.quantiles': 'Five flags x<=fitted column quantiles at 10/25/50/75/90 '
                                  'percent; quantiles use pandas defaults, state can be reused and '
                                  'missing comparisons emit zero.',
 'tabular.statistical.robust': 'Median-centered value divided by 1.4826*MAD and by IQR, two '
                               'outputs per row; fitted statistics can be reused, a zero '
                               'corresponding scale yields NaN.',
 'tabular.statistical.tail': 'Four strict tail flags against fitted column 1/5/95/99 percentiles: '
                             'below q01/q05, above q95/q99; supplied state reusable, missing '
                             'comparisons give zero.',
 'tabular.text.char_count': 'Character length of each string-cast cell, replacing missing values '
                            'by an empty string and therefore zero; not UTF-8 byte count.',
 'tabular.text.digit_ratio': 'Fraction of Python isdigit characters in each string-cast cell, '
                             'missing replaced by empty string; empty strings yield zero.',
 'tabular.text.word_count': 'Whitespace-separated token count per string-cast cell, missing values '
                            "replaced by an empty string; differs from the text modality's regex "
                            'tokenizer.',
 'text.bm25.bm25': 'BM25 score against required query using lowercase regex tokens, k1=1.5 and '
                   'b=0.75; repeated query terms contribute repeatedly, in-memory calls fit their '
                   'corpus and stateful execution freezes training statistics.',
 'text.character.character_ngrams': 'Normalized stable-hash counts of contiguous n=3 raw character '
                                    'grams into buckets=64 per document; case and spaces are '
                                    'retained, no grams yield zeros.',
 'text.character.punctuation': 'Per-character frequencies of fourteen fixed punctuation marks, '
                               'including quotes and brackets, divided by total document character '
                               'count; empty documents give zeros.',
 'text.character.unicode': 'Character fractions in Unicode categories L/N/P/S/Z/M and the '
                           'non-ASCII fraction; seven values per document with empty documents '
                           'returning zeros.',
 'text.embeddings.document': 'Alias of SentenceTransformer document encoding, default '
                             'sentence-transformers/all-MiniLM-L6-v2; forwards model_name, no '
                             'additional document-level aggregation is performed.',
 'text.embeddings.sentence': 'SentenceTransformer encode vector per document, default '
                             "sentence-transformers/all-MiniLM-L6-v2; uses the model's own "
                             'preprocessing and output width, requiring available weights.',
 'text.embeddings.transformer': 'Mean of last-hidden-state token vectors per document from default '
                                'distilbert-base-uncased; tokenizer truncates at 512 tokens, '
                                'special tokens are included, model weights required.',
 'text.embeddings.word': 'Mean of user-supplied model[word] vectors over lowercase regex tokens; '
                         'lookup failures are skipped and no successful lookups yield zeros of dim '
                         'or model.vector_size.',
 'text.emotion.emotion': 'Hugging Face text-classification label scores, sorted by first-result '
                         'labels; default j-hartmann/emotion-english-distilroberta-base, first '
                         '5000 characters only, requires available model weights.',
 'text.information.character_entropy': 'Empirical character-frequency Shannon entropy in nats per '
                                       'document, including whitespace and case; empty documents '
                                       'return zero.',
 'text.information.compression': 'zlib-compressed UTF-8 byte count divided by max(original byte '
                                 'count,1); one value per document, including compression overhead '
                                 'for empty strings and short texts.',
 'text.information.entropy': 'Character and lowercase regex word-token Shannon entropies in nats, '
                             'two values per document; empty distributions return zero.',
 'text.information.surprisal': 'Mean negative natural log of add-alpha=1 smoothed corpus token '
                               'probabilities per document; empty documents score zero, in-memory '
                               'calls fit their supplied corpus while stateful execution uses '
                               'frozen fit counts.',
 'text.lexical.avg_word_length': 'Mean character length of lowercase regex word tokens per '
                                 'document; no tokens return zero.',
 'text.lexical.char_count': 'Python character count per normalized document, including spaces and '
                            'punctuation; not UTF-8 byte length.',
 'text.lexical.digit_ratio': 'Fraction of document characters satisfying Python isdigit, including '
                             'Unicode digits; empty documents return zero.',
 'text.lexical.lexical_diversity': 'Alias of distinct-to-total lowercase regex word-token ratio '
                                   'per document; no tokens return zero.',
 'text.lexical.ngrams': 'Normalized stable-hash counts of contiguous n=2 lowercase word-token '
                        'grams into buckets=64 per document; collisions combine counts, no grams '
                        'yield zeros.',
 'text.lexical.punctuation_ratio': 'Fraction of characters that are neither alphanumeric nor '
                                   'whitespace; includes symbols, empty documents return zero.',
 'text.lexical.unique_word_ratio': 'Distinct lowercase regex word tokens divided by total token '
                                   'count per document; no tokens return zero.',
 'text.lexical.vocabulary': 'Number of distinct lowercase regex word tokens within each document, '
                            'not a corpus-wide vocabulary size.',
 'text.lexical.word_count': 'Count of lowercase Unicode regex word tokens per document; tokenizer '
                            'retains word characters, apostrophes and hyphens within word '
                            'boundaries.',
 'text.morphology.morphemes': 'Mean, population std and maximum number of spaCy morphological '
                              "attributes per nonspace token; model='en_core_web_sm' must be "
                              'installed, counts attributes rather than segmented morphemes.',
 'text.readability.flesch': 'Flesch reading ease, grade, words/sentence and syllables/word using '
                            'regex sentence splitting and an English vowel-group syllable '
                            'heuristic; counts are floored at one, not a language-model '
                            'assessment.',
 'text.semantics.semantic_density': 'Mean off-diagonal cosine similarity between per-document '
                                    'sentence TF-IDF vectors fitted by sklearn; .!? splits '
                                    'sentences, fewer than two sentences return one.',
 'text.semantics.similarity': 'TF-IDF dot similarity to reference, default first document; '
                              'in-memory sklearn fits docs plus reference, whereas stateful '
                              'execution freezes training-only vocabulary/IDF and its first '
                              'training document as default reference.',
 'text.semantics.topic': 'Whole-corpus sklearn TF-IDF (up to 2000 terms) followed by NMF topic '
                         'weights; n_topics=8 capped by matrix shape, random_state=0, not '
                         'integrated with frozen fit/transform state.',
 'text.sentiment.sentiment': 'Hugging Face text-classification scores for labels sorted from the '
                             'first result; default '
                             'distilbert-base-uncased-finetuned-sst-2-english, input truncated to '
                             '5000 characters and model weights required.',
 'text.stylometry.style': 'Seven per-document values: word/sentence counts, mean word length, '
                          'distinct-word ratio and uppercase/digit/punctuation character '
                          'fractions; regex words and .!? sentence splitting.',
 'text.syntax.dependencies': 'Normalized stable-hash counts of spaCy dependency-label strings into '
                             "buckets=32 per document; model='en_core_web_sm', labels can collide "
                             'and all parsed tokens are counted.',
 'text.syntax.parse_depth': 'Mean and maximum ancestor-chain depth across spaCy tokens, stopping '
                            "at a self-headed root or cycle; model='en_core_web_sm', empty "
                            'documents yield zeros.',
 'text.syntax.pos': 'Twelve normalized spaCy POS-category counts per document, excluding space '
                    "tokens and combining unlisted tags into OTHER; model='en_core_web_sm' must be "
                    'available.',
 'text.tfidf.tfidf': 'L2-normalized raw token counts times smoothed IDF log((1+N)/(1+df))+1; '
                     'max_features=128 selects frequent training terms, sorted vocabulary is '
                     'frozen for stateful transform, empty vocabulary emits one zero column.',
 'timeseries.autocorrelation.acf': 'Pearson correlations of each finite-filtered series with its '
                                   'lagged copies at lags 1..max_lag (default 10); one vector per '
                                   'row. Short or constant slices yield NaN.',
 'timeseries.autocorrelation.autocovariance': 'Mean lagged products after subtracting the row '
                                              'nanmean and replacing NaNs by zero; max_lag=8, one '
                                              'vector per row. Insufficient length yields NaN.',
 'timeseries.autocorrelation.lag1': 'Pearson correlation of adjacent samples after filtering '
                                    'nonfinite values per series row; one scalar. Short or '
                                    'constant slices yield NaN.',
 'timeseries.autocorrelation.pacf': 'Statsmodels Yule-Walker MLE partial autocorrelations at up to '
                                    'nlags=8 after finite filtering; computed lag count capped by '
                                    'length, exceptions and unavailable trailing lags yield NaNs.',
 'timeseries.basic.derivatives': 'Mean absolute first/second NumPy gradient and their population '
                                 'std per row, ignoring NaNs in summaries; unit sample spacing and '
                                 'default gradient endpoints, four outputs.',
 'timeseries.basic.differences': 'Three statistics of adjacent first differences per series row: '
                                 'NaN-ignoring mean, population standard deviation and median.',
 'timeseries.basic.expanding': 'Final cumulative mean, population std and range of cumulative '
                               'means per row; nan_to_num values are included in cumulative '
                               'counts, so NaNs contribute zero rather than being excluded.',
 'timeseries.basic.lags': 'Mean raw lagged products at fixed lags 1, 2, 4 and 8 per series row, '
                          'ignoring NaNs; not normalized autocorrelation. Too-short lags yield '
                          'NaN.',
 'timeseries.basic.rolling': 'Four summaries per series row of sliding means and population '
                             'standard deviations: mean/std of means, mean/max of stds. window=8; '
                             'shorter rows yield four NaNs.',
 'timeseries.change_point.mean_shift': 'Absolute difference between NaN-ignoring means of the '
                                       'first floor(n/2) samples and the remaining samples; one '
                                       'fixed midpoint contrast, no change-location search.',
 'timeseries.change_point.regime_count': 'Count of adjacent changes in a moving average exceeding '
                                         'median+threshold*1.4826*MAD; threshold=2.5, '
                                         'window=max(4,length//20), NaNs mean-imputed, zero MAD '
                                         'returns zero.',
 'timeseries.change_point.variance_shift': 'Absolute difference between NaN-ignoring population '
                                           'variances of the two midpoint halves; one scalar per '
                                           'row, not a change-point significance test.',
 'timeseries.complexity.approximate_entropy': 'Approximate entropy per finite-filtered series row '
                                              'with fixed m=2 and r=0.2*population std; includes '
                                              'self matches and uses natural logarithms. Constant '
                                              'rows return zero; short rows NaN.',
 'timeseries.complexity.compressibility': 'zlib-compressed raw numeric row bytes divided by '
                                          'original byte count; empty rows return zero, short rows '
                                          'may exceed one, no amplitude normalization.',
 'timeseries.complexity.fuzzy_entropy': 'Log ratio of mean '
                                        'exp(-Chebyshev_distance**n/max(r,1e-12)) between length-m '
                                        'and m+1 windows, excluding self; m=2,n=2,r=0.2*nanstd, no '
                                        'local window demeaning, invalid ratios yield NaN.',
 'timeseries.complexity.lempel_ziv': 'Lempel-Ziv-style phrase counter on a binary string formed by '
                                     'thresholding at the row nanmedian; fixed helper defaults, '
                                     'NaNs map to the median/zero symbol, unnormalized count.',
 'timeseries.complexity.multiscale_entropy': 'Fixed-helper sample entropy after nonoverlapping '
                                             'block averaging at scales=(1,2,3,4,5); discarded '
                                             'tails, coarse rows of length<=5 give NaN, '
                                             'm=2/r=0.2*std within each scale.',
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
 'timeseries.cross_series.coherence': 'Means of SciPy magnitude-squared coherence across bands=4 '
                                      'contiguous bin groups; paired-series nan_to_num input, '
                                      'segment length min(128,length), empty bands give NaN.',
 'timeseries.cross_series.cross_correlation': 'Pearson correlations of overlapping paired-series '
                                              'slices at lags -max_lag..max_lag, default 5; '
                                              'positive lag compares later A with earlier B, short '
                                              'or constant slices give NaN.',
 'timeseries.cross_series.dynamic_time_warping': 'Unconstrained dynamic-time-warping cost using '
                                                 'absolute sample differences and three '
                                                 'predecessor moves, divided by the sum of paired '
                                                 'sequence lengths; no finite filtering.',
 'timeseries.cross_series.granger': 'Statsmodels SSR F-test p-values for A predicting B at lags '
                                    '1..maxlag=2; input columns ordered [B,A], exceptions produce '
                                    'NaNs, no missing-value preprocessing.',
 'timeseries.cross_series.lead_lag': 'Lag and signed Pearson coefficient with largest absolute '
                                     'overlap correlation within +/-max_lag=12; positive lag '
                                     'compares later A to earlier B, all-NaN correlations raise an '
                                     'error.',
 'timeseries.cross_series.phase_synchrony': 'Magnitude of mean complex Hilbert phase difference '
                                            'for each series pair after nan_to_num; one '
                                            'phase-locking value per pair.',
 'timeseries.cross_series.transfer_entropy': 'One-step A-to-B conditional-MI estimate from B[1:], '
                                             'A[:-1], B[:-1]; bins=8 quantile discretization with '
                                             'within-conditioning-bin rebinning, natural-log '
                                             'units.',
 'timeseries.dynamical.attractor_geometry': 'Three descending covariance eigenvalues and '
                                            'sqrt(max(covariance determinant,0)) of the '
                                            'autocorrelation-delay 3D embedding; fewer than five '
                                            'embedded points yields four NaNs.',
 'timeseries.dynamical.correlation_dimension': 'Log-log slope of positive embedded-pair distance '
                                               'fractions against their 10%-50% quantile radii; 3D '
                                               'autocorrelation-delay embedding, at most 400 '
                                               'evenly indexed points, insufficient distances give '
                                               'NaN.',
 'timeseries.dynamical.embedding_dimension': 'First dimension whose nearest-neighbor '
                                             'added-coordinate ratio exceeds 10 for under 5% of '
                                             'points; max_dim=6, autocorrelation-selected delay, '
                                             'fewer than 20 embedding rows returns current '
                                             'dimension.',
 'timeseries.dynamical.lyapunov': 'Slope of mean log nearest-neighbor separation over up to 11 '
                                  'forward steps in a 3D delay embedding; neighbors must be '
                                  'separated by >2*delay in time, fewer than 30 embedding rows '
                                  'yields NaN, no physical-time scaling.',
 'timeseries.dynamical.phase_space': 'Autocorrelation-selected delay, ascending covariance '
                                     'eigenvalues and covariance determinant of a dim=3 delay '
                                     'embedding; fewer than three embedded points gives NaNs, not '
                                     'embedded coordinates themselves.',
 'timeseries.dynamical.poincare': 'Count of strict median crossings and mean/std of their index '
                                  'gaps per series; no crossing interpolation or phase-space '
                                  'section, fewer than two crossings gives NaN gap summaries.',
 'timeseries.dynamical.time_delay': 'First lag with finite-filtered Pearson autocorrelation below '
                                    '1/e, searching up to min(50,length//3); fallback delay is one '
                                    'sample.',
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
 'timeseries.fractal.katz': 'Katz-style log10(n)/(log10(n)+log10(d/L)) on finite samples, with '
                            'L=sum(sqrt(1+diff^2)) and d=max distance from (0,x0); fewer than two '
                            'samples gives NaN.',
 'timeseries.fractal.multifractal': 'Generalized means of positive finite absolute first '
                                    'differences at qs=(-2,-1,0,1,2), q=0 using geometric mean; '
                                    'five amplitude summaries, not multifractal scaling exponents.',
 'timeseries.fractal.petrosian': 'Petrosian log-ratio formula using the count of strict sign '
                                 'reversals in adjacent first differences and factor 0.4; fixed '
                                 'helper, no finite filtering, length below three gives NaN.',
 'timeseries.recurrence.determinism': 'Fraction of recurrent entries in diagonal runs of length>=2 '
                                      'after removing the diagonal; fixed 2D delay-1 embedding, at '
                                      'most 600 finite points, 10th-percentile positive-distance '
                                      'threshold. Empty matrix gives NaN.',
 'timeseries.recurrence.laminarity': 'Fraction of recurrent entries in vertical runs of length>=2 '
                                     'after removing the diagonal; fixed 2D delay-1 embedding, at '
                                     'most 600 finite points and 10th-percentile positive-distance '
                                     'threshold. Empty matrix gives NaN.',
 'timeseries.recurrence.recurrence_network': 'Recurrence density, mean degree and triangle count '
                                             'after diagonal removal; fixed 2D delay-1 embedding, '
                                             'at most 600 finite points, 10th-percentile '
                                             'positive-distance threshold, empty matrix yields '
                                             'three NaNs.',
 'timeseries.recurrence.recurrence_rate': 'Fraction of true entries in the recurrence matrix after '
                                          'diagonal removal, denominator includes all matrix '
                                          'cells; fixed 2D delay-1 embedding, finite rows, at most '
                                          '600 points, 10th-percentile positive-distance '
                                          'threshold.',
 'timeseries.recurrence.trapping_time': 'Mean vertical recurrence-run length among runs>=2, in '
                                        'embedded-index steps; fixed 2D delay-1 embedding, at most '
                                        '600 finite points, default distance quantile threshold. '
                                        'No runs gives zero, empty matrix NaN.',
 'timeseries.seasonality.harmonic': 'Magnitudes of the n_harmonics=4 strongest non-DC centered FFT '
                                    'bins, ordered weakest to strongest within the selected set; '
                                    'does not require integer multiples of a fundamental '
                                    'frequency.',
 'timeseries.seasonality.periodicity': 'Reciprocal dominant centered-FFT frequency in samples and '
                                       "that bin's power fraction; DC forced zero, zero frequency "
                                       'gives NaN period and zero-power rows give zero strength.',
 'timeseries.seasonality.seasonal_strength': 'Largest centered non-DC FFT-bin power divided by '
                                             'total power; zero-power rows give zero, a single-bin '
                                             'concentration measure rather than '
                                             'decomposition-based seasonal strength.',
 'timeseries.spectral.bandpower': 'Sum of centered real-FFT power within bands=4 nearly equal '
                                  'groups of bin indices; DC forced zero, empty groups yield zero, '
                                  'no frequency-width or total-power normalization.',
 'timeseries.spectral.bispectrum': 'Magnitudes of diagonal FFT triple products X[i]^2*conj(X[2i]) '
                                   'divided by series length, after nanmean centering and '
                                   'nan_to_num; n_freq=6, unavailable entries NaN-padded.',
 'timeseries.spectral.centroid': 'Magnitude-weighted centroid of the mean-centered one-sided FFT '
                                 'per series row, in cycles per sample; zero total magnitude '
                                 'yields zero.',
 'timeseries.spectral.cepstrum': 'First n_coeffs=12 available inverse-real-FFT coefficients of '
                                 'floored log centered-spectrum magnitude; adaptive floor and zero '
                                 'DC, no output padding.',
 'timeseries.spectral.dominant_frequency': 'Frequency of the largest mean-centered one-sided FFT '
                                           'power bin per series row, in cycles per sample; ties '
                                           'choose the first bin.',
 'timeseries.spectral.fft': 'First n_coeffs=16 available centered real-FFT magnitudes followed by '
                            'phases in radians; DC forced zero, near-zero phases stabilized, '
                            'nan_to_num after centering, width twice retained bins.',
 'timeseries.spectral.flatness': 'Geometric mean of power floored at 1e-15 divided by arithmetic '
                                 'mean power of the mean-centered one-sided FFT per series row; '
                                 'zero-power rows yield zero.',
 'timeseries.spectral.higher_order_spectra': 'Eight fixed diagonal bispectral magnitudes '
                                             'X[i]^2*conj(X[2i])/length after centering, '
                                             'absolute-valued and NaN-padded; not a full '
                                             'higher-order spectral tensor.',
 'timeseries.spectral.periodogram': 'First n_bins=16 available squared centered real-FFT '
                                    'magnitudes, DC forced zero; unnormalized raw powers, no '
                                    'padding or frequency-density scaling.',
 'timeseries.spectral.psd': 'Welch power spectral density per series row using SciPy defaults '
                            '(fs=1, nperseg=min(256,length)); first n_bins=16 values, padded with '
                            'NaNs when shorter.',
 'timeseries.spectral.spectral_entropy': 'Normalized natural-log Shannon entropy of uncentered '
                                         'one-sided FFT power per series row, divided by '
                                         'log(max(number of bins,2)); zero-power rows yield zero.',
 'timeseries.stationarity.adf': 'Statsmodels augmented Dickey-Fuller statistic, p-value and '
                                "selected lag after finite filtering, with autolag='AIC'; "
                                'exceptions produce three NaNs, remaining settings use backend '
                                'defaults.',
 'timeseries.stationarity.kpss': 'Statsmodels KPSS statistic, p-value and lag count after finite '
                                 "filtering, regression='c' and nlags='auto'; exceptions yield "
                                 'three NaNs.',
 'timeseries.stationarity.variance_ratio': 'NaN-ignoring population variance of lag-step changes '
                                           'divided by lag times the variance of first '
                                           'differences; lag=2, nonpositive first-difference '
                                           'variance yields NaN.',
 'timeseries.statistical.change_statistics': 'Four statistics of first differences per series row: '
                                             'mean, population standard deviation, mean absolute '
                                             'difference and maximum absolute difference, ignoring '
                                             'NaNs.',
 'timeseries.statistical.crossings': 'Counts of adjacent sign-bit changes and strict crossings of '
                                     'the row nanmedian; zero is nonnegative, exact-median touches '
                                     'do not count as median crossings, NaNs are not prefiltered.',
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
 'timeseries.topology.betti_curves': '1+count(H0 MST deaths>threshold) at n_thresholds=8 evenly '
                                     'spaced thresholds from zero to largest death; 2D delay '
                                     'embedding, at most 300 finite points, no deaths yields NaNs.',
 'timeseries.topology.persistence': 'Largest n=8 positive H0 MST death lengths from a finite 2D '
                                    'autocorrelation-delay embedding of at most 300 points; '
                                    'ascending retained tail left-padded with zeros.',
 'timeseries.topology.persistence_image': 'Death-length-weighted H0 MST histogram with bins=8 '
                                          'spanning zero to largest death, from a 2D delay '
                                          'embedding of at most 300 finite points; no Gaussian '
                                          'smoothing, no deaths gives zeros.',
 'timeseries.topology.persistence_landscape': 'Maximum H0 bar tent sampled at n_grid=8 points from '
                                              'zero to largest MST death in a 2D delay embedding; '
                                              'at most 300 finite points, no deaths gives zeros.',
 'timeseries.trend.linear': 'Least-squares slope, intercept and R-squared against zero-based '
                            'sample indices per series row; nonfinite values are excluded. '
                            'Constant rows have NaN R-squared.',
 'timeseries.trend.linear_slope': 'Least-squares slope against zero-based sample indices per '
                                  'series row, excluding nonfinite values; units are amplitude per '
                                  'sample. Fewer than two finite samples yield NaN.',
 'timeseries.trend.piecewise': 'Least-squares slopes against local unit-spaced indices in '
                               'segments=3 contiguous nearly equal parts; nonfinite values '
                               'excluded, fewer than two finite values in a segment yields NaN.',
 'timeseries.trend.polynomial': 'Polynomial least-squares coefficients in descending degree '
                                'against sample positions scaled to [-1,1]; degree=2, nonfinite '
                                'values excluded, insufficient finite points yield NaNs.',
 'timeseries.trend.robust': 'SciPy Theil-Sen slope, intercept and default 95% slope interval per '
                            'series row against sample indices, excluding nonfinite values; '
                            'fitting exceptions yield four NaNs.',
 'timeseries.wavelet.cwt': 'Mean squared same-length convolution magnitude at scales=(2,4,8,16) '
                           'using unnormalized mean-subtracted complex Morlet-like kernels; '
                           'nan_to_num input, one value per scale.',
 'timeseries.wavelet.dwt': 'Mean squared Haar details at levels=4 plus final approximation energy; '
                           'nan_to_num input, odd tails discarded. Early decomposition termination '
                           'on short rows can leave fewer values than declared names.',
 'timeseries.wavelet.scattering': 'Four fixed-scale Morlet-like convolution energies followed by '
                                  'their square roots; eight values per row, not a learned or '
                                  'multilayer wavelet-scattering transform.',
 'timeseries.wavelet.wavelet_entropy': 'Shannon entropy of Haar detail and final approximation '
                                       'energies divided by log(output width); levels=4, inherits '
                                       'DWT short-row output-width limitations.',
 'timeseries.wavelet.wavelet_packet': 'Mean squared Haar packet-node amplitudes at depth=3, giving '
                                      '2**depth values; odd tails discarded, nodes too short to '
                                      'split are duplicated, nan_to_num input.',
 'topology.betti.numbers': 'Counts of all returned Rips persistence intervals in H0/H1/H2, '
                           'including infinite ones, rather than Betti numbers at a chosen scale. '
                           'GUDHI is attempted; unavailable or failed calls fall back to SciPy H0 '
                           'MST only.',
 'topology.euler.characteristic': 'H0 interval count minus H1 count plus H2 count from Rips '
                                  'persistence, not Euler characteristic at a fixed filtration '
                                  'scale. GUDHI is attempted; unavailable or failed calls fall '
                                  'back to SciPy H0 MST only.',
 'topology.local_topology.local_persistence': 'Mean, population std and maximum of total positive '
                                              'MST edge length in k=20-point neighborhoods of at '
                                              'most max_centers=50 evenly indexed centers; H0-only '
                                              'summaries per cloud.',
 'topology.mapper.mapper': 'PCA-one-axis overlapping-cover DBSCAN graph summary: node count, '
                           'overlap-edge count and max(nodes-edges,1) component proxy; '
                           'n_intervals=8, overlap=0.3, proxy is not an exact component count.',
 'topology.persistence.entropy': 'Shannon entropy in nats of positive Euclidean '
                                 'minimum-spanning-tree edge lengths normalized to sum one; '
                                 'H0-only, at most 400 evenly indexed points, no edges gives zero.',
 'topology.persistence.image': 'Histogram of positive H0 MST death lengths, weighted by those same '
                               'lengths, with bins=16 spanning zero to maximum death; at most 400 '
                               'points, not a Gaussian persistence image.',
 'topology.persistence.landscape': 'First H0 tent-envelope sampled at grid=16 times from zero to '
                                   'the maximum positive MST edge length; at most 400 evenly '
                                   'indexed points, no intervals gives zeros.',
 'topology.persistent_homology.alpha_complex': 'Nine alpha-complex persistence summaries: count '
                                               'and max/total finite lifetime for H0/H1/H2; '
                                               'infinite bars counted but excluded from lifetimes. '
                                               'GUDHI is attempted, with SciPy H0-only MST '
                                               'fallback on failure/unavailability.',
 'topology.persistent_homology.cech': 'Nine Cech persistence summaries: count and max/total finite '
                                      'lifetime for H0/H1/H2; attempts GUDHI CechComplex with '
                                      'bounding-box diagonal max radius, then SciPy H0-only MST '
                                      'fallback if unsupported, unavailable or unsuccessful.',
 'topology.persistent_homology.cubical_complex': 'Nine GUDHI cubical-persistence summaries per '
                                                 'image: interval count, maximum finite lifetime '
                                                 'and total finite lifetime for H0/H1/H2; counts '
                                                 'include infinite intervals, lifetime sums '
                                                 'exclude them.',
 'topology.persistent_homology.vietoris_rips': 'Nine Rips persistence summaries: count and '
                                               'max/total finite lifetime for H0/H1/H2; counts '
                                               'include infinite bars. GUDHI is attempted, with '
                                               'SciPy H0-only MST fallback if unavailable or '
                                               'unsuccessful.',
 'topology.persistent_homology.witness_complex': 'Nine witness-complex persistence summaries: '
                                                 'count and max/total finite lifetime for '
                                                 'H0/H1/H2, up to 50 evenly indexed landmarks; '
                                                 'GUDHI attempted, with SciPy H0-only MST fallback '
                                                 'on failure/unavailability.',
 'video.action.statistics': 'Three grayscale clip summaries: squared adjacent-frame difference '
                            'energy, mean pixel temporal population standard deviation and maximum '
                            'scene-change difference; no action classifier is run.',
 'video.camera_motion.global_flow': 'Alias of mean frame-pair median Farneback dx/dy; two '
                                    'displacement summaries per clip, without camera calibration '
                                    'or motion-model fitting.',
 'video.dynamic_texture.lbp_top': 'Concatenated normalized uniform-LBP histograms on the middle '
                                  'frame, middle-row time/width slice and middle-column '
                                  'time/height slice; each plane rescales to uint8, P=8/R=1, '
                                  'bins=10 per plane.',
 'video.embeddings.3d_cnn': 'Flattened vector returned by a required user callable model(video) '
                            'per clip; no built-in 3D CNN, pretrained weights or pooling rule is '
                            'supplied.',
 'video.embeddings.timesformer': 'Flattened vector returned by a required user callable '
                                 'model(video) per clip; no built-in TimeSformer, pretrained '
                                 'weights or pooling rule is supplied.',
 'video.embeddings.video_mae': 'Flattened vector returned by a required user callable model(video) '
                               'per clip; no built-in VideoMAE, pretrained weights or pooling rule '
                               'is supplied.',
 'video.facial_motion.landmarks': 'Flattened fixed-width vector returned by a required user '
                                  'callable model(video), one per clip; this adapter supplies no '
                                  'facial-landmark detector.',
 'video.frame_statistics.intensity': 'Mean, population standard deviation, minimum and maximum of '
                                     'per-frame mean grayscale intensity, four values per clip.',
 'video.frame_statistics.mean_intensity': 'Mean grayscale intensity across all frames and pixels '
                                          'per clip; RGB conversion uses weights '
                                          '0.2126/0.7152/0.0722 without intensity rescaling.',
 'video.motion.frame_difference_energy': 'Mean squared grayscale difference between adjacent '
                                         'frames over all pixels and frame pairs; one value per '
                                         'clip, not divided by elapsed time.',
 'video.motion.motion_histograms': 'Magnitude-weighted Farneback direction histogram accumulated '
                                   'across all adjacent frame pairs and normalized to sum one; '
                                   'bins=8, no motion gives zeros.',
 'video.motion.motion_vectors': 'Mean over adjacent frame pairs of spatial median Farneback dx and '
                                'dy, two values per clip in pixels per frame pair; no pairs yield '
                                'zeros.',
 'video.motion.optical_flow': 'Six Farneback-flow summaries: mean/std/max of frame-pair mean '
                              'magnitudes, mean frame-pair median dx/dy and standard deviation of '
                              'frame-pair circular mean angles; no pairs yield zeros.',
 'video.motion.trajectories': 'Mean frame-pair median Farneback dx, dy and their Euclidean norm '
                              'per clip; summarizes flow without tracking object trajectories or '
                              'applying a frame rate.',
 'video.object_tracks.track_count': 'Flattened fixed-width vector returned by a required user '
                                    'callable model(video), one per clip; this adapter does not '
                                    'itself detect or count tracks.',
 'video.pose_dynamics.joint_velocity': 'Flattened fixed-width vector returned by a required user '
                                       'callable model(video), one per clip; pose detection and '
                                       'joint-velocity definitions belong to the supplied model.',
 'video.scene_change.cuts': 'Count and mean strength of frame differences exceeding '
                            'mean+threshold_std*population standard deviation, plus that '
                            'threshold; threshold_std=3, strength is mean absolute grayscale '
                            'change.',
 'video.scene_change.scene_change_max': 'Largest adjacent-frame mean absolute grayscale difference '
                                        'per clip; clips with no frame pairs return zero.',
 'video.shot_statistics.duration': 'Mean, population standard deviation, maximum duration and '
                                   'count of shots separated by frame differences above '
                                   'mean+3*std; fps=30 converts frame counts to seconds.',
 'video.temporal_frequency.fft': 'First bins=8 available unnormalized real-FFT powers of '
                                 'mean-centered frame-average grayscale intensities; one vector '
                                 'per clip, no frame-rate frequency calibration.',
 'video.temporal_statistics.temporal_std': "Mean across pixels of each pixel's temporal population "
                                           'standard deviation after grayscale conversion; one '
                                           'value per clip.',
 'video.temporal_statistics.variation': 'Mean, population standard deviation and maximum of '
                                        'adjacent-frame mean absolute grayscale differences; three '
                                        'values per clip, requiring at least two frames for the '
                                        'maximum.',
 'video.track_geometry.velocity': 'Alias of the three Farneback flow displacement summaries dx, dy '
                                  'and norm; no object tracks or elapsed-time velocity are '
                                  'computed.'}

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
