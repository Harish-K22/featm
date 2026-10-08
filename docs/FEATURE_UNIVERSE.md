# featm feature universe

This document is generated from the executable registry. Every entry below has an implementation callable. Some entries require an optional scientific backend, pretrained model, or user-supplied model at execution time.

**Total implemented entries:** 552  
**Modalities:** 18

## Counts by modality

| Modality | Features |
| --- | ---: |
| tabular | 90 |
| timeseries | 89 |
| graph | 50 |
| image | 50 |
| audio | 36 |
| text | 33 |
| geometry | 32 |
| pointcloud | 27 |
| signal | 27 |
| video | 23 |
| information | 18 |
| sequence | 16 |
| matrix | 12 |
| spatial | 12 |
| topology | 12 |
| manifold | 9 |
| event | 8 |
| multimodal | 8 |

## Complete registry

### audio

- ✅ `audio.embeddings.clap` — implemented (optional: torch, transformers; output=vector; cost=extreme). audio.embeddings.clap
- ✅ `audio.embeddings.hubert` — implemented (optional: torch, transformers; output=vector; cost=extreme). audio.embeddings.hubert
- ✅ `audio.embeddings.wav2vec` — implemented (optional: torch, transformers; output=vector; cost=extreme). audio.embeddings.wav2vec
- ✅ `audio.embeddings.whisper` — implemented (optional: torch, transformers; output=vector; cost=extreme). audio.embeddings.whisper
- ✅ `audio.perceptual.bark` — implemented (output=vector; cost=medium). audio.perceptual.bark
- ✅ `audio.perceptual.gfcc` — implemented (output=vector; cost=medium). audio.perceptual.gfcc
- ✅ `audio.perceptual.loudness` — implemented (cost=tiny). audio.perceptual.loudness
- ✅ `audio.perceptual.mel` — implemented (optional: librosa; output=vector; cost=medium). audio.perceptual.mel
- ✅ `audio.perceptual.mfcc` — implemented (optional: librosa; output=vector; cost=medium). audio.perceptual.mfcc
- ✅ `audio.rhythm.beats` — implemented (optional: librosa; output=vector; cost=medium). audio.rhythm.beats
- ✅ `audio.rhythm.onset` — implemented (optional: librosa; output=vector; cost=medium). audio.rhythm.onset
- ✅ `audio.rhythm.tempo` — implemented (optional: librosa; cost=medium). audio.rhythm.tempo
- ✅ `audio.rhythm.tempogram` — implemented (optional: librosa; output=vector; cost=high). audio.rhythm.tempogram
- ✅ `audio.spatial.ild` — implemented (cost=tiny). audio.spatial.ild
- ✅ `audio.spatial.itd` — implemented (cost=medium). audio.spatial.itd
- ✅ `audio.spectral.bandwidth` — implemented (optional: librosa). audio.spectral.bandwidth
- ✅ `audio.spectral.centroid` — implemented (optional: librosa). audio.spectral.centroid
- ✅ `audio.spectral.centroid_normalized` — implemented (optional: librosa). audio.spectral.centroid_normalized
- ✅ `audio.spectral.contrast` — implemented (optional: librosa; output=vector; cost=medium). audio.spectral.contrast
- ✅ `audio.spectral.flatness` — implemented (optional: librosa). audio.spectral.flatness
- ✅ `audio.spectral.flux` — implemented (optional: librosa; cost=medium). audio.spectral.flux
- ✅ `audio.spectral.rolloff` — implemented (optional: librosa). audio.spectral.rolloff
- ✅ `audio.tonal.chroma` — implemented (optional: librosa; output=vector; cost=medium). audio.tonal.chroma
- ✅ `audio.tonal.harmonicity` — implemented (cost=medium). audio.tonal.harmonicity
- ✅ `audio.tonal.key` — implemented (optional: librosa; output=vector; cost=medium). audio.tonal.key
- ✅ `audio.tonal.pitch` — implemented (optional: librosa; cost=medium). audio.tonal.pitch
- ✅ `audio.tonal.tonnetz` — implemented (optional: librosa; output=vector; cost=high). audio.tonal.tonnetz
- ✅ `audio.voice.f0` — implemented (optional: librosa; cost=medium). audio.voice.f0
- ✅ `audio.voice.formants` — implemented (optional: librosa; output=vector; cost=high). audio.voice.formants
- ✅ `audio.voice.hnr` — implemented (cost=medium). audio.voice.hnr
- ✅ `audio.voice.jitter` — implemented (optional: librosa; cost=high). audio.voice.jitter
- ✅ `audio.voice.shimmer` — implemented (cost=high). audio.voice.shimmer
- ✅ `audio.waveform.crest_factor` — implemented (cost=tiny). audio.waveform.crest_factor
- ✅ `audio.waveform.dynamic_range` — implemented (cost=tiny). audio.waveform.dynamic_range
- ✅ `audio.waveform.rms` — implemented (cost=tiny). audio.waveform.rms
- ✅ `audio.waveform.zero_crossing_rate` — implemented (cost=tiny). audio.waveform.zero_crossing_rate

### event

- ✅ `event.point_process.temporal_point_process` — implemented (output=vector). event.point_process.temporal_point_process
- ✅ `event.process.hawkes` — implemented (optional: scipy; output=vector; cost=high). event.process.hawkes
- ✅ `event.sessions.session` — implemented (output=vector). event.sessions.session
- ✅ `event.survival.survival` — implemented (output=vector). event.survival.survival
- ✅ `event.timing.burstiness` — implemented (cost=tiny). event.timing.burstiness
- ✅ `event.timing.event_rate` — implemented (cost=tiny). event.timing.event_rate
- ✅ `event.timing.interarrival_time` — implemented (output=vector; cost=tiny). event.timing.interarrival_time
- ✅ `event.timing.recency` — implemented (cost=tiny). event.timing.recency

### geometry

- ✅ `geometry.alpha_shapes.alpha_shape` — implemented (optional: scipy; output=vector; cost=high). geometry.alpha_shapes.alpha_shape
- ✅ `geometry.angles.local_angles` — implemented (optional: scipy; output=vector; cost=medium). geometry.angles.local_angles
- ✅ `geometry.distances.euclidean` — implemented (optional: scipy; cost=tiny). geometry.distances.euclidean
- ✅ `geometry.distances.mahalanobis` — implemented (optional: scipy). geometry.distances.mahalanobis
- ✅ `geometry.hull.convex_hull` — implemented (optional: scipy; output=vector; cost=high). geometry.hull.convex_hull
- ✅ `geometry.hull.hull_area` — implemented (optional: scipy; cost=high). geometry.hull.hull_area
- ✅ `geometry.hull.hull_distance` — implemented (optional: scipy; cost=medium). geometry.hull.hull_distance
- ✅ `geometry.hull.hull_volume` — implemented (optional: scipy; cost=high). geometry.hull.hull_volume
- ✅ `geometry.hull.solidity` — implemented (optional: scipy; cost=high). geometry.hull.solidity
- ✅ `geometry.projections.projection` — implemented (optional: scipy; output=vector; cost=medium). geometry.projections.projection
- ✅ `geometry.proximity.gabriel_graph` — implemented (optional: scipy; output=vector; cost=extreme). geometry.proximity.gabriel_graph
- ✅ `geometry.proximity.minimum_spanning_tree` — implemented (optional: scipy; output=vector; cost=extreme). geometry.proximity.minimum_spanning_tree
- ✅ `geometry.proximity.relative_neighborhood_graph` — implemented (optional: scipy; output=vector; cost=extreme). geometry.proximity.relative_neighborhood_graph
- ✅ `geometry.stochastic_geometry.point_process` — implemented (optional: scipy; output=vector; cost=medium). geometry.stochastic_geometry.point_process
- ✅ `geometry.triangulation.aspect_ratio` — implemented (optional: scipy; cost=high). geometry.triangulation.aspect_ratio
- ✅ `geometry.triangulation.circumradius` — implemented (optional: scipy; cost=high). geometry.triangulation.circumradius
- ✅ `geometry.triangulation.delaunay_degree` — implemented (optional: scipy; cost=high). geometry.triangulation.delaunay_degree
- ✅ `geometry.triangulation.edge_length` — implemented (optional: scipy; cost=high). geometry.triangulation.edge_length
- ✅ `geometry.triangulation.inradius` — implemented (optional: scipy; cost=high). geometry.triangulation.inradius
- ✅ `geometry.triangulation.max_edge_length` — implemented (optional: scipy; cost=high). geometry.triangulation.max_edge_length
- ✅ `geometry.triangulation.mean_edge_length` — implemented (optional: scipy; cost=high). geometry.triangulation.mean_edge_length
- ✅ `geometry.triangulation.mean_simplex_volume` — implemented (optional: scipy; cost=high). geometry.triangulation.mean_simplex_volume
- ✅ `geometry.triangulation.min_edge_length` — implemented (optional: scipy; cost=high). geometry.triangulation.min_edge_length
- ✅ `geometry.triangulation.simplex_quality` — implemented (optional: scipy; cost=high). geometry.triangulation.simplex_quality
- ✅ `geometry.triangulation.simplex_volume` — implemented (optional: scipy; cost=high). geometry.triangulation.simplex_volume
- ✅ `geometry.triangulation.std_edge_length` — implemented (optional: scipy; cost=high). geometry.triangulation.std_edge_length
- ✅ `geometry.triangulation.triangle_angles` — implemented (optional: scipy; output=vector; cost=high). geometry.triangulation.triangle_angles
- ✅ `geometry.triangulation.triangle_area` — implemented (optional: scipy; cost=high). geometry.triangulation.triangle_area
- ✅ `geometry.voronoi.boundary_length` — implemented (optional: scipy; cost=high). geometry.voronoi.boundary_length
- ✅ `geometry.voronoi.cell_area` — implemented (optional: scipy; cost=high). geometry.voronoi.cell_area
- ✅ `geometry.voronoi.cell_volume` — implemented (optional: scipy; cost=extreme). geometry.voronoi.cell_volume
- ✅ `geometry.voronoi.neighbor_count` — implemented (optional: scipy; cost=high). geometry.voronoi.neighbor_count

### graph

- ✅ `graph.assortativity.degree` — implemented (optional: networkx; cost=medium). graph.assortativity.degree
- ✅ `graph.centrality.betweenness` — implemented (optional: networkx; output=vector; cost=high). graph.centrality.betweenness
- ✅ `graph.centrality.closeness` — implemented (optional: networkx; output=vector; cost=medium). graph.centrality.closeness
- ✅ `graph.centrality.current_flow` — implemented (optional: networkx; output=vector; cost=extreme). graph.centrality.current_flow
- ✅ `graph.centrality.degree` — implemented (optional: networkx; output=vector; cost=medium). graph.centrality.degree
- ✅ `graph.centrality.eigenvector` — implemented (optional: networkx; output=vector; cost=high). graph.centrality.eigenvector
- ✅ `graph.centrality.harmonic` — implemented (optional: networkx; output=vector; cost=medium). graph.centrality.harmonic
- ✅ `graph.centrality.katz` — implemented (optional: networkx; output=vector; cost=high). graph.centrality.katz
- ✅ `graph.centrality.pagerank` — implemented (optional: networkx; output=vector; cost=medium). graph.centrality.pagerank
- ✅ `graph.cliques.cliques` — implemented (optional: networkx; output=vector; cost=high). graph.cliques.cliques
- ✅ `graph.clustering.average_clustering` — implemented (optional: networkx; cost=medium). graph.clustering.average_clustering
- ✅ `graph.clustering.coefficient` — implemented (optional: networkx; output=vector; cost=medium). graph.clustering.coefficient
- ✅ `graph.clustering.transitivity` — implemented (optional: networkx; cost=medium). graph.clustering.transitivity
- ✅ `graph.clustering.triangles` — implemented (optional: networkx; output=vector; cost=medium). graph.clustering.triangles
- ✅ `graph.community.leiden` — implemented (optional: igraph, leidenalg; output=vector; cost=high). graph.community.leiden
- ✅ `graph.community.louvain` — implemented (optional: networkx; output=vector; cost=high). graph.community.louvain
- ✅ `graph.community.modularity` — implemented (optional: networkx; cost=high). graph.community.modularity
- ✅ `graph.connectivity.articulation` — implemented (optional: networkx; cost=medium). graph.connectivity.articulation
- ✅ `graph.connectivity.bridges` — implemented (optional: networkx; cost=medium). graph.connectivity.bridges
- ✅ `graph.connectivity.components` — implemented (optional: networkx). graph.connectivity.components
- ✅ `graph.connectivity.density` — implemented (optional: networkx; cost=tiny). graph.connectivity.density
- ✅ `graph.cores.k_core` — implemented (optional: networkx; output=vector; cost=medium). graph.cores.k_core
- ✅ `graph.cores.k_shell` — implemented (optional: networkx; output=vector; cost=medium). graph.cores.k_shell
- ✅ `graph.cores.k_truss` — implemented (optional: networkx; output=vector; cost=high). graph.cores.k_truss
- ✅ `graph.cores.onion_layers` — implemented (optional: networkx; output=vector; cost=medium). graph.cores.onion_layers
- ✅ `graph.cycles.cycle_count` — implemented (optional: networkx; cost=high). graph.cycles.cycle_count
- ✅ `graph.cycles.girth` — implemented (optional: networkx; cost=high). graph.cycles.girth
- ✅ `graph.degree.average_degree` — implemented (optional: networkx; cost=tiny). graph.degree.average_degree
- ✅ `graph.embeddings.deepwalk` — implemented (optional: gensim; output=vector; cost=extreme). graph.embeddings.deepwalk
- ✅ `graph.embeddings.gnn` — implemented (optional: user-model; output=vector; cost=extreme). graph.embeddings.gnn
- ✅ `graph.embeddings.node2vec` — implemented (optional: node2vec; output=vector; cost=extreme). graph.embeddings.node2vec
- ✅ `graph.graphlets.graphlets` — implemented (optional: networkx; output=vector; cost=high). graph.graphlets.graphlets
- ✅ `graph.homophily.homophily` — implemented (optional: networkx). graph.homophily.homophily
- ✅ `graph.link_prediction.common_neighbors` — implemented (optional: networkx; output=vector; cost=high). graph.link_prediction.common_neighbors
- ✅ `graph.mixing.mixing` — implemented (optional: networkx; output=vector; cost=medium). graph.mixing.mixing
- ✅ `graph.motifs.motifs` — implemented (optional: networkx; output=vector; cost=high). graph.motifs.motifs
- ✅ `graph.paths.diameter` — implemented (optional: networkx; cost=high). graph.paths.diameter
- ✅ `graph.paths.eccentricity` — implemented (optional: networkx; output=vector; cost=high). graph.paths.eccentricity
- ✅ `graph.paths.efficiency` — implemented (optional: networkx; cost=high). graph.paths.efficiency
- ✅ `graph.paths.radius` — implemented (optional: networkx; cost=high). graph.paths.radius
- ✅ `graph.paths.shortest_paths` — implemented (optional: networkx; output=vector; cost=high). graph.paths.shortest_paths
- ✅ `graph.reciprocity.reciprocity` — implemented (optional: networkx). graph.reciprocity.reciprocity
- ✅ `graph.size.n_edges` — implemented (optional: networkx; cost=tiny). graph.size.n_edges
- ✅ `graph.size.n_nodes` — implemented (optional: networkx; cost=tiny). graph.size.n_nodes
- ✅ `graph.spectral.algebraic_connectivity` — implemented (optional: networkx; cost=high). graph.spectral.algebraic_connectivity
- ✅ `graph.spectral.eigenvalues` — implemented (optional: networkx; output=vector; cost=high). graph.spectral.eigenvalues
- ✅ `graph.spectral.laplacian` — implemented (optional: networkx; output=vector; cost=high). graph.spectral.laplacian
- ✅ `graph.spectral.spectral_gap` — implemented (optional: networkx; cost=high). graph.spectral.spectral_gap
- ✅ `graph.topology.clique_complex` — implemented (optional: networkx; output=vector; cost=high). graph.topology.clique_complex
- ✅ `graph.topology.persistent_homology` — implemented (optional: networkx; output=vector; cost=high). graph.topology.persistent_homology

### image

- ✅ `image.color.blue_mean` — implemented (cost=tiny). image.color.blue_mean
- ✅ `image.color.colorfulness` — implemented (cost=tiny). image.color.colorfulness
- ✅ `image.color.green_mean` — implemented (cost=tiny). image.color.green_mean
- ✅ `image.color.hsv_histogram` — implemented (optional: opencv-python; output=vector; cost=medium). image.color.hsv_histogram
- ✅ `image.color.lab_histogram` — implemented (optional: opencv-python; output=vector; cost=medium). image.color.lab_histogram
- ✅ `image.color.red_mean` — implemented (cost=tiny). image.color.red_mean
- ✅ `image.color.rgb_histogram` — implemented (optional: opencv-python; output=vector; cost=medium). image.color.rgb_histogram
- ✅ `image.edges.edge_density` — implemented (optional: scikit-image). image.edges.edge_density
- ✅ `image.edges.gradient_energy` — implemented. Mean finite-difference gradient magnitude per image.
- ✅ `image.embeddings.clip` — implemented (optional: torch, transformers; output=vector; cost=extreme). image.embeddings.clip
- ✅ `image.embeddings.convnext` — implemented (optional: torch, timm; output=vector; cost=extreme). image.embeddings.convnext
- ✅ `image.embeddings.dino` — implemented (optional: torch, timm; output=vector; cost=extreme). image.embeddings.dino
- ✅ `image.embeddings.efficientnet` — implemented (optional: torch, timm; output=vector; cost=extreme). image.embeddings.efficientnet
- ✅ `image.embeddings.resnet` — implemented (optional: torch, timm; output=vector; cost=extreme). image.embeddings.resnet
- ✅ `image.embeddings.timm` — implemented (optional: torch, timm; output=vector; cost=extreme). image.embeddings.timm
- ✅ `image.embeddings.vit` — implemented (optional: torch, timm; output=vector; cost=extreme). image.embeddings.vit
- ✅ `image.frequency.dct` — implemented (optional: scipy; output=vector; cost=medium). image.frequency.dct
- ✅ `image.frequency.fft2` — implemented (output=vector; cost=medium). image.frequency.fft2
- ✅ `image.frequency.wavelet` — implemented (output=vector; cost=medium). image.frequency.wavelet
- ✅ `image.gradient.hog` — implemented (optional: scikit-image; output=vector; cost=medium). image.gradient.hog
- ✅ `image.gradient.structure_tensor` — implemented (optional: scikit-image; output=vector; cost=medium). image.gradient.structure_tensor
- ✅ `image.intensity.entropy` — implemented. image.intensity.entropy
- ✅ `image.intensity.mean` — implemented (cost=tiny). Per-image mean intensity over all pixels/channels.
- ✅ `image.intensity.std` — implemented (cost=tiny). Per-image intensity standard deviation.
- ✅ `image.keypoints.fast` — implemented (optional: opencv-python; output=vector; cost=medium). image.keypoints.fast
- ✅ `image.keypoints.orb` — implemented (optional: opencv-python; output=vector; cost=medium). image.keypoints.orb
- ✅ `image.keypoints.sift` — implemented (optional: opencv-python; output=vector; cost=medium). image.keypoints.sift
- ✅ `image.morphology.components` — implemented (optional: scikit-image; output=vector). image.morphology.components
- ✅ `image.quality.blur` — implemented (optional: opencv-python). image.quality.blur
- ✅ `image.quality.noise` — implemented (optional: scipy). image.quality.noise
- ✅ `image.saliency.statistics` — implemented (optional: scipy; output=vector; cost=medium). image.saliency.statistics
- ✅ `image.shape.area` — implemented (optional: scikit-image). image.shape.area
- ✅ `image.shape.circularity` — implemented (optional: scikit-image). image.shape.circularity
- ✅ `image.shape.eccentricity` — implemented (optional: scikit-image). image.shape.eccentricity
- ✅ `image.shape.fourier_descriptors` — implemented (optional: opencv-python; output=vector; cost=medium). image.shape.fourier_descriptors
- ✅ `image.shape.hu_moments` — implemented (optional: scikit-image; output=vector). image.shape.hu_moments
- ✅ `image.shape.perimeter` — implemented (optional: scikit-image). image.shape.perimeter
- ✅ `image.shape.solidity` — implemented (optional: scikit-image). image.shape.solidity
- ✅ `image.shape.zernike` — implemented (output=vector; cost=high). image.shape.zernike
- ✅ `image.symmetry.horizontal` — implemented (cost=tiny). image.symmetry.horizontal
- ✅ `image.symmetry.vertical` — implemented (cost=tiny). image.symmetry.vertical
- ✅ `image.texture.fractal` — implemented (cost=medium). image.texture.fractal
- ✅ `image.texture.gabor` — implemented (optional: skimage; output=vector; cost=high). image.texture.gabor
- ✅ `image.texture.glcm` — implemented (optional: scikit-image; output=vector; cost=medium). image.texture.glcm
- ✅ `image.texture.haralick` — implemented (optional: scikit-image; output=vector; cost=medium). image.texture.haralick
- ✅ `image.texture.laws` — implemented (optional: scipy; output=vector; cost=medium). image.texture.laws
- ✅ `image.texture.lbp` — implemented (optional: scikit-image; output=vector; cost=medium). image.texture.lbp
- ✅ `image.topology.euler_number` — implemented (optional: scikit-image). image.topology.euler_number
- ✅ `image.topology.holes` — implemented (optional: scikit-image). image.topology.holes
- ✅ `image.topology.persistent_homology` — implemented (optional: scikit-image; output=vector; cost=high). image.topology.persistent_homology

### information

- ✅ `information.dependence.conditional_mi` — implemented (cost=high). information.dependence.conditional_mi
- ✅ `information.dependence.interaction_information` — implemented (cost=high). information.dependence.interaction_information
- ✅ `information.dependence.mutual_information` — implemented (cost=medium). information.dependence.mutual_information
- ✅ `information.dependence.transfer_entropy` — implemented (cost=high). information.dependence.transfer_entropy
- ✅ `information.divergence.jensen_shannon` — implemented (cost=tiny). information.divergence.jensen_shannon
- ✅ `information.divergence.kl` — implemented (cost=tiny). information.divergence.kl
- ✅ `information.divergence.variation_of_information` — implemented (cost=medium). information.divergence.variation_of_information
- ✅ `information.entropy.conditional` — implemented (cost=medium). information.entropy.conditional
- ✅ `information.entropy.joint` — implemented (cost=medium). information.entropy.joint
- ✅ `information.entropy.renyi` — implemented (cost=tiny). information.entropy.renyi
- ✅ `information.entropy.shannon` — implemented (cost=tiny). information.entropy.shannon
- ✅ `information.entropy.tsallis` — implemented (cost=tiny). information.entropy.tsallis
- ✅ `information.geometry.information_geometry` — implemented (output=vector). information.geometry.information_geometry
- ✅ `information.multivariate.dual_total_correlation` — implemented (cost=high). information.multivariate.dual_total_correlation
- ✅ `information.multivariate.total_correlation` — implemented (cost=high). information.multivariate.total_correlation
- ✅ `information.pid.redundancy` — implemented (cost=extreme). information.pid.redundancy
- ✅ `information.pid.synergy` — implemented (cost=extreme). information.pid.synergy
- ✅ `information.pid.unique_information` — implemented (cost=extreme). information.pid.unique_information

### manifold

- ✅ `manifold.curvature.curvature` — implemented (optional: scipy, scikit-learn; cost=medium). manifold.curvature.curvature
- ✅ `manifold.dimension.intrinsic_dimension` — implemented (optional: scipy, scikit-learn; cost=medium). manifold.dimension.intrinsic_dimension
- ✅ `manifold.dimension.local_dimension` — implemented (optional: scipy, scikit-learn; cost=medium). manifold.dimension.local_dimension
- ✅ `manifold.distance.geodesic` — implemented (optional: scipy, scikit-learn; cost=high). manifold.distance.geodesic
- ✅ `manifold.embedding.diffusion_maps` — implemented (optional: scipy, scikit-learn; output=vector; cost=high). manifold.embedding.diffusion_maps
- ✅ `manifold.embedding.isomap` — implemented (optional: scipy, scikit-learn; output=vector; cost=high). manifold.embedding.isomap
- ✅ `manifold.embedding.laplacian_eigenmaps` — implemented (optional: scipy, scikit-learn; output=vector; cost=high). manifold.embedding.laplacian_eigenmaps
- ✅ `manifold.residual.local_manifold_residual` — implemented (optional: scipy, scikit-learn; cost=medium). manifold.residual.local_manifold_residual
- ✅ `manifold.tangent.local_tangent_space` — implemented (optional: scipy, scikit-learn; output=vector; cost=medium). manifold.tangent.local_tangent_space

### matrix

- ✅ `matrix.algebra.condition_number` — implemented (cost=medium). matrix.algebra.condition_number
- ✅ `matrix.algebra.determinant` — implemented (cost=tiny). matrix.algebra.determinant
- ✅ `matrix.algebra.eigenvalues` — implemented (output=vector; cost=medium). matrix.algebra.eigenvalues
- ✅ `matrix.algebra.frobenius_norm` — implemented (cost=tiny). matrix.algebra.frobenius_norm
- ✅ `matrix.algebra.norms` — implemented (output=vector). matrix.algebra.norms
- ✅ `matrix.algebra.rank` — implemented (cost=medium). matrix.algebra.rank
- ✅ `matrix.algebra.singular_values` — implemented (output=vector; cost=medium). matrix.algebra.singular_values
- ✅ `matrix.algebra.sparsity` — implemented (cost=tiny). matrix.algebra.sparsity
- ✅ `matrix.algebra.spectral_norm` — implemented (cost=tiny). matrix.algebra.spectral_norm
- ✅ `matrix.algebra.trace` — implemented (cost=tiny). matrix.algebra.trace
- ✅ `matrix.information.matrix_entropy` — implemented (cost=medium). matrix.information.matrix_entropy
- ✅ `matrix.spectral.spectral_statistics` — implemented (output=vector; cost=medium). matrix.spectral.spectral_statistics

### multimodal

- ✅ `multimodal.alignment.audio_text` — implemented (output=vector). multimodal.alignment.audio_text
- ✅ `multimodal.alignment.image_text` — implemented (output=vector). multimodal.alignment.image_text
- ✅ `multimodal.alignment.video_text` — implemented (output=vector). multimodal.alignment.video_text
- ✅ `multimodal.embedding_distances.cross_modal` — implemented (output=vector). multimodal.embedding_distances.cross_modal
- ✅ `multimodal.fusion.image_tabular` — implemented (output=vector). multimodal.fusion.image_tabular
- ✅ `multimodal.fusion.late_fusion` — implemented (output=vector). multimodal.fusion.late_fusion
- ✅ `multimodal.fusion.temporal_tabular` — implemented (output=vector). multimodal.fusion.temporal_tabular
- ✅ `multimodal.similarity.cross_modal` — implemented. multimodal.similarity.cross_modal

### pointcloud

- ✅ `pointcloud.descriptors.fpfh` — implemented (optional: open3d; output=vector; cost=high). pointcloud.descriptors.fpfh
- ✅ `pointcloud.descriptors.shape_context` — implemented (output=vector; cost=high). pointcloud.descriptors.shape_context
- ✅ `pointcloud.descriptors.shot` — implemented (optional: pclpy; output=vector; cost=extreme). pointcloud.descriptors.shot
- ✅ `pointcloud.descriptors.spin_images` — implemented (optional: pclpy; output=vector; cost=extreme). pointcloud.descriptors.spin_images
- ✅ `pointcloud.embeddings.point_transformer` — implemented (optional: user-model; output=vector; cost=extreme). pointcloud.embeddings.point_transformer
- ✅ `pointcloud.embeddings.pointnet` — implemented (optional: user-model; output=vector; cost=extreme). pointcloud.embeddings.pointnet
- ✅ `pointcloud.geometry.bbox_volume` — implemented (cost=tiny). pointcloud.geometry.bbox_volume
- ✅ `pointcloud.geometry.centroid_radius_mean` — implemented (cost=tiny). pointcloud.geometry.centroid_radius_mean
- ✅ `pointcloud.geometry.centroid_radius_std` — implemented (cost=tiny). pointcloud.geometry.centroid_radius_std
- ✅ `pointcloud.geometry.curvature` — implemented (output=vector; cost=medium). pointcloud.geometry.curvature
- ✅ `pointcloud.geometry.density` — implemented (output=vector; cost=medium). pointcloud.geometry.density
- ✅ `pointcloud.geometry.normals` — implemented (output=vector; cost=medium). pointcloud.geometry.normals
- ✅ `pointcloud.geometry.roughness` — implemented (output=vector; cost=medium). pointcloud.geometry.roughness
- ✅ `pointcloud.local_geometry.anisotropy` — implemented (output=vector; cost=medium). pointcloud.local_geometry.anisotropy
- ✅ `pointcloud.local_geometry.eigenentropy` — implemented (output=vector; cost=medium). pointcloud.local_geometry.eigenentropy
- ✅ `pointcloud.local_geometry.linearity` — implemented (output=vector; cost=medium). pointcloud.local_geometry.linearity
- ✅ `pointcloud.local_geometry.omnivariance` — implemented (output=vector; cost=medium). pointcloud.local_geometry.omnivariance
- ✅ `pointcloud.local_geometry.pca_linearity` — implemented (output=vector; cost=medium). pointcloud.local_geometry.pca_linearity
- ✅ `pointcloud.local_geometry.pca_planarity` — implemented (output=vector; cost=medium). pointcloud.local_geometry.pca_planarity
- ✅ `pointcloud.local_geometry.pca_sphericity` — implemented (output=vector; cost=medium). pointcloud.local_geometry.pca_sphericity
- ✅ `pointcloud.local_geometry.planarity` — implemented (output=vector; cost=medium). pointcloud.local_geometry.planarity
- ✅ `pointcloud.local_geometry.scattering` — implemented (output=vector; cost=medium). pointcloud.local_geometry.scattering
- ✅ `pointcloud.size.n_points` — implemented (cost=tiny). pointcloud.size.n_points
- ✅ `pointcloud.surface.surface_variation` — implemented (output=vector; cost=medium). pointcloud.surface.surface_variation
- ✅ `pointcloud.topology.persistence` — implemented (output=vector; cost=high). pointcloud.topology.persistence
- ✅ `pointcloud.triangulation.delaunay` — implemented (output=vector; cost=high). pointcloud.triangulation.delaunay
- ✅ `pointcloud.voxel.voxel_density` — implemented (output=vector). pointcloud.voxel.voxel_density

### sequence

- ✅ `sequence.alignment.alignment` — implemented (output=vector; cost=high). sequence.alignment.alignment
- ✅ `sequence.basic.length` — implemented (cost=tiny). sequence.basic.length
- ✅ `sequence.basic.unique_ratio` — implemented (cost=tiny). sequence.basic.unique_ratio
- ✅ `sequence.compression.compression` — implemented. sequence.compression.compression
- ✅ `sequence.distance.edit_distance` — implemented (cost=high). sequence.distance.edit_distance
- ✅ `sequence.entropy.shannon` — implemented (cost=tiny). sequence.entropy.shannon
- ✅ `sequence.entropy.shannon_entropy` — implemented (cost=tiny). sequence.entropy.shannon_entropy
- ✅ `sequence.markov.markov` — implemented (output=vector). sequence.markov.markov
- ✅ `sequence.motifs.motifs` — implemented (output=vector; cost=medium). sequence.motifs.motifs
- ✅ `sequence.ngrams.ngrams` — implemented (output=vector). sequence.ngrams.ngrams
- ✅ `sequence.run_length.run_count` — implemented (cost=tiny). sequence.run_length.run_count
- ✅ `sequence.run_length.runs` — implemented (output=vector; cost=tiny). sequence.run_length.runs
- ✅ `sequence.subsequences.subsequences` — implemented (output=vector; cost=medium). sequence.subsequences.subsequences
- ✅ `sequence.symbolic_dynamics.symbolic_dynamics` — implemented (output=vector; cost=medium). sequence.symbolic_dynamics.symbolic_dynamics
- ✅ `sequence.transitions.transition_matrix` — implemented (output=vector). sequence.transitions.transition_matrix
- ✅ `sequence.transitions.transition_rate` — implemented (cost=tiny). sequence.transitions.transition_rate

### signal

- ✅ `signal.cepstral.cepstrum` — implemented (output=vector; cost=medium). signal.cepstral.cepstrum
- ✅ `signal.decomposition.eemd` — implemented (optional: PyEMD; output=vector; cost=extreme). signal.decomposition.eemd
- ✅ `signal.decomposition.emd` — implemented (optional: PyEMD; output=vector; cost=extreme). signal.decomposition.emd
- ✅ `signal.decomposition.ica` — implemented (optional: scikit-learn; output=vector; cost=high). signal.decomposition.ica
- ✅ `signal.decomposition.ssa` — implemented (output=vector; cost=high). signal.decomposition.ssa
- ✅ `signal.decomposition.vmd` — implemented (optional: vmdpy; output=vector; cost=extreme). signal.decomposition.vmd
- ✅ `signal.entropy.sample_entropy` — implemented (cost=high). signal.entropy.sample_entropy
- ✅ `signal.entropy.spectral_entropy` — implemented (cost=medium). signal.entropy.spectral_entropy
- ✅ `signal.envelope.hilbert` — implemented (output=vector; cost=medium). signal.envelope.hilbert
- ✅ `signal.fractal.hurst` — implemented (cost=medium). signal.fractal.hurst
- ✅ `signal.frequency.fft` — implemented (output=vector; cost=medium). signal.frequency.fft
- ✅ `signal.frequency.psd` — implemented (output=vector; cost=medium). signal.frequency.psd
- ✅ `signal.higher_order_spectra.bispectrum` — implemented (output=vector; cost=high). signal.higher_order_spectra.bispectrum
- ✅ `signal.modulation.amplitude_modulation` — implemented (output=vector; cost=medium). signal.modulation.amplitude_modulation
- ✅ `signal.nonlinear.lyapunov` — implemented (cost=high). signal.nonlinear.lyapunov
- ✅ `signal.oscillation.frequency` — implemented (output=vector; cost=medium). signal.oscillation.frequency
- ✅ `signal.peaks.peak_statistics` — implemented (output=vector; cost=medium). signal.peaks.peak_statistics
- ✅ `signal.phase.instantaneous_phase` — implemented (output=vector; cost=medium). signal.phase.instantaneous_phase
- ✅ `signal.synchronization.phase_locking` — implemented (cost=medium). signal.synchronization.phase_locking
- ✅ `signal.time_domain.crest_factor` — implemented (cost=tiny). signal.time_domain.crest_factor
- ✅ `signal.time_domain.impulse_factor` — implemented (cost=tiny). signal.time_domain.impulse_factor
- ✅ `signal.time_domain.line_length` — implemented (cost=tiny). signal.time_domain.line_length
- ✅ `signal.time_domain.peak_to_peak` — implemented (cost=tiny). signal.time_domain.peak_to_peak
- ✅ `signal.time_domain.rms` — implemented (cost=tiny). signal.time_domain.rms
- ✅ `signal.time_frequency.stft` — implemented (output=vector; cost=medium). signal.time_frequency.stft
- ✅ `signal.wavelet.cwt` — implemented (output=vector; cost=high). signal.wavelet.cwt
- ✅ `signal.wavelet.dwt` — implemented (output=vector; cost=medium). signal.wavelet.dwt

### spatial

- ✅ `spatial.autocorrelation.geary_c` — implemented (optional: scipy; cost=medium). spatial.autocorrelation.geary_c
- ✅ `spatial.autocorrelation.moran_i` — implemented (optional: scipy; cost=medium). spatial.autocorrelation.moran_i
- ✅ `spatial.clustering.spatial_clustering` — implemented (optional: scipy; output=vector; cost=medium). spatial.clustering.spatial_clustering
- ✅ `spatial.density.kernel_density` — implemented (optional: scipy; cost=medium). spatial.density.kernel_density
- ✅ `spatial.entropy.spatial_entropy` — implemented (optional: scipy; cost=medium). spatial.entropy.spatial_entropy
- ✅ `spatial.graph.spatial_graph` — implemented (optional: scipy; output=vector; cost=medium). spatial.graph.spatial_graph
- ✅ `spatial.hotspots.getis_ord` — implemented (optional: scipy; cost=medium). spatial.hotspots.getis_ord
- ✅ `spatial.neighbors.nearest_neighbor` — implemented (optional: scipy; output=vector). spatial.neighbors.nearest_neighbor
- ✅ `spatial.point_patterns.pair_correlation` — implemented (optional: scipy; output=vector; cost=high). spatial.point_patterns.pair_correlation
- ✅ `spatial.point_patterns.ripley_k` — implemented (optional: scipy; output=vector; cost=high). spatial.point_patterns.ripley_k
- ✅ `spatial.tessellation.voronoi` — implemented (optional: scipy; output=vector; cost=high). spatial.tessellation.voronoi
- ✅ `spatial.triangulation.delaunay` — implemented (optional: scipy; output=vector; cost=high). spatial.triangulation.delaunay

### tabular

- ✅ `tabular.categorical.binary_encoding` — implemented (output=vector; cost=tiny). tabular.categorical.binary_encoding
- ✅ `tabular.categorical.cooccurrence` — implemented (cost=tiny). tabular.categorical.cooccurrence
- ✅ `tabular.categorical.count` — implemented (cost=tiny). tabular.categorical.count
- ✅ `tabular.categorical.embeddings` — implemented (output=vector; cost=tiny). tabular.categorical.embeddings
- ✅ `tabular.categorical.entropy` — implemented (cost=tiny). tabular.categorical.entropy
- ✅ `tabular.categorical.frequency` — implemented. Training-set category frequency; unseen categories map to zero.
- ✅ `tabular.categorical.hashing` — implemented (cost=tiny). tabular.categorical.hashing
- ✅ `tabular.categorical.is_rare` — implemented (cost=tiny). tabular.categorical.is_rare
- ✅ `tabular.categorical.rarity` — implemented (cost=tiny). tabular.categorical.rarity
- ✅ `tabular.categorical.string_length` — implemented (cost=tiny). tabular.categorical.string_length
- ✅ `tabular.categorical.transition` — implemented (cost=tiny). tabular.categorical.transition
- ✅ `tabular.datetime.astronomical` — implemented (output=vector; cost=tiny). tabular.datetime.astronomical
- ✅ `tabular.datetime.business_calendar` — implemented (cost=tiny). tabular.datetime.business_calendar
- ✅ `tabular.datetime.calendar` — implemented (output=vector; cost=tiny). tabular.datetime.calendar
- ✅ `tabular.datetime.cyclical` — implemented (output=vector; cost=tiny). tabular.datetime.cyclical
- ✅ `tabular.datetime.day` — implemented (cost=tiny). Calendar component: day.
- ✅ `tabular.datetime.dayofweek` — implemented (cost=tiny). Calendar component: dayofweek.
- ✅ `tabular.datetime.dayofweek_cos` — implemented (cost=tiny). Cyclical cosine encoding of day of week.
- ✅ `tabular.datetime.dayofweek_sin` — implemented (cost=tiny). Cyclical sine encoding of day of week.
- ✅ `tabular.datetime.dayofyear` — implemented (cost=tiny). Calendar component: dayofyear.
- ✅ `tabular.datetime.dayofyear_cos` — implemented (cost=tiny). tabular.datetime.dayofyear_cos
- ✅ `tabular.datetime.dayofyear_sin` — implemented (cost=tiny). tabular.datetime.dayofyear_sin
- ✅ `tabular.datetime.elapsed_time` — implemented (cost=tiny). tabular.datetime.elapsed_time
- ✅ `tabular.datetime.holidays` — implemented. tabular.datetime.holidays
- ✅ `tabular.datetime.hour` — implemented (cost=tiny). tabular.datetime.hour
- ✅ `tabular.datetime.hour_cos` — implemented (cost=tiny). tabular.datetime.hour_cos
- ✅ `tabular.datetime.hour_sin` — implemented (cost=tiny). tabular.datetime.hour_sin
- ✅ `tabular.datetime.month` — implemented (cost=tiny). Calendar component: month.
- ✅ `tabular.datetime.month_cos` — implemented (cost=tiny). tabular.datetime.month_cos
- ✅ `tabular.datetime.month_sin` — implemented (cost=tiny). tabular.datetime.month_sin
- ✅ `tabular.datetime.quarter` — implemented (cost=tiny). Calendar component: quarter.
- ✅ `tabular.datetime.year` — implemented (cost=tiny). Calendar component: year.
- ✅ `tabular.encoding.category_frequency` — implemented (stateful; approximate-available). Training-set category frequency encoding; categories unseen during fit map to 0.
- ✅ `tabular.geographic.unit_x` — implemented (cost=tiny). tabular.geographic.unit_x
- ✅ `tabular.geographic.unit_y` — implemented (cost=tiny). tabular.geographic.unit_y
- ✅ `tabular.geographic.unit_z` — implemented (cost=tiny). tabular.geographic.unit_z
- ✅ `tabular.interactions.abs_difference` — implemented (cost=tiny). tabular.interactions.abs_difference
- ✅ `tabular.interactions.difference` — implemented (cost=tiny). tabular.interactions.difference
- ✅ `tabular.interactions.logical` — implemented (output=vector; cost=tiny). tabular.interactions.logical
- ✅ `tabular.interactions.pairwise` — implemented (output=vector; cost=tiny). tabular.interactions.pairwise
- ✅ `tabular.interactions.polynomial` — implemented (output=vector; cost=tiny). tabular.interactions.polynomial
- ✅ `tabular.interactions.product` — implemented. Pairwise numeric product.
- ✅ `tabular.interactions.ratio_safe` — implemented. Pairwise numeric ratio with zero denominator mapped to NaN.
- ✅ `tabular.interactions.sum` — implemented (cost=tiny). tabular.interactions.sum
- ✅ `tabular.interactions.three_way` — implemented (cost=tiny). tabular.interactions.three_way
- ✅ `tabular.missingness.co_missingness` — implemented (cost=tiny). tabular.missingness.co_missingness
- ✅ `tabular.missingness.entropy` — implemented (cost=tiny). tabular.missingness.entropy
- ✅ `tabular.missingness.indicator_categorical` — implemented (cost=tiny). tabular.missingness.indicator_categorical
- ✅ `tabular.missingness.indicator_numeric` — implemented (cost=tiny). tabular.missingness.indicator_numeric
- ✅ `tabular.missingness.patterns` — implemented (cost=tiny). tabular.missingness.patterns
- ✅ `tabular.numeric.abs` — implemented (cost=tiny). tabular.numeric.abs
- ✅ `tabular.numeric.arithmetic` — implemented (output=vector; cost=tiny). tabular.numeric.arithmetic
- ✅ `tabular.numeric.binning` — implemented (cost=tiny). tabular.numeric.binning
- ✅ `tabular.numeric.cube_signed` — implemented (cost=tiny). tabular.numeric.cube_signed
- ✅ `tabular.numeric.ecdf` — implemented (cost=tiny). tabular.numeric.ecdf
- ✅ `tabular.numeric.exponential` — implemented (cost=tiny). tabular.numeric.exponential
- ✅ `tabular.numeric.hyperbolic` — implemented (output=vector; cost=tiny). tabular.numeric.hyperbolic
- ✅ `tabular.numeric.logarithmic` — implemented (output=vector; cost=tiny). tabular.numeric.logarithmic
- ✅ `tabular.numeric.minmax` — implemented (cost=tiny). tabular.numeric.minmax
- ✅ `tabular.numeric.percentile` — implemented (cost=tiny). tabular.numeric.percentile
- ✅ `tabular.numeric.piecewise` — implemented (cost=tiny). tabular.numeric.piecewise
- ✅ `tabular.numeric.powers` — implemented (output=vector; cost=tiny). tabular.numeric.powers
- ✅ `tabular.numeric.quantization` — implemented (cost=tiny). tabular.numeric.quantization
- ✅ `tabular.numeric.rank` — implemented (cost=tiny). tabular.numeric.rank
- ✅ `tabular.numeric.ratios` — implemented (output=vector; cost=tiny). tabular.numeric.ratios
- ✅ `tabular.numeric.reciprocal_safe` — implemented (cost=tiny). Safe reciprocal with zero mapped to NaN.
- ✅ `tabular.numeric.roots` — implemented (output=vector; cost=tiny). tabular.numeric.roots
- ✅ `tabular.numeric.sigmoid` — implemented (cost=tiny). tabular.numeric.sigmoid
- ✅ `tabular.numeric.signed_log1p` — implemented (cost=tiny). Sign-preserving log(1 + abs(x)).
- ✅ `tabular.numeric.sqrt_abs` — implemented (cost=tiny). Square root of absolute magnitude.
- ✅ `tabular.numeric.square` — implemented (cost=tiny). Second power.
- ✅ `tabular.numeric.trigonometric` — implemented (output=vector; cost=tiny). tabular.numeric.trigonometric
- ✅ `tabular.numeric.zscore` — implemented (cost=tiny). tabular.numeric.zscore
- ✅ `tabular.ordinal.cumulative` — implemented (cost=tiny). tabular.ordinal.cumulative
- ✅ `tabular.ordinal.threshold` — implemented (output=vector; cost=tiny). tabular.ordinal.threshold
- ✅ `tabular.relational.entity_counts` — implemented (cost=tiny). tabular.relational.entity_counts
- ✅ `tabular.relational.entity_graph` — implemented (output=vector; cost=tiny). tabular.relational.entity_graph
- ✅ `tabular.relational.groupby` — implemented (output=vector; cost=tiny). tabular.relational.groupby
- ✅ `tabular.relational.recency` — implemented (cost=tiny). tabular.relational.recency
- ✅ `tabular.scaling.minmax` — implemented (stateful; approximate-available). Min-max scaling fit via mergeable running extrema across batches.
- ✅ `tabular.scaling.zscore` — implemented (stateful; approximate-available). Z-score standardization fit via mergeable Welford accumulation across batches.
- ✅ `tabular.statistical.moments` — implemented (output=vector; cost=tiny). tabular.statistical.moments
- ✅ `tabular.statistical.order_statistics` — implemented (output=vector; cost=tiny). tabular.statistical.order_statistics
- ✅ `tabular.statistical.outlier_scores` — implemented (output=vector; cost=tiny). tabular.statistical.outlier_scores
- ✅ `tabular.statistical.quantiles` — implemented (output=vector; cost=tiny). tabular.statistical.quantiles
- ✅ `tabular.statistical.robust` — implemented (output=vector; cost=tiny). tabular.statistical.robust
- ✅ `tabular.statistical.tail` — implemented (output=vector; cost=tiny). tabular.statistical.tail
- ✅ `tabular.text.char_count` — implemented (cost=tiny). tabular.text.char_count
- ✅ `tabular.text.digit_ratio` — implemented (cost=tiny). tabular.text.digit_ratio
- ✅ `tabular.text.word_count` — implemented (cost=tiny). tabular.text.word_count

### text

- ✅ `text.bm25.bm25` — implemented (cost=medium). text.bm25.bm25
- ✅ `text.character.character_ngrams` — implemented (output=vector). text.character.character_ngrams
- ✅ `text.character.punctuation` — implemented (output=vector; cost=tiny). text.character.punctuation
- ✅ `text.character.unicode` — implemented (output=vector; cost=tiny). text.character.unicode
- ✅ `text.embeddings.document` — implemented (optional: sentence-transformers; output=vector; cost=extreme). text.embeddings.document
- ✅ `text.embeddings.sentence` — implemented (optional: sentence-transformers; output=vector; cost=extreme). text.embeddings.sentence
- ✅ `text.embeddings.transformer` — implemented (optional: transformers, torch; output=vector; cost=extreme). text.embeddings.transformer
- ✅ `text.embeddings.word` — implemented (optional: user-model; output=vector; cost=extreme). text.embeddings.word
- ✅ `text.emotion.emotion` — implemented (optional: transformers; output=vector; cost=extreme). text.emotion.emotion
- ✅ `text.information.character_entropy` — implemented (cost=tiny). text.information.character_entropy
- ✅ `text.information.compression` — implemented. text.information.compression
- ✅ `text.information.entropy` — implemented (output=vector; cost=tiny). text.information.entropy
- ✅ `text.information.surprisal` — implemented. text.information.surprisal
- ✅ `text.lexical.avg_word_length` — implemented (cost=tiny). text.lexical.avg_word_length
- ✅ `text.lexical.char_count` — implemented (cost=tiny). text.lexical.char_count
- ✅ `text.lexical.digit_ratio` — implemented (cost=tiny). text.lexical.digit_ratio
- ✅ `text.lexical.lexical_diversity` — implemented (cost=tiny). text.lexical.lexical_diversity
- ✅ `text.lexical.ngrams` — implemented (output=vector). text.lexical.ngrams
- ✅ `text.lexical.punctuation_ratio` — implemented (cost=tiny). text.lexical.punctuation_ratio
- ✅ `text.lexical.unique_word_ratio` — implemented (cost=tiny). text.lexical.unique_word_ratio
- ✅ `text.lexical.vocabulary` — implemented (cost=tiny). text.lexical.vocabulary
- ✅ `text.lexical.word_count` — implemented (cost=tiny). text.lexical.word_count
- ✅ `text.morphology.morphemes` — implemented (optional: spacy; output=vector; cost=high). text.morphology.morphemes
- ✅ `text.readability.flesch` — implemented (output=vector). text.readability.flesch
- ✅ `text.semantics.semantic_density` — implemented (optional: scikit-learn; cost=medium). text.semantics.semantic_density
- ✅ `text.semantics.similarity` — implemented (optional: scikit-learn; cost=medium). text.semantics.similarity
- ✅ `text.semantics.topic` — implemented (optional: scikit-learn; output=vector; cost=high). text.semantics.topic
- ✅ `text.sentiment.sentiment` — implemented (optional: transformers; output=vector; cost=extreme). text.sentiment.sentiment
- ✅ `text.stylometry.style` — implemented (output=vector). text.stylometry.style
- ✅ `text.syntax.dependencies` — implemented (optional: spacy; output=vector; cost=high). text.syntax.dependencies
- ✅ `text.syntax.parse_depth` — implemented (optional: spacy; output=vector; cost=high). text.syntax.parse_depth
- ✅ `text.syntax.pos` — implemented (optional: spacy; output=vector; cost=high). text.syntax.pos
- ✅ `text.tfidf.tfidf` — implemented (optional: scikit-learn; output=vector; cost=medium). text.tfidf.tfidf

### timeseries

- ✅ `timeseries.autocorrelation.acf` — implemented (output=vector; cost=medium). timeseries.autocorrelation.acf
- ✅ `timeseries.autocorrelation.autocovariance` — implemented (output=vector). timeseries.autocorrelation.autocovariance
- ✅ `timeseries.autocorrelation.lag1` — implemented (cost=tiny). timeseries.autocorrelation.lag1
- ✅ `timeseries.autocorrelation.pacf` — implemented (optional: statsmodels; output=vector; cost=medium). timeseries.autocorrelation.pacf
- ✅ `timeseries.basic.derivatives` — implemented (output=vector; cost=tiny). timeseries.basic.derivatives
- ✅ `timeseries.basic.differences` — implemented (output=vector; cost=tiny). timeseries.basic.differences
- ✅ `timeseries.basic.expanding` — implemented (output=vector). timeseries.basic.expanding
- ✅ `timeseries.basic.lags` — implemented (output=vector; cost=tiny). timeseries.basic.lags
- ✅ `timeseries.basic.rolling` — implemented (output=vector). timeseries.basic.rolling
- ✅ `timeseries.change_point.mean_shift` — implemented. timeseries.change_point.mean_shift
- ✅ `timeseries.change_point.regime_count` — implemented (cost=medium). timeseries.change_point.regime_count
- ✅ `timeseries.change_point.variance_shift` — implemented. timeseries.change_point.variance_shift
- ✅ `timeseries.complexity.approximate_entropy` — implemented (cost=high). timeseries.complexity.approximate_entropy
- ✅ `timeseries.complexity.compressibility` — implemented (cost=medium). timeseries.complexity.compressibility
- ✅ `timeseries.complexity.fuzzy_entropy` — implemented (cost=high). timeseries.complexity.fuzzy_entropy
- ✅ `timeseries.complexity.lempel_ziv` — implemented (cost=medium). timeseries.complexity.lempel_ziv
- ✅ `timeseries.complexity.multiscale_entropy` — implemented (output=vector; cost=high). timeseries.complexity.multiscale_entropy
- ✅ `timeseries.complexity.permutation_entropy` — implemented (cost=high). timeseries.complexity.permutation_entropy
- ✅ `timeseries.complexity.sample_entropy` — implemented (cost=high). timeseries.complexity.sample_entropy
- ✅ `timeseries.complexity.zero_crossing_rate` — implemented. Fraction of adjacent samples that cross zero.
- ✅ `timeseries.cross_series.coherence` — implemented (output=vector; cost=medium). timeseries.cross_series.coherence
- ✅ `timeseries.cross_series.cross_correlation` — implemented (output=vector; cost=medium). timeseries.cross_series.cross_correlation
- ✅ `timeseries.cross_series.dynamic_time_warping` — implemented (cost=high). timeseries.cross_series.dynamic_time_warping
- ✅ `timeseries.cross_series.granger` — implemented (optional: statsmodels; output=vector; cost=high). timeseries.cross_series.granger
- ✅ `timeseries.cross_series.lead_lag` — implemented (output=vector; cost=medium). timeseries.cross_series.lead_lag
- ✅ `timeseries.cross_series.phase_synchrony` — implemented (cost=medium). timeseries.cross_series.phase_synchrony
- ✅ `timeseries.cross_series.transfer_entropy` — implemented (cost=high). timeseries.cross_series.transfer_entropy
- ✅ `timeseries.dynamical.attractor_geometry` — implemented (output=vector; cost=high). timeseries.dynamical.attractor_geometry
- ✅ `timeseries.dynamical.correlation_dimension` — implemented (cost=high). timeseries.dynamical.correlation_dimension
- ✅ `timeseries.dynamical.embedding_dimension` — implemented (cost=high). timeseries.dynamical.embedding_dimension
- ✅ `timeseries.dynamical.lyapunov` — implemented (cost=high). timeseries.dynamical.lyapunov
- ✅ `timeseries.dynamical.phase_space` — implemented (output=vector; cost=medium). timeseries.dynamical.phase_space
- ✅ `timeseries.dynamical.poincare` — implemented (output=vector; cost=medium). timeseries.dynamical.poincare
- ✅ `timeseries.dynamical.time_delay` — implemented (cost=medium). timeseries.dynamical.time_delay
- ✅ `timeseries.fractal.dfa` — implemented (cost=medium). timeseries.fractal.dfa
- ✅ `timeseries.fractal.higuchi` — implemented (cost=medium). timeseries.fractal.higuchi
- ✅ `timeseries.fractal.hurst` — implemented (cost=medium). timeseries.fractal.hurst
- ✅ `timeseries.fractal.katz` — implemented (cost=medium). timeseries.fractal.katz
- ✅ `timeseries.fractal.multifractal` — implemented (output=vector; cost=high). timeseries.fractal.multifractal
- ✅ `timeseries.fractal.petrosian` — implemented (cost=medium). timeseries.fractal.petrosian
- ✅ `timeseries.recurrence.determinism` — implemented (cost=high). timeseries.recurrence.determinism
- ✅ `timeseries.recurrence.laminarity` — implemented (cost=high). timeseries.recurrence.laminarity
- ✅ `timeseries.recurrence.recurrence_network` — implemented (output=vector; cost=high). timeseries.recurrence.recurrence_network
- ✅ `timeseries.recurrence.recurrence_rate` — implemented (cost=high). timeseries.recurrence.recurrence_rate
- ✅ `timeseries.recurrence.trapping_time` — implemented (cost=high). timeseries.recurrence.trapping_time
- ✅ `timeseries.seasonality.harmonic` — implemented (output=vector; cost=medium). timeseries.seasonality.harmonic
- ✅ `timeseries.seasonality.periodicity` — implemented (output=vector; cost=medium). timeseries.seasonality.periodicity
- ✅ `timeseries.seasonality.seasonal_strength` — implemented (cost=medium). timeseries.seasonality.seasonal_strength
- ✅ `timeseries.spectral.bandpower` — implemented (output=vector; cost=medium). timeseries.spectral.bandpower
- ✅ `timeseries.spectral.bispectrum` — implemented (output=vector; cost=high). timeseries.spectral.bispectrum
- ✅ `timeseries.spectral.centroid` — implemented. timeseries.spectral.centroid
- ✅ `timeseries.spectral.cepstrum` — implemented (output=vector; cost=medium). timeseries.spectral.cepstrum
- ✅ `timeseries.spectral.dominant_frequency` — implemented. timeseries.spectral.dominant_frequency
- ✅ `timeseries.spectral.fft` — implemented (output=vector; cost=medium). timeseries.spectral.fft
- ✅ `timeseries.spectral.flatness` — implemented. timeseries.spectral.flatness
- ✅ `timeseries.spectral.higher_order_spectra` — implemented (output=vector; cost=high). timeseries.spectral.higher_order_spectra
- ✅ `timeseries.spectral.periodogram` — implemented (output=vector; cost=medium). timeseries.spectral.periodogram
- ✅ `timeseries.spectral.psd` — implemented (optional: scipy; output=vector; cost=medium). timeseries.spectral.psd
- ✅ `timeseries.spectral.spectral_entropy` — implemented (cost=medium). Normalized Shannon entropy of the one-sided power spectrum.
- ✅ `timeseries.stationarity.adf` — implemented (optional: statsmodels; output=vector; cost=high). timeseries.stationarity.adf
- ✅ `timeseries.stationarity.kpss` — implemented (optional: statsmodels; output=vector; cost=high). timeseries.stationarity.kpss
- ✅ `timeseries.stationarity.variance_ratio` — implemented. timeseries.stationarity.variance_ratio
- ✅ `timeseries.statistical.change_statistics` — implemented (output=vector). timeseries.statistical.change_statistics
- ✅ `timeseries.statistical.crossings` — implemented (output=vector). timeseries.statistical.crossings
- ✅ `timeseries.statistical.energy` — implemented (cost=tiny). timeseries.statistical.energy
- ✅ `timeseries.statistical.extrema` — implemented (output=vector). timeseries.statistical.extrema
- ✅ `timeseries.statistical.kurtosis` — implemented (cost=tiny). timeseries.statistical.kurtosis
- ✅ `timeseries.statistical.max` — implemented (cost=tiny). timeseries.statistical.max
- ✅ `timeseries.statistical.mean` — implemented (cost=tiny). Mean value of each time series.
- ✅ `timeseries.statistical.median` — implemented (cost=tiny). timeseries.statistical.median
- ✅ `timeseries.statistical.min` — implemented (cost=tiny). timeseries.statistical.min
- ✅ `timeseries.statistical.range` — implemented (cost=tiny). timeseries.statistical.range
- ✅ `timeseries.statistical.rms` — implemented (cost=tiny). timeseries.statistical.rms
- ✅ `timeseries.statistical.skewness` — implemented (cost=tiny). timeseries.statistical.skewness
- ✅ `timeseries.statistical.std` — implemented (cost=tiny). Standard deviation of each time series.
- ✅ `timeseries.topology.betti_curves` — implemented (output=vector; cost=high). timeseries.topology.betti_curves
- ✅ `timeseries.topology.persistence` — implemented (output=vector; cost=high). timeseries.topology.persistence
- ✅ `timeseries.topology.persistence_image` — implemented (output=vector; cost=high). timeseries.topology.persistence_image
- ✅ `timeseries.topology.persistence_landscape` — implemented (output=vector; cost=high). timeseries.topology.persistence_landscape
- ✅ `timeseries.trend.linear` — implemented (output=vector). timeseries.trend.linear
- ✅ `timeseries.trend.linear_slope` — implemented. timeseries.trend.linear_slope
- ✅ `timeseries.trend.piecewise` — implemented (output=vector). timeseries.trend.piecewise
- ✅ `timeseries.trend.polynomial` — implemented (output=vector; cost=medium). timeseries.trend.polynomial
- ✅ `timeseries.trend.robust` — implemented (output=vector; cost=medium). timeseries.trend.robust
- ✅ `timeseries.wavelet.cwt` — implemented (output=vector; cost=high). timeseries.wavelet.cwt
- ✅ `timeseries.wavelet.dwt` — implemented (output=vector; cost=medium). timeseries.wavelet.dwt
- ✅ `timeseries.wavelet.scattering` — implemented (output=vector; cost=high). timeseries.wavelet.scattering
- ✅ `timeseries.wavelet.wavelet_entropy` — implemented (cost=medium). timeseries.wavelet.wavelet_entropy
- ✅ `timeseries.wavelet.wavelet_packet` — implemented (output=vector; cost=high). timeseries.wavelet.wavelet_packet

### topology

- ✅ `topology.betti.numbers` — implemented (output=vector; cost=high). topology.betti.numbers
- ✅ `topology.euler.characteristic` — implemented (output=vector; cost=medium). topology.euler.characteristic
- ✅ `topology.local_topology.local_persistence` — implemented (output=vector; cost=extreme). topology.local_topology.local_persistence
- ✅ `topology.mapper.mapper` — implemented (optional: scikit-learn; output=vector; cost=extreme). topology.mapper.mapper
- ✅ `topology.persistence.entropy` — implemented (output=vector; cost=high). topology.persistence.entropy
- ✅ `topology.persistence.image` — implemented (output=vector; cost=high). topology.persistence.image
- ✅ `topology.persistence.landscape` — implemented (output=vector; cost=high). topology.persistence.landscape
- ✅ `topology.persistent_homology.alpha_complex` — implemented (optional: gudhi (optional for H1+); output=vector; cost=high). topology.persistent_homology.alpha_complex
- ✅ `topology.persistent_homology.cech` — implemented (optional: gudhi (optional for H1+); output=vector; cost=high). topology.persistent_homology.cech
- ✅ `topology.persistent_homology.cubical_complex` — implemented (optional: gudhi; output=vector; cost=high). topology.persistent_homology.cubical_complex
- ✅ `topology.persistent_homology.vietoris_rips` — implemented (optional: gudhi (optional for H1+); output=vector; cost=high). topology.persistent_homology.vietoris_rips
- ✅ `topology.persistent_homology.witness_complex` — implemented (optional: gudhi (optional for H1+); output=vector; cost=high). topology.persistent_homology.witness_complex

### video

- ✅ `video.action.statistics` — implemented (output=vector; cost=medium). video.action.statistics
- ✅ `video.camera_motion.global_flow` — implemented (optional: opencv-python; output=vector; cost=high). video.camera_motion.global_flow
- ✅ `video.dynamic_texture.lbp_top` — implemented (optional: scikit-image; output=vector; cost=high). video.dynamic_texture.lbp_top
- ✅ `video.embeddings.3d_cnn` — implemented (optional: user-model; output=vector; cost=extreme). video.embeddings.3d_cnn
- ✅ `video.embeddings.timesformer` — implemented (optional: user-model; output=vector; cost=extreme). video.embeddings.timesformer
- ✅ `video.embeddings.video_mae` — implemented (optional: user-model; output=vector; cost=extreme). video.embeddings.video_mae
- ✅ `video.facial_motion.landmarks` — implemented (optional: user-model; output=vector; cost=extreme). video.facial_motion.landmarks
- ✅ `video.frame_statistics.intensity` — implemented (output=vector; cost=tiny). video.frame_statistics.intensity
- ✅ `video.frame_statistics.mean_intensity` — implemented (cost=tiny). video.frame_statistics.mean_intensity
- ✅ `video.motion.frame_difference_energy` — implemented (cost=tiny). video.motion.frame_difference_energy
- ✅ `video.motion.motion_histograms` — implemented (optional: opencv-python; output=vector; cost=high). video.motion.motion_histograms
- ✅ `video.motion.motion_vectors` — implemented (optional: opencv-python; output=vector; cost=high). video.motion.motion_vectors
- ✅ `video.motion.optical_flow` — implemented (optional: opencv-python; output=vector; cost=high). video.motion.optical_flow
- ✅ `video.motion.trajectories` — implemented (optional: opencv-python; output=vector; cost=high). video.motion.trajectories
- ✅ `video.object_tracks.track_count` — implemented (optional: user-model; output=vector; cost=extreme). video.object_tracks.track_count
- ✅ `video.pose_dynamics.joint_velocity` — implemented (optional: user-model; output=vector; cost=extreme). video.pose_dynamics.joint_velocity
- ✅ `video.scene_change.cuts` — implemented (output=vector). video.scene_change.cuts
- ✅ `video.scene_change.scene_change_max` — implemented. video.scene_change.scene_change_max
- ✅ `video.shot_statistics.duration` — implemented (output=vector). video.shot_statistics.duration
- ✅ `video.temporal_frequency.fft` — implemented (output=vector; cost=medium). video.temporal_frequency.fft
- ✅ `video.temporal_statistics.temporal_std` — implemented (cost=tiny). video.temporal_statistics.temporal_std
- ✅ `video.temporal_statistics.variation` — implemented (output=vector; cost=tiny). video.temporal_statistics.variation
- ✅ `video.track_geometry.velocity` — implemented (optional: opencv-python; output=vector; cost=high). video.track_geometry.velocity

