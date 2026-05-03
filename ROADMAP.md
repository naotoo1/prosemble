# Prosemble Roadmap

## v1.0 (Current)

Released with:
- 38 models (23 supervised, 11 fuzzy clustering, 4 topology)
- JIT-compiled training (`lax.scan`) and inference
- Mixed precision training (float16/bfloat16) for supervised models
- Mini-batch training with early stopping and patience
- Callbacks, visualization, save/load, quantization (float16/bfloat16/int8)
- `get_params()`/`set_params()` sklearn estimator protocol
- Resume-from-checkpoint with full optimizer state preservation
- `partial_fit()` for incremental/online learning
- `jax.export` for deployment

## v1.1 — Post-Launch Polish

- `score(X, y)` — sklearn-style evaluation method
- Mixed precision for clustering models (FCM, PCM, etc.)
- Advanced examples showcasing all features
- `cross_val_score` wrapper around `k_fold_split_jax`

## v2.0 — Multi-Device & Scale

- **`jax.sharding` / `shard_map`** — multi-GPU/TPU parallel training
  - Primary use case: Image variants (ImageGLVQ, ImageGMLVQ, ImageGTLVQ, ImageCBC) with trainable backbones on large datasets
  - Use modern `jax.sharding` API (not deprecated `pmap`)
  - Data parallelism via `NamedSharding` + `PartitionSpec`
- **`jax.remat` (gradient checkpointing)** — for deep backbone training
  - Only needed when Image/Siamese backbones are unfrozen
  - One-line `@jax.checkpoint` on `_compute_loss`
- **`custom_vjp`** — if new models require non-standard gradient rules
  - e.g., optimal transport distances, hard thresholding operations

## Future Considerations

- One-class / novelty detection models
  - Nearest Prototype Classifier with rejection (per-prototype distance thresholds)
  - One-Class SVM (hyperplane separating normal data from origin)
  - SVDD (smallest hypersphere enclosing normal data)
  - Leverages existing prototype + distance infrastructure
- Publication on novel neighborhood cooperative LVQ variants (SMNG, SLNG, STNG)
  - SMNG: Supervised Matrix Neural Gas (GMLVQ + neighborhood cooperation)
  - SLNG: Supervised Localized Matrix Neural Gas (LGMLVQ + neighborhood cooperation)
  - STNG: Supervised Tangent Neural Gas (GTLVQ + neighborhood cooperation)
  - Extends Hammer et al. (2003) SRNG framework to full matrix, local matrix, and tangent metrics
  - Experimental evaluation against base models (GMLVQ, LGMLVQ, GTLVQ) on benchmark datasets
- Publication on metric-adaptive one-class classification extensions (SVQ-OCC-R and beyond)
  - SVQ-OCC-R: SVQ-OCC + per-feature relevance weighting (GRLVQ-style metric adaptation)
  - Reject option integration for safety-critical applications
  - Extends Staps et al. (2022) SVQ-OCC with learned discriminative distances
  - Potential further variants: SVQ-OCC-M (global matrix), SVQ-OCC-LM (local matrix), SVQ-OCC-T (tangent)
  - Experimental evaluation against SVDD, Isolation Forest, and base SVQ-OCC on benchmark datasets
- Supervised Riemannian Neural Gas variants (R-SRNG, R-SMNG, R-SLNG, R-STNG)
  - Extends Riemannian Neural Gas (unsupervised) to supervised classification on manifolds
  - Combines geodesic distance + Exp/Log map updates with GLVQ-style cost functions
  - Requires validation of optax backpropagation through matrix exponential/logarithm chains
  - Application domains: EEG/BCI classification (SPD), robot grasp classification (SO(3)), hyperspectral image classification (Grassmannian)
  - Would be a novel contribution — no published supervised Riemannian NG exists
- Publication on Riemannian Neural Gas implementation and extensions
  - Implements Schwarz, Psenickova, Villmann, Röhrbein (ESANN 2026) with SO(n), SPD(n), Gr(n,k) manifolds
  - JAX-based with pluggable manifold abstraction and injectivity radius safety bounds
  - Potential extension to supervised variants as above
- Protocol types for duck-typed contracts (typing.Protocol)
- Save/load logic consolidation across base classes
- Data loading pipeline (JAX-native data iterators)
- ONNX export alongside `jax.export`
