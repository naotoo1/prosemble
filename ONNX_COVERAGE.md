# ONNX Export Coverage

Status of ONNX export support across all 87 prosemble models.

## Currently Supported (73 models)

### Supervised LVQ — squared euclidean (9)

GLVQ, GLVQ1, GLVQ21, LVQ1, LVQ21, MedianLVQ, CELVQ, SLVQ, RSLVQ

### Supervised LVQ — global omega (2)

GMLVQ, MRSLVQ

### Supervised LVQ — local omega (2)

LGMLVQ, LMRSLVQ

### Supervised LVQ — relevance-weighted (1)

GRLVQ

### Supervised LVQ — tangent (1)

GTLVQ

### Supervised NG — squared euclidean (3)

SRNG, CELVQ_NG, RSLVQ_NG

### Supervised NG — global omega (3)

SMNG, MCELVQ_NG, MRSLVQ_NG

### Supervised NG — local omega (3)

SLNG, LCELVQ_NG, LMRSLVQ_NG

### Supervised NG — tangent (2)

STNG, TCELVQ_NG

### Unsupervised — squared euclidean (4)

NeuralGas, GrowingNeuralGas, KohonenSOM, HeskesSOM

### Fuzzy clustering — squared euclidean (8)

FCM, PCM, FPCM, PFCM, AFCM, HCM, IPCM, IPCM2

For FCM and variants with fuzzifier m > 1, membership u_ij is monotonically decreasing in distance d_ij, so `argmax(membership) = argmin(distance)`. The ONNX graph uses `argmin(distance)` which produces identical predictions.

### One-class — hard nearest decision (10)

OCGLVQ, OCGLVQ_NG, OCGRLVQ, OCGRLVQ_NG, OCGMLVQ, OCGMLVQ_NG, OCLGMLVQ, OCLGMLVQ_NG, OCGTLVQ, OCGTLVQ_NG

Decision: `argmin(d) → gather(d, theta) → mu = (d-theta)/(d+theta) → 1 - sigmoid(beta*mu) → threshold 0.5`. Distance varies per model (squared euclidean, relevance-weighted, global omega, local omega, tangent).

### One-class — Gaussian soft decision (3)

OCRSLVQ, OCMRSLVQ, OCLMRSLVQ

Decision: softmax Gaussian weights over all prototypes, weighted mu aggregation, sigmoid threshold.

### One-class — Gaussian+NG soft decision (3)

OCRSLVQ_NG, OCMRSLVQ_NG, OCLMRSLVQ_NG

Decision: combined Gaussian × NG rank weights (using converged gamma), weighted mu aggregation. NG ranks computed via TopK + ScatterElements.

### SVQ-OCC — response model (5)

SVQOCC, SVQOCC_R, SVQOCC_M, SVQOCC_LM, SVQOCC_T

Decision: response probability (Gaussian softmax, Student-t, or uniform) × differentiable Heaviside sigmoid → summed responsibility → threshold 0.5.

### Encoder — MLP backbone + WTAC (4)

SiameseGLVQ, SiameseGMLVQ, SiameseGTLVQ, LVQMLN

Input encoded through MLP layers (MatMul → Add → Activation per layer). Prototypes pre-computed in latent space at export time (Siamese: re-encoded; LVQMLN: already latent). Distance varies per model (squared euclidean, global omega, tangent).

### Encoder — CNN backbone + WTAC (3)

ImageGLVQ, ImageGMLVQ, ImageGTLVQ

Input reshaped to NHWC, transposed to NCHW, then Conv layers (SAME padding) → GlobalAveragePool → Linear head. Prototypes pre-computed in latent space at export time. Distance varies per model (squared euclidean, global omega, tangent).

### Encoder — PLVQ Gaussian mixture (1)

PLVQ

MLP encoder → squared Euclidean distance → Softmax soft assignment → class aggregation via MatMul with binary class mask → ArgMax.

### CBC — reasoning matrices (2)

CBC, ImageCBC

Squared Euclidean distance → Gaussian similarity `exp(-d²/(2σ²))` → CBCC reasoning: `probs = (detections @ (pk - nk) + sum(nk)) / (sum(pk + nk) + eps)` → ArgMax. ImageCBC adds a CNN encoder before distance computation, with components pre-computed in latent space.

### Riemannian — SO(n) chordal distance (1)

RiemannianSRNG (with SO(n) manifold)

Chordal distance: `d²(R,S) = ||R - S||²_F`. Pure Frobenius norm on flattened rotation matrices.

### Riemannian — SO(n) tangent-space metric (3)

RiemannianSMNG, RiemannianSLNG, RiemannianSTNG (with SO(n) manifold)

Log map: `Log_R(S) = R @ skew(R^T S)` where `skew(A) = (A - A^T)/2`. Then metric adaptation (global omega, local omega, or tangent subspace) applied to flattened tangent vectors. All ops are MatMul/Sub/Transpose.

### Riemannian — Grassmannian tangent-space metric (same 3 models, alternate manifold)

RiemannianSMNG, RiemannianSLNG, RiemannianSTNG (with Grassmannian(n,k) manifold)

Log map: `Log_{Q1}(Q2) = Q2 - Q1(Q1^T Q2)`. Then metric adaptation applied to flattened tangent vectors. All ops are MatMul/Sub.

---

## Not Supported (14 models)

### Kernel fuzzy clustering (7) — kernel distance not exportable

KFCM, KPCM, KFPCM, KPFCM, KAFCM, KIPCM, KIPCM2

**Blocker:** Predict uses Gaussian kernel distance (1 - K) computed via `batch_gaussian_kernel()`. Kernel evaluation requires the `sigma` parameter and a different distance computation that has no standard ONNX equivalent.

### Riemannian models — non-exportable (1)

- **RiemannianNeuralGas** — uses `jsl.funm` (matrix logarithm via Schur decomposition, no ONNX op)

Note: The 4 supervised Riemannian models (RiemannianSRNG/SMNG/SLNG/STNG) are exportable on SO(n) and Grassmannian manifolds. SPD(n) manifold is not exportable (eigendecomposition required) but is a runtime configuration choice, not a separate model. RiemannianSRNG on Grassmannian is also not exportable (SVD required) but again is the same model with a different manifold choice.

### Other (2)

- **KNN** — k-nearest-neighbor logic, not prototype-based
- **NPC** — nearest prototype classifier with different predict pattern

### Utility models not applicable (4)

KMeansPlusPlus, Kmeans, SOM, BGPC — don't inherit from prototype model base classes.

---

## Distance Functions Supported

| Distance | ONNX Implementation | Models |
|----------|-------------------|--------|
| Squared Euclidean | expansion trick | GLVQ family, NG, NeuralGas, FCM, OCGLVQ, OCRSLVQ, SVQOCC, Siamese/Image/LVQMLN/PLVQ, CBC |
| Euclidean | sqrt of above | (available) |
| Manhattan | broadcast + abs + sum | (available) |
| Global Omega | project then sq. euclidean | GMLVQ, MRSLVQ, SMNG, OCGMLVQ, OCMRSLVQ, SVQOCC_M, SiameseGMLVQ, ImageGMLVQ |
| Relevance-weighted | element-wise weighted sq. diff | GRLVQ, OCGRLVQ, SVQOCC_R |
| Local Omega | batched MatMul | LGMLVQ, SLNG, LCELVQ_NG, LMRSLVQ, OCLGMLVQ, SVQOCC_LM |
| Tangent | batched MatMul project-reconstruct | GTLVQ, STNG, TCELVQ_NG, OCGTLVQ, SVQOCC_T, SiameseGTLVQ, ImageGTLVQ |
| SO(n) Chordal | broadcast subtract + Frobenius | RiemannianSRNG |
| SO(n) Tangent | skew-symmetric log map + metric | RiemannianSMNG/SLNG/STNG (SO) |
| Grassmannian Tangent | projection log map + metric | RiemannianSMNG/SLNG/STNG (Gr) |

## Encoder Support

| Encoder | ONNX Implementation | Models |
|---------|-------------------|--------|
| MLP | MatMul → Add → Activation per layer | SiameseGLVQ, SiameseGMLVQ, SiameseGTLVQ, LVQMLN, PLVQ |
| CNN | Reshape → Transpose → Conv(SAME) → GlobalAveragePool → Linear | ImageGLVQ, ImageGMLVQ, ImageGTLVQ, ImageCBC |

Activations supported: Sigmoid, Relu, Tanh, LeakyRelu, Selu.
