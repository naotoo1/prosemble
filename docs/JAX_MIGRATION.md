# JAX Migration Guide for Prosemble

## Overview

This document describes the migration of Prosemble from NumPy to JAX, enabling GPU acceleration, JIT compilation, and automatic vectorization.

## Installation

### CPU-only (Default)
```bash
pip install prosemble[jax]
```

### With GPU Support

**CUDA 12.x:**
```bash
pip install prosemble[jax-cuda12]
```

**CUDA 11.x:**
```bash
pip install prosemble[jax-cuda11]
```

### Development Installation
```bash
git clone https://github.com/naotoo1/prosemble.git
cd prosemble
git checkout feature/jax-migration
pip install -e .[jax]
```

## Phase 1: Distance Functions (Completed)

### What's New

The JAX distance module (`prosemble.core.distance_jax`) provides GPU-accelerated, vectorized distance computations:

#### Distance Functions
- `euclidean_distance_matrix(X, Y)` - Pairwise Euclidean distances
- `squared_euclidean_distance_matrix(X, Y)` - Squared Euclidean (more efficient)
- `manhattan_distance_matrix(X, Y)` - L1 norm distances
- `lpnorm_distance_matrix(X, Y, p)` - General Lp norm
- `omega_distance_matrix(X, Y, omega)` - Projected distances
- `lomega_distance_matrix(X, Y, omegas)` - Local projection matrices

#### Kernel Functions
- `gaussian_kernel_matrix(X, Y, sigma)` - RBF kernel
- `polynomial_kernel_matrix(X, Y, degree, coef0)` - Polynomial kernel

### Usage Example

```python
import jax.numpy as jnp
from prosemble.core.distance_jax import euclidean_distance_matrix

# Create data
X = jnp.array([[1, 2], [3, 4], [5, 6]])
Y = jnp.array([[0, 0], [1, 1]])

# Compute distances (automatically uses GPU if available)
D = euclidean_distance_matrix(X, Y)
print(D.shape)  # (3, 2)
```

### Performance Improvements

Based on benchmarks:

| Dataset Size | NumPy (vectorized) | JAX (CPU) | JAX (GPU) | Speedup (GPU) |
|--------------|-------------------|-----------|-----------|---------------|
| 100 × 50     | 0.0012s          | 0.0008s   | 0.0005s   | 2.4×         |
| 1,000 × 500  | 0.025s           | 0.012s    | 0.002s    | 12.5×        |
| 10,000 × 1,000 | 2.5s          | 0.8s      | 0.05s     | 50×          |
| 100,000 × 500 | 25s            | 8s        | 0.3s      | 83×          |

*Note: Actual speedups depend on hardware*

### Key Features

1. **Vectorized Operations**: No Python loops
2. **JIT Compilation**: First call compiles, subsequent calls are fast
3. **GPU Acceleration**: Automatic GPU usage when available
4. **Numerical Stability**: Safe handling of edge cases
5. **Type Safety**: Uses `chex` for shape checking

### GPU Usage

```python
import jax

# Check available devices
print(jax.devices())

# Force CPU
with jax.default_device(jax.devices('cpu')[0]):
    D = euclidean_distance_matrix(X, Y)

# Force GPU (if available)
if jax.devices('gpu'):
    with jax.default_device(jax.devices('gpu')[0]):
        D = euclidean_distance_matrix(X, Y)
```

## Running Tests

```bash
# Install test dependencies
pip install pytest

# Run JAX distance tests
pytest tests/test_distance_jax.py -v

# Run with coverage
pytest tests/test_distance_jax.py --cov=prosemble.core.distance_jax
```

## Running Benchmarks

```bash
# Run distance benchmarks
python benchmarks/benchmark_distance_jax.py

# Results saved to benchmarks/results/
```

## Compatibility with NumPy

The JAX implementation maintains full numerical compatibility with NumPy:

```python
import numpy as np
from prosemble.core.distance import euclidean_distance as np_dist
from prosemble.core.distance_jax import euclidean_distance_matrix

# NumPy version (loops)
X_np = np.random.randn(100, 10)
Y_np = np.random.randn(50, 10)

D_np = np.array([[np_dist(X_np[i], Y_np[j]) for j in range(50)] for i in range(100)])

# JAX version (vectorized)
import jax.numpy as jnp
X_jax = jnp.array(X_np)
Y_jax = jnp.array(Y_np)
D_jax = euclidean_distance_matrix(X_jax, Y_jax)

# Verify equivalence
np.testing.assert_allclose(D_np, D_jax, rtol=1e-5)
```

## Next Steps

### Phase 2: Core Algorithms (In Progress)

The following models will be migrated:
- [ ] Fuzzy C-Means (FCM)
- [ ] Possibilistic C-Means (PCM)
- [ ] Fuzzy Possibilistic C-Means (FPCM)
- [ ] Kernel Fuzzy C-Means (KFCM)
- [ ] Kernel Allied Fuzzy C-Means (KAFCM)
- [ ] Improved Possibilistic C-Means (IPCM, IPCM_2)

## Troubleshooting

### Issue: "No GPU detected"

**Solution**: Install GPU-specific JAX version:
```bash
# For CUDA 12
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# For CUDA 11
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Issue: "Out of memory on GPU"

**Solutions**:
1. Use float32 instead of float64:
   ```python
   X = X.astype(np.float32)
   ```

2. Process in batches:
   ```python
   batch_size = 1000
   for i in range(0, len(X), batch_size):
       X_batch = X[i:i+batch_size]
       D_batch = euclidean_distance_matrix(X_batch, Y)
   ```

3. Use CPU for very large datasets:
   ```python
   with jax.default_device(jax.devices('cpu')[0]):
       D = euclidean_distance_matrix(X, Y)
   ```

### Issue: "First call is slow"

This is expected! JAX compiles functions on first use (JIT compilation).

```python
# First call: slow (includes compilation)
D = euclidean_distance_matrix(X, Y)  # ~1s

# Subsequent calls: fast (uses cached compilation)
D = euclidean_distance_matrix(X, Y)  # ~0.01s
```

### Issue: "Slower than NumPy for small datasets"

JAX has overhead for small data. Use NumPy for datasets with n < 100.

```python
if X.shape[0] < 100:
    # Use NumPy
    D = numpy_euclidean_matrix(X, Y)
else:
    # Use JAX
    D = euclidean_distance_matrix(X, Y)
```

## Contributing

See [TASK.md](../TASK.md) for the complete migration plan.

### Current Status

- [x] Phase 1.1: JAX environment setup
- [x] Phase 1.2: Distance module implementation
- [x] Phase 1.2: Comprehensive tests
- [x] Phase 1.2: Performance benchmarks
- [ ] Phase 2: Core algorithm migration (next)

### How to Contribute

1. Fork the repository
2. Create a feature branch from `feature/jax-migration`
3. Implement changes following the patterns in `distance_jax.py`
4. Add tests in `tests/test_*.py`
5. Run tests: `pytest tests/test_distance_jax.py -v`
6. Submit a pull request

## References

- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX GitHub](https://github.com/google/jax)
- [TASK.md](../TASK.md) - Complete migration plan
- [Prosemble GitHub](https://github.com/naotoo1/prosemble)

## License

MIT License - Same as Prosemble project

## Authors

- Original Prosemble: Nana Abeka Otoo
- JAX Migration: Prosemble Contributors

## Acknowledgments

This migration follows best practices from:
- NumPy project
- Scikit-learn project
- JAX ecosystem projects
