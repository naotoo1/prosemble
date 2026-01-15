# Migration from NumPy to JAX: Comprehensive Task Plan

**Project**: Prosemble - Prototype-Based Machine Learning Library
**Date**: January 15, 2026
**Objective**: Migrate from NumPy to JAX with vectorization, parallelization, JIT compilation, and GPU support

---

## Executive Summary

This document outlines a comprehensive, phased approach to migrate Prosemble from NumPy to JAX. Based on thorough analysis of the codebase, this migration will enable:

- **Hardware Acceleration**: GPU/TPU support for faster training
- **Just-In-Time (JIT) Compilation**: Significant performance improvements
- **Automatic Vectorization**: Better utilization of SIMD instructions
- **Automatic Differentiation**: Future ML enhancements
- **Parallelization**: Multi-device computation

**Codebase Analyzed**:
- Core distance functions (6 distance metrics)
- 21 clustering/classification models
- Primary focus: FCM, PCM, FPCM, KFCM, KAFCM families

---

## Current Implementation Analysis

### 1. Core Distance Module (`prosemble/core/distance.py`)

**Current Implementation**:
```python
# Simple NumPy-based distances
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def squared_euclidean_distance(x, y):
    return euclidean_distance(x, y) ** 2

def manhattan_distance(point1, point2):
    # Uses Python loops - NOT vectorized!
    sum_ = 0
    for x1, x2 in zip(point1, point2):
        difference = x2 - x1
        absolute_difference = abs(difference)
        sum_ += absolute_difference
    return sum_

def lpnorm_distance(x, y, p):
    return np.linalg.norm((x - y), ord=p)

def omega_distance(x, y, omega):
    # Matrix projection-based distance
    projected_x = x @ omega
    projected_y = y @ omega
    return squared_euclidean_distance(projected_x, projected_y)

def lomega_distance(x, y, omegas):
    # Multi-projection distance with broadcasting
    projected_x = x @ omegas
    projected_y = (np.array(y @ omegas).diagonal()).T
    expanded_y = np.expand_dims(projected_y, axis=1)
    differences_squared = (expanded_y - projected_x)**2
    distances = np.sum(differences_squared, axis=2)
    return distances.transpose(1, 0)
```

**Issues Identified**:
1. ❌ `manhattan_distance` uses Python loops (slow!)
2. ❌ `squared_euclidean_distance` computes sqrt then squares it (wasteful)
3. ❌ No batch processing - processes one pair at a time
4. ❌ `lomega_distance` uses `.diagonal().T` which is inefficient
5. ✅ Some vectorization present but not optimal

---

### 2. Fuzzy C-Means (FCM) - Representative Algorithm

**Algorithm Structure**:
```
Initialize fuzzy membership matrix U (n_samples × n_clusters)
Repeat until convergence:
    1. Compute centroids from U matrix
    2. Update U matrix based on distances to centroids
    3. Compute objective function
    4. Check convergence
```

**Current Implementation Issues**:

#### A. Centroid Computation (`compute_centroids`)
```python
# Current: Uses list comprehensions with nested loops
fuzzified_assignments = [
    np.power([u_ik[i] for _, u_ik in enumerate(fuzzy_matrix)], self.fuzzifier)
    for i in range(self.num_clusters)
]
sum_fuzzified_assigments = [np.sum(i) for i in fuzzified_assignments]

centroid_numerator = [
    [np.multiply(fuzzified_assignments[cluster_index][index], sample) for
     index, sample in enumerate(self.data)]
    for cluster_index in range(self.num_clusters)
]

centroid = np.array([
    np.sum(v, axis=0) / sum_fuzzified_assigments[i]
    for i, v in enumerate(centroid_numerator)
])
```

**Problems**:
- ❌ Triple nested loops (clusters × samples × features)
- ❌ Creates intermediate lists
- ❌ Manual indexing prevents JIT optimization
- ❌ Not vectorized - processes one element at a time

**Mathematical Formulation**:
```
v_j = Σ(u_ij^m * x_i) / Σ(u_ij^m)
```

**Should be**: Single matrix operation: `V = (U^m)^T @ X / sum((U^m)^T)`

#### B. Fuzzy Matrix Update (`update_fuzzy_matrix`)
```python
# Current: Nested loops over samples and clusters
initial_u_matrix = u_matrix
for i in range(len(self.data)):  # For each sample
    denomenator = 0
    for j in range(self.num_clusters):  # For each cluster
        denomenator += np.power(
            1 / euclidean_distance(centroids[j], self.data[i]),
            2 / (self.fuzzifier - 1)
        )
    for j in range(self.num_clusters):  # Second pass
        uik_new = np.power(
            1 / euclidean_distance(centroids[j], self.data[i]),
            2 / (self.fuzzifier - 1)
        ) / denomenator
        initial_u_matrix[i][j] = uik_new
return initial_u_matrix
```

**Problems**:
- ❌ Double nested loop: O(n_samples × n_clusters)
- ❌ Distance computed twice for same pairs
- ❌ In-place updates prevent parallelization
- ❌ Calls `euclidean_distance` in loop (function call overhead)

**Mathematical Formulation**:
```
u_ij = 1 / Σ_k (d_ij / d_kj)^(2/(m-1))
```

**Should be**: Batch distance computation then vectorized division

#### C. Objective Function (`compute_objective_function`)
```python
# Current: Nested list comprehensions
objective_function = np.sum(
    [[squared_euclidean_distance(self.data[i], centroids[j]) *
      np.power(u_matrix[i][j], self.fuzzifier)
      for i in range(len(self.data))]
     for j in range(self.num_clusters)]
)
```

**Problems**:
- ❌ Nested loops
- ❌ Recomputes distances already computed in update step
- ❌ Not cacheable

**Mathematical Formulation**:
```
J = Σ_i Σ_j u_ij^m * ||x_i - v_j||^2
```

**Should be**: `J = sum(U^m * D^2)` where D is precomputed distance matrix

---

### 3. Possibilistic C-Means (PCM)

**Additional Complexity**:
- Requires FCM initialization
- Computes gamma (scale) parameters
- Has typicality matrix T in addition to U matrix

**Current Issues**:
```python
def compute_gamma(self, fuzzy_matrix, centroids):
    # Same nested loop pattern as FCM
    fuzzified_assignments = [...]
    centroid_numerator = [
        [np.multiply(fuzzified_assignments[cluster_index][index],
                     squared_euclidean_distance(sample, centroids[cluster_index]))
         for index, sample in enumerate(self.data)]
        for cluster_index in range(self.num_clusters)
    ]
    gamma = np.array([
        np.sum(v, axis=0) * self.k / sum_fuzzified_assigments[i]
        for i, v in enumerate(centroid_numerator)
    ])
    return gamma
```

**Same vectorization issues as FCM**

---

### 4. Kernel-Based Methods (KFCM, KAFCM, etc.)

**Current Implementation**:
```python
def gaussian_kernel(self, x, y):
    return np.exp(-squared_euclidean_distance(x, y) / (self.sigma ** 2))

def compute_centroids(self, fuzzy_matrix, centroids):
    fuzzified_assignments = [...]

    # Kernel-weighted centroid computation
    centroid_numerator = [
        [np.multiply(fuzzified_assignments[cluster_index][index], sample) *
         self.gaussian_kernel(sample, centroids[cluster_index])
         for index, sample in enumerate(self.data)]
        for cluster_index in range(self.num_clusters)
    ]

    centroid_denomenator = [
        [np.multiply(fuzzified_assignments[cluster_index][index],
                     self.gaussian_kernel(sample, centroids[cluster_index]))
         for index, sample in enumerate(self.data)]
        for cluster_index in range(self.num_clusters)
    ]
    # ...
```

**Additional Issues**:
- ❌ Kernel computed pairwise in loops
- ❌ Should use kernel matrix: K[i,j] = k(x_i, v_j)
- ❌ Kernel trick not utilized for feature space operations
- ❌ Memory inefficient for large datasets

**Mathematical Note**:
Kernel methods map data to high-dimensional space: φ(x)
```
Instead of: ||φ(x_i) - φ(v_j)||^2
Use kernel trick: k(x_i, x_i) - 2k(x_i, v_j) + k(v_j, v_j)
```

---

## Migration Strategy

### Phase 1: Foundation (Weeks 1-2)

#### 1.1 Setup JAX Environment

**Tasks**:
- [ ] Add JAX dependencies to `pyproject.toml`
  ```toml
  dependencies = [
      "jax[cpu]>=0.4.23",
      "jaxlib>=0.4.23",
      # Optional GPU support
      # "jax[cuda12]>=0.4.23",  # For CUDA 12
      # "jax[cuda11]>=0.4.23",  # For CUDA 11
  ]
  ```
- [ ] Update `devenv.nix` to include JAX packages
- [ ] Create JAX compatibility test suite
- [ ] Document GPU/TPU setup requirements

**Deliverables**:
- Updated dependency files
- Installation instructions for CPU/GPU/TPU
- Basic JAX import tests

#### 1.2 Create JAX Distance Module

**File**: `prosemble/core/distance_jax.py`

**Implementation Plan**:

```python
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

# Vectorized pairwise distance computation
@jit
def euclidean_distance_matrix(X, Y):
    """
    Compute pairwise Euclidean distances between rows of X and Y.

    Args:
        X: (n, d) array
        Y: (m, d) array

    Returns:
        D: (n, m) distance matrix where D[i,j] = ||X[i] - Y[j]||

    Mathematical Formula:
        D[i,j] = sqrt(Σ_k (X[i,k] - Y[j,k])^2)

    Optimized Implementation:
        D^2 = ||X||^2 + ||Y||^2 - 2*X@Y^T
        This avoids explicit broadcasting and is O(nmd) instead of O(nmd^2)
    """
    X_sq = jnp.sum(X**2, axis=1, keepdims=True)  # (n, 1)
    Y_sq = jnp.sum(Y**2, axis=1, keepdims=True).T  # (1, m)
    XY = X @ Y.T  # (n, m)
    D_sq = X_sq + Y_sq - 2 * XY
    D_sq = jnp.maximum(D_sq, 0)  # Numerical stability
    return jnp.sqrt(D_sq)

@jit
def squared_euclidean_distance_matrix(X, Y):
    """Squared Euclidean distance - more efficient than euclidean^2"""
    X_sq = jnp.sum(X**2, axis=1, keepdims=True)
    Y_sq = jnp.sum(Y**2, axis=1, keepdims=True).T
    XY = X @ Y.T
    D_sq = X_sq + Y_sq - 2 * XY
    return jnp.maximum(D_sq, 0)

@jit
def manhattan_distance_matrix(X, Y):
    """
    Vectorized Manhattan distance.

    Uses: ||X - Y||_1 = Σ|X - Y|
    Implementation: vmap over rows to compute all pairwise distances
    """
    def single_manhattan(x, Y):
        return jnp.sum(jnp.abs(x - Y), axis=1)

    # Vectorize over rows of X
    return vmap(single_manhattan, in_axes=(0, None))(X, Y)

@jit
def lpnorm_distance_matrix(X, Y, p):
    """
    Generalized L-p norm distance matrix.

    Formula: ||X - Y||_p = (Σ|X - Y|^p)^(1/p)
    """
    def single_lpnorm(x, Y):
        return jnp.power(jnp.sum(jnp.power(jnp.abs(x - Y), p), axis=1), 1/p)

    return vmap(single_lpnorm, in_axes=(0, None))(X, Y)

@jit
def omega_distance_matrix(X, Y, omega):
    """
    Projected distance with projection matrix omega.

    Formula: d(x, y) = ||x@Ω - y@Ω||^2
    """
    X_proj = X @ omega  # (n, k) where k is projection dim
    Y_proj = Y @ omega  # (m, k)
    return squared_euclidean_distance_matrix(X_proj, Y_proj)

@jit
def lomega_distance_matrix(X, Y, omegas):
    """
    Multi-projection lomega distance.

    Args:
        X: (n, d) data matrix
        Y: (m, d) centroid matrix
        omegas: (d, d, p) projection matrices (p projections)

    Returns:
        D: (n, m) distance matrix

    Formula: Sum distances across all projections
    """
    # X @ omegas: (n, d) @ (d, d, p) -> need to handle 3D
    # Use einsum for clarity
    X_proj = jnp.einsum('nd,ddp->ndp', X, omegas)  # (n, d, p)
    Y_proj = jnp.einsum('md,ddp->mdp', Y, omegas)  # (m, d, p)

    # Compute squared differences and sum over feature and projection dims
    # Broadcasting: (n, 1, d, p) - (1, m, d, p) -> (n, m, d, p)
    diff_sq = (X_proj[:, None, :, :] - Y_proj[None, :, :, :]) ** 2

    # Sum over features and projections
    distances = jnp.sum(diff_sq, axis=(2, 3))
    return distances

@jit
def gaussian_kernel_matrix(X, Y, sigma):
    """
    Gaussian (RBF) kernel matrix.

    Formula: K[i,j] = exp(-||X[i] - Y[j]||^2 / (2σ^2))

    Returns:
        K: (n, m) kernel matrix
    """
    D_sq = squared_euclidean_distance_matrix(X, Y)
    return jnp.exp(-D_sq / (2 * sigma**2))
```

**Testing Requirements**:
- [ ] Unit tests comparing JAX vs NumPy outputs
- [ ] Numerical precision tests (tolerance: 1e-6)
- [ ] Performance benchmarks (CPU vs GPU)
- [ ] Edge case tests (empty arrays, single points, high dimensions)

**Performance Targets**:
- 10-100× speedup on large datasets (n > 10,000)
- GPU speedup: 50-1000× for matrix sizes > 1000×1000

---

### Phase 2: Core Algorithm Migration (Weeks 3-6)

#### 2.1 Fuzzy C-Means (FCM) - JAX Implementation

**File**: `prosemble/models/fcm_jax.py`

**Architecture**:
```python
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from typing import NamedTuple, Tuple
import chex  # For shape checking

class FCMState(NamedTuple):
    """Immutable state for FCM algorithm (required for JAX)"""
    centroids: chex.Array  # (n_clusters, n_features)
    U: chex.Array  # (n_samples, n_clusters) fuzzy membership
    objective: chex.Array  # (,) scalar
    iteration: int

class FCM_JAX:
    """
    JAX implementation of Fuzzy C-Means clustering.

    Key Differences from NumPy version:
    1. All operations vectorized (no Python loops)
    2. Immutable state (functional programming)
    3. JIT-compiled methods
    4. GPU-compatible
    5. Automatic differentiation support

    Mathematical Algorithm:
    -------------------------
    Given data X (n × d), find:
    - Centroids V (c × d)
    - Fuzzy membership U (n × c) where Σ_j u_ij = 1

    Objective Function:
        J = Σ_i Σ_j u_ij^m * ||x_i - v_j||^2

    Update Rules:
        v_j = Σ_i (u_ij^m * x_i) / Σ_i u_ij^m
        u_ij = 1 / Σ_k (||x_i - v_j|| / ||x_i - v_k||)^(2/(m-1))

    Convergence:
        ||V_new - V_old|| < epsilon
    """

    def __init__(
        self,
        n_clusters: int,
        fuzzifier: float = 2.0,
        max_iter: int = 100,
        epsilon: float = 1e-5,
        init_method: str = 'random',
        random_seed: int = 42
    ):
        self.n_clusters = n_clusters
        self.fuzzifier = fuzzifier
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.init_method = init_method
        self.key = jax.random.PRNGKey(random_seed)

    @partial(jit, static_argnums=(0,))
    def _initialize_U(self, X: chex.Array, key: chex.PRNGKey) -> chex.Array:
        """
        Initialize fuzzy membership matrix U.

        Strategy: Use Dirichlet distribution to ensure Σ_j u_ij = 1

        Args:
            X: (n, d) data matrix
            key: JAX random key

        Returns:
            U: (n, c) fuzzy membership matrix
        """
        n_samples = X.shape[0]
        # Dirichlet ensures row sums = 1
        U = jax.random.dirichlet(
            key,
            alpha=jnp.ones(self.n_clusters),
            shape=(n_samples,)
        )
        return U

    @partial(jit, static_argnums=(0,))
    def _compute_centroids(self, X: chex.Array, U: chex.Array) -> chex.Array:
        """
        Compute cluster centroids from fuzzy membership matrix.

        Mathematical Formula:
            v_j = Σ_i (u_ij^m * x_i) / Σ_i u_ij^m

        Vectorized Implementation:
            V = (U^m)^T @ X / sum((U^m)^T, axis=1, keepdims=True)

        Args:
            X: (n, d) data matrix
            U: (n, c) fuzzy membership matrix

        Returns:
            V: (c, d) centroid matrix

        Complexity: O(ncd) - single matrix multiply
        Old complexity: O(ncd) but with loop overhead
        """
        # Fuzzify membership: U^m
        U_fuzz = jnp.power(U, self.fuzzifier)  # (n, c)

        # Numerator: (c, n) @ (n, d) = (c, d)
        numerator = U_fuzz.T @ X

        # Denominator: (c, n) @ (n, 1) = (c, 1)
        denominator = jnp.sum(U_fuzz, axis=0, keepdims=True).T

        # Avoid division by zero
        denominator = jnp.maximum(denominator, 1e-10)

        centroids = numerator / denominator
        return centroids

    @partial(jit, static_argnums=(0,))
    def _update_U(
        self,
        X: chex.Array,
        centroids: chex.Array
    ) -> chex.Array:
        """
        Update fuzzy membership matrix.

        Mathematical Formula:
            u_ij = 1 / Σ_k (d_ij / d_ik)^(2/(m-1))
            where d_ij = ||x_i - v_j||

        Vectorized Implementation:
            1. Compute distance matrix D: (n, c)
            2. Compute ratio matrix: (d_ij / d_ik) for all i, j, k
            3. Apply power and sum

        Args:
            X: (n, d) data matrix
            centroids: (c, d) centroid matrix

        Returns:
            U: (n, c) updated fuzzy membership matrix

        Complexity: O(ncd) for distance + O(nc^2) for membership
        Old complexity: O(nc^2d) with nested loops
        """
        # Compute pairwise distances: (n, c)
        D = euclidean_distance_matrix(X, centroids)

        # Add small epsilon to avoid division by zero
        D = jnp.maximum(D, 1e-10)

        # Compute power for formula
        power = 2.0 / (self.fuzzifier - 1)

        # For each i, j: sum over k of (d_ij / d_ik)^power
        # Reshape for broadcasting: (n, c, 1) / (n, 1, c) = (n, c, c)
        ratios = (D[:, :, None] / D[:, None, :]) ** power

        # Sum over k dimension: (n, c, c) -> (n, c)
        denominators = jnp.sum(ratios, axis=2)

        # U_ij = 1 / denominator_ij
        U = 1.0 / denominators

        # Normalize rows to sum to 1 (numerical stability)
        U = U / jnp.sum(U, axis=1, keepdims=True)

        return U

    @partial(jit, static_argnums=(0,))
    def _compute_objective(
        self,
        X: chex.Array,
        centroids: chex.Array,
        U: chex.Array
    ) -> chex.Array:
        """
        Compute FCM objective function.

        Mathematical Formula:
            J = Σ_i Σ_j u_ij^m * ||x_i - v_j||^2

        Vectorized Implementation:
            J = sum(U^m ⊙ D^2)
            where ⊙ is element-wise product

        Args:
            X: (n, d) data matrix
            centroids: (c, d) centroids
            U: (n, c) fuzzy membership

        Returns:
            J: scalar objective value

        Complexity: O(ncd) - single pass
        Old complexity: O(ncd) but with nested loops
        """
        # Squared distances: (n, c)
        D_sq = squared_euclidean_distance_matrix(X, centroids)

        # Fuzzified membership: (n, c)
        U_fuzz = jnp.power(U, self.fuzzifier)

        # Element-wise multiply and sum all
        objective = jnp.sum(U_fuzz * D_sq)

        return objective

    @partial(jit, static_argnums=(0,))
    def _check_convergence(
        self,
        centroids_old: chex.Array,
        centroids_new: chex.Array
    ) -> chex.Array:
        """
        Check if centroids have converged.

        Formula: ||V_new - V_old||_F < epsilon

        Returns:
            Boolean scalar (as JAX array)
        """
        diff = jnp.linalg.norm(centroids_new - centroids_old, ord='fro')
        return diff < self.epsilon

    @partial(jit, static_argnums=(0,))
    def _single_iteration(
        self,
        state: FCMState,
        X: chex.Array
    ) -> Tuple[FCMState, dict]:
        """
        Single iteration of FCM algorithm.

        This function is JIT-compiled and used in lax.scan for fast looping.

        Args:
            state: Current algorithm state
            X: Data matrix (passed as auxiliary data)

        Returns:
            new_state: Updated state
            metrics: Dictionary of metrics for this iteration
        """
        # Update membership matrix
        U_new = self._update_U(X, state.centroids)

        # Compute new centroids
        centroids_new = self._compute_centroids(X, U_new)

        # Compute objective
        obj_new = self._compute_objective(X, centroids_new, U_new)

        # Check convergence
        converged = self._check_convergence(state.centroids, centroids_new)

        # Create new state
        new_state = FCMState(
            centroids=centroids_new,
            U=U_new,
            objective=obj_new,
            iteration=state.iteration + 1
        )

        # Return metrics for tracking
        metrics = {
            'objective': obj_new,
            'centroid_change': jnp.linalg.norm(
                centroids_new - state.centroids
            ),
            'converged': converged
        }

        return new_state, metrics

    @partial(jit, static_argnums=(0,))
    def _fit_loop(
        self,
        X: chex.Array,
        initial_state: FCMState
    ) -> Tuple[FCMState, dict]:
        """
        Main training loop using jax.lax.scan for efficiency.

        lax.scan is like a compiled for-loop that maintains state.
        Much faster than Python for-loop.

        Args:
            X: (n, d) data matrix
            initial_state: Starting state

        Returns:
            final_state: Converged state
            history: Training history
        """
        def scan_fn(state, _):
            return self._single_iteration(state, X)

        # Run for max_iter iterations
        final_state, history = jax.lax.scan(
            scan_fn,
            initial_state,
            None,
            length=self.max_iter
        )

        return final_state, history

    def fit(self, X: jnp.ndarray) -> 'FCM_JAX':
        """
        Fit FCM model to data.

        Args:
            X: (n_samples, n_features) data matrix (NumPy or JAX array)

        Returns:
            self
        """
        # Convert to JAX array if needed
        X = jnp.asarray(X)

        # Validate input
        chex.assert_rank(X, 2)

        # Initialize membership matrix
        self.key, subkey = jax.random.split(self.key)
        U_init = self._initialize_U(X, subkey)

        # Initialize centroids from U
        centroids_init = self._compute_centroids(X, U_init)

        # Initialize objective
        obj_init = self._compute_objective(X, centroids_init, U_init)

        # Create initial state
        initial_state = FCMState(
            centroids=centroids_init,
            U=U_init,
            objective=obj_init,
            iteration=0
        )

        # Run optimization
        final_state, self.history_ = self._fit_loop(X, initial_state)

        # Store results
        self.centroids_ = final_state.centroids
        self.U_ = final_state.U
        self.objective_ = final_state.objective
        self.n_iter_ = final_state.iteration

        return self

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Predict cluster labels for X.

        Args:
            X: (n_samples, n_features) data

        Returns:
            labels: (n_samples,) cluster assignments
        """
        X = jnp.asarray(X)

        # Compute membership matrix
        U = self._update_U(X, self.centroids_)

        # Hard assignment: argmax over clusters
        labels = jnp.argmax(U, axis=1)

        return labels

    def predict_proba(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Predict fuzzy membership for X.

        Args:
            X: (n_samples, n_features) data

        Returns:
            U: (n_samples, n_clusters) fuzzy membership
        """
        X = jnp.asarray(X)
        return self._update_U(X, self.centroids_)

    def get_objective_history(self) -> jnp.ndarray:
        """Return objective function values across iterations."""
        return self.history_['objective']

    def final_centroids(self) -> jnp.ndarray:
        """Return learned cluster centroids."""
        return self.centroids_
```

**Testing Strategy**:
```python
# tests/test_fcm_jax.py

def test_fcm_jax_vs_numpy():
    """Test that JAX and NumPy implementations give same results."""
    from sklearn.datasets import load_iris
    X, _ = load_iris(return_X_y=True)

    # Fit both models with same random seed
    fcm_numpy = FCM(X, c=3, m=2, num_iter=100, epsilon=1e-5, ord='fro')
    fcm_numpy.fit()

    fcm_jax = FCM_JAX(n_clusters=3, fuzzifier=2.0, max_iter=100,
                      epsilon=1e-5, random_seed=42)
    fcm_jax.fit(X)

    # Compare centroids (may need sorting due to label switching)
    np.testing.assert_allclose(
        sort_centroids(fcm_numpy.final_centroids()),
        sort_centroids(fcm_jax.final_centroids()),
        rtol=1e-4, atol=1e-6
    )

def test_fcm_jax_gpu():
    """Test GPU execution if available."""
    if jax.devices('gpu'):
        X = jnp.array(np.random.randn(10000, 50))
        model = FCM_JAX(n_clusters=10)
        model.fit(X)  # Should automatically use GPU
        assert model.centroids_.device() == jax.devices('gpu')[0]
```

**Performance Testing**:
```python
def benchmark_fcm():
    """Compare CPU vs GPU performance."""
    sizes = [100, 1000, 10000, 100000]
    n_features = 50
    n_clusters = 10

    results = []
    for n in sizes:
        X = np.random.randn(n, n_features)

        # NumPy version
        start = time.time()
        fcm_np = FCM(X, c=n_clusters, m=2, num_iter=100,
                     epsilon=1e-5, ord='fro')
        fcm_np.fit()
        numpy_time = time.time() - start

        # JAX CPU version
        with jax.default_device(jax.devices('cpu')[0]):
            start = time.time()
            fcm_jax_cpu = FCM_JAX(n_clusters=n_clusters)
            fcm_jax_cpu.fit(X)
            jax_cpu_time = time.time() - start

        # JAX GPU version (if available)
        if jax.devices('gpu'):
            with jax.default_device(jax.devices('gpu')[0]):
                start = time.time()
                fcm_jax_gpu = FCM_JAX(n_clusters=n_clusters)
                fcm_jax_gpu.fit(X)
                jax_gpu_time = time.time() - start

        results.append({
            'n_samples': n,
            'numpy': numpy_time,
            'jax_cpu': jax_cpu_time,
            'jax_gpu': jax_gpu_time if jax.devices('gpu') else None,
            'speedup_cpu': numpy_time / jax_cpu_time,
            'speedup_gpu': numpy_time / jax_gpu_time if jax.devices('gpu') else None
        })

    return pd.DataFrame(results)
```

---

#### 2.2 Possibilistic C-Means (PCM) - JAX Implementation

**File**: `prosemble/models/pcm_jax.py`

**Additional Complexity**:
1. Gamma parameter computation
2. Typicality matrix T (in addition to U)
3. Requires FCM initialization

**Key Differences**:
```python
@partial(jit, static_argnums=(0,))
def _compute_gamma(
    self,
    X: chex.Array,
    centroids: chex.Array,
    U: chex.Array
) -> chex.Array:
    """
    Compute gamma (scale) parameters for PCM.

    Mathematical Formula:
        γ_j = (K / Σ_i u_ij^m) * Σ_i (u_ij^m * ||x_i - v_j||^2)

    Vectorized Implementation:
        D_sq = squared_euclidean_distance_matrix(X, centroids)  # (n, c)
        U_fuzz = U^m  # (n, c)
        gamma = (U_fuzz * D_sq).sum(axis=0) * K / U_fuzz.sum(axis=0)

    Returns:
        gamma: (c,) scale parameters
    """
    D_sq = squared_euclidean_distance_matrix(X, centroids)
    U_fuzz = jnp.power(U, self.fuzzifier)

    numerator = jnp.sum(U_fuzz * D_sq, axis=0)  # (c,)
    denominator = jnp.sum(U_fuzz, axis=0)  # (c,)

    gamma = (numerator / denominator) * self.k
    return gamma

@partial(jit, static_argnums=(0,))
def _update_T(
    self,
    X: chex.Array,
    centroids: chex.Array,
    gamma: chex.Array
) -> chex.Array:
    """
    Update typicality matrix T.

    Mathematical Formula:
        t_ij = 1 / (1 + (||x_i - v_j||^2 / γ_j)^(1/(m-1)))

    Args:
        X: (n, d) data
        centroids: (c, d) centroids
        gamma: (c,) scale parameters

    Returns:
        T: (n, c) typicality matrix
    """
    D_sq = squared_euclidean_distance_matrix(X, centroids)  # (n, c)

    # Broadcasting: (n, c) / (c,) = (n, c)
    ratios = D_sq / gamma[None, :]

    power = 1.0 / (self.fuzzifier - 1)
    denominators = 1.0 + jnp.power(ratios, power)

    T = 1.0 / denominators
    return T

@partial(jit, static_argnums=(0,))
def _compute_objective_pcm(
    self,
    X: chex.Array,
    centroids: chex.Array,
    T: chex.Array,
    gamma: chex.Array
) -> chex.Array:
    """
    PCM objective function.

    Formula:
        J = Σ_i Σ_j t_ij^m * ||x_i - v_j||^2
            + Σ_j γ_j * Σ_i (1 - t_ij)^m
    """
    D_sq = squared_euclidean_distance_matrix(X, centroids)
    T_fuzz = jnp.power(T, self.fuzzifier)

    # First term: Σ t_ij^m * d_ij^2
    term1 = jnp.sum(T_fuzz * D_sq)

    # Second term: Σ_j γ_j * Σ_i (1 - t_ij)^m
    term2 = jnp.sum(
        gamma * jnp.sum(jnp.power(1 - T, self.fuzzifier), axis=0)
    )

    return term1 + term2
```

---

#### 2.3 Kernel Methods - JAX Implementation

**File**: `prosemble/models/kfcm_jax.py`

**Key Innovation**: Use kernel matrices instead of pairwise computation

```python
@partial(jit, static_argnums=(0,))
def _compute_kernel_matrix(
    self,
    X: chex.Array,
    centroids: chex.Array
) -> chex.Array:
    """
    Compute RBF kernel matrix.

    Formula: K[i,j] = exp(-||x_i - v_j||^2 / (2σ^2))

    Returns:
        K: (n, c) kernel matrix
    """
    D_sq = squared_euclidean_distance_matrix(X, centroids)
    K = jnp.exp(-D_sq / (2 * self.sigma**2))
    return K

@partial(jit, static_argnums=(0,))
def _compute_kernel_centroids(
    self,
    X: chex.Array,
    centroids_old: chex.Array,
    U: chex.Array
) -> chex.Array:
    """
    Update centroids in kernel space.

    Mathematical Formula:
        v_j = Σ_i (u_ij^m * x_i * k(x_i, v_j)) / Σ_i (u_ij^m * k(x_i, v_j))

    Note: centroids_old used for kernel computation
    """
    U_fuzz = jnp.power(U, self.fuzzifier)  # (n, c)
    K = self._compute_kernel_matrix(X, centroids_old)  # (n, c)

    # Weight by both fuzzy membership and kernel similarity
    weights = U_fuzz * K  # (n, c)

    # Numerator: (c, n) @ (n, d) = (c, d)
    numerator = weights.T @ X

    # Denominator: (c,)
    denominator = jnp.sum(weights, axis=0, keepdims=True).T
    denominator = jnp.maximum(denominator, 1e-10)

    centroids = numerator / denominator
    return centroids

@partial(jit, static_argnums=(0,))
def _update_U_kernel(
    self,
    X: chex.Array,
    centroids: chex.Array
) -> chex.Array:
    """
    Update fuzzy membership using kernel distances.

    Kernel distance: d_k(x, v) = 2 - 2*k(x, v)

    Formula:
        u_ij = 1 / Σ_k ((1 - k(x_i, v_j)) / (1 - k(x_i, v_k)))^(1/(m-1))
    """
    K = self._compute_kernel_matrix(X, centroids)  # (n, c)

    # Kernel distance: d_k = 2 - 2K (normalized so k=1 gives d=0)
    # Simplify: use (1 - K) directly
    D_kernel = 1 - K  # (n, c)
    D_kernel = jnp.maximum(D_kernel, 1e-10)

    power = 1.0 / (self.fuzzifier - 1)

    # (n, c, 1) / (n, 1, c) = (n, c, c)
    ratios = (D_kernel[:, :, None] / D_kernel[:, None, :]) ** power

    # Sum over k: (n, c)
    denominators = jnp.sum(ratios, axis=2)

    U = 1.0 / denominators
    U = U / jnp.sum(U, axis=1, keepdims=True)

    return U
```

---

### Phase 3: Parallel Processing & Optimization (Weeks 7-8)

#### 3.1 Batch Processing

**Problem**: Current implementation processes entire dataset at once. For large datasets, use mini-batches.

```python
@partial(jit, static_argnums=(0,))
def _fit_minibatch(
    self,
    batch_X: chex.Array,
    state: FCMState
) -> FCMState:
    """
    Single mini-batch update.

    Strategy:
    1. Compute U for this batch only
    2. Update centroids with weighted average
    3. Exponential moving average for stability
    """
    # Compute membership for batch
    U_batch = self._update_U(batch_X, state.centroids)

    # Compute centroid update from batch
    centroids_batch = self._compute_centroids(batch_X, U_batch)

    # Exponential moving average
    alpha = 0.1  # Learning rate
    centroids_new = (1 - alpha) * state.centroids + alpha * centroids_batch

    # ... update state
    return new_state

def fit_minibatch(
    self,
    X: jnp.ndarray,
    batch_size: int = 1000
) -> 'FCM_JAX':
    """
    Fit using mini-batch updates.

    Useful for very large datasets that don't fit in GPU memory.
    """
    n_samples = X.shape[0]
    n_batches = n_samples // batch_size

    for epoch in range(self.max_iter):
        # Shuffle data
        key, subkey = jax.random.split(self.key)
        perm = jax.random.permutation(subkey, n_samples)
        X_shuffled = X[perm]

        # Process batches
        for i in range(n_batches):
            batch = X_shuffled[i*batch_size:(i+1)*batch_size]
            self.state_ = self._fit_minibatch(batch, self.state_)

        # Check convergence on full dataset periodically
        if epoch % 10 == 0:
            # ... convergence check
            pass

    return self
```

#### 3.2 Multi-Device Parallelism

**Strategy**: Split data across multiple GPUs/TPUs

```python
import jax.experimental.pjit as pjit
from jax.experimental import mesh_utils
from jax.experimental import PartitionSpec as P

def fit_multi_device(
    self,
    X: jnp.ndarray
) -> 'FCM_JAX':
    """
    Fit using multiple devices in parallel.

    Strategy:
    - Shard data across devices (dimension 0)
    - Replicate centroids on all devices
    - Compute local updates
    - Aggregate via all-reduce
    """
    # Create device mesh
    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = jax.experimental.maps.Mesh(devices, ('data',))

    # Partition specification
    data_spec = P('data')  # Shard along first dimension
    centroid_spec = P()  # Replicate

    @partial(
        pjit.pjit,
        in_axis_resources=(data_spec, centroid_spec),
        out_axis_resources=centroid_spec
    )
    def distributed_update(X_shard, centroids):
        # Each device processes its shard
        U_shard = self._update_U(X_shard, centroids)

        # Local centroid computation
        centroids_local = self._compute_centroids(X_shard, U_shard)

        # All-reduce to get global centroids
        centroids_global = jax.lax.pmean(
            centroids_local,
            axis_name='data'
        )

        return centroids_global

    # Run training with data parallelism
    with mesh:
        for i in range(self.max_iter):
            self.centroids_ = distributed_update(X, self.centroids_)
            # ... rest of training loop

    return self
```

#### 3.3 JIT Compilation Optimization

**Strategy**: Optimize compilation for different input sizes

```python
# Cache compiled functions for common sizes
_jit_cache = {}

def get_jitted_update(n_samples, n_features):
    """Get JIT-compiled function for specific input size."""
    key = (n_samples, n_features)
    if key not in _jit_cache:
        @jit
        def update_fn(X, centroids):
            # ... update logic with shape (n_samples, n_features)
            pass
        _jit_cache[key] = update_fn
    return _jit_cache[key]

# Usage
update_fn = get_jitted_update(X.shape[0], X.shape[1])
U = update_fn(X, centroids)
```

---

### Phase 4: Testing & Validation (Weeks 9-10)

#### 4.1 Correctness Tests

**Test Matrix**:
```
Models to Test:
- FCM, PCM, FPCM, PFCM
- KFCM, KPCM, KFPCM, KPFCM
- AFCM, KAFCM
- IPCM, IPCM_2, KIPCM, KIPCM2
- HCM, BGPC, NPC
- Kmeans, KNN, SOM

Test Cases per Model:
1. Small dataset (n=150, d=4) - Iris
2. Medium dataset (n=1000, d=20)
3. Large dataset (n=100000, d=50)
4. High-dimensional (n=1000, d=500)

Validation Metrics:
- Centroid agreement (MSE < 1e-4)
- Membership matrix agreement (MAE < 1e-5)
- Objective function agreement (relative error < 1e-3)
- Prediction agreement (accuracy = 100%)
```

**Test Suite Structure**:
```python
# tests/test_jax_numpy_parity.py

@pytest.mark.parametrize("model_class", [
    (FCM, FCM_JAX),
    (PCM, PCM_JAX),
    (FPCM, FPCM_JAX),
    # ... all models
])
@pytest.mark.parametrize("dataset", [
    "iris",
    "breast_cancer",
    "random_small",
    "random_large"
])
def test_model_parity(model_class, dataset):
    """Test JAX vs NumPy parity for all models."""
    numpy_model, jax_model = model_class
    X, y = get_dataset(dataset)

    # Fit both with same initialization
    np.random.seed(42)
    # ... fit numpy model

    jax.random.PRNGKey(42)
    # ... fit jax model

    # Compare results
    assert_centroids_close(numpy_centroids, jax_centroids)
    assert_memberships_close(numpy_U, jax_U)
    assert_objectives_close(numpy_obj, jax_obj)
```

#### 4.2 Performance Benchmarks

**Benchmark Suite**:
```python
# benchmarks/benchmark_speed.py

def benchmark_all_models():
    """Comprehensive performance comparison."""

    configs = [
        # (n_samples, n_features, n_clusters)
        (100, 10, 3),
        (1000, 20, 5),
        (10000, 50, 10),
        (100000, 100, 20),
    ]

    models = ['FCM', 'PCM', 'KFCM', ...]

    results = []
    for n, d, c in configs:
        X = generate_data(n, d, c)

        for model_name in models:
            # NumPy timing
            numpy_time = time_model(get_numpy_model(model_name), X, c)

            # JAX CPU timing
            jax_cpu_time = time_model(get_jax_model(model_name), X, c, device='cpu')

            # JAX GPU timing
            if has_gpu():
                jax_gpu_time = time_model(get_jax_model(model_name), X, c, device='gpu')

            results.append({
                'model': model_name,
                'n_samples': n,
                'n_features': d,
                'numpy_time': numpy_time,
                'jax_cpu_time': jax_cpu_time,
                'jax_gpu_time': jax_gpu_time,
                'speedup_cpu': numpy_time / jax_cpu_time,
                'speedup_gpu': numpy_time / jax_gpu_time,
            })

    return pd.DataFrame(results)

# Expected Results:
# Small datasets (n < 1000): JAX ~1-5× faster
# Medium datasets (n ~ 10000): JAX ~10-50× faster
# Large datasets (n > 100000): JAX ~100-1000× faster (GPU)
```

#### 4.3 Numerical Stability Tests

```python
# tests/test_numerical_stability.py

def test_stability_edge_cases():
    """Test numerical edge cases."""

    # Test 1: Very small distances (potential division by zero)
    X = np.array([[0, 0], [1e-10, 1e-10], [0, 0]])
    model = FCM_JAX(n_clusters=2)
    model.fit(X)  # Should not raise or produce NaN
    assert not jnp.isnan(model.centroids_).any()

    # Test 2: Very large values
    X = np.random.randn(100, 10) * 1e6
    model.fit(X)
    assert jnp.isfinite(model.centroids_).all()

    # Test 3: High condition number data
    X = generate_ill_conditioned_data(1000, 50)
    model.fit(X)
    # Check that algorithm converges
    assert model.n_iter_ < model.max_iter
```

---

### Phase 5: Documentation & Examples (Weeks 11-12)

#### 5.1 Migration Guide

**File**: `docs/migration_guide.md`

```markdown
# Migrating from NumPy to JAX

## Quick Start

### Before (NumPy):
```python
from prosemble.models import FCM

X, y = load_iris(return_X_y=True)
model = FCM(data=X, c=3, m=2, num_iter=100, epsilon=1e-5, ord='fro')
model.fit()
labels = model.predict()
centroids = model.final_centroids()
```

### After (JAX):
```python
from prosemble.models.jax import FCM_JAX

X, y = load_iris(return_X_y=True)
model = FCM_JAX(n_clusters=3, fuzzifier=2.0, max_iter=100, epsilon=1e-5)
model.fit(X)
labels = model.predict(X)
centroids = model.final_centroids()
```

## Key Differences

### 1. API Changes
- NumPy: `FCM(data=X, c=3, m=2, ...)`
- JAX: `FCM_JAX(n_clusters=3, fuzzifier=2.0, ...)`
- More scikit-learn-like API

### 2. GPU Usage
```python
# Automatic GPU usage if available
import jax
print(f"Devices: {jax.devices()}")  # Check available devices

# Force CPU
with jax.default_device(jax.devices('cpu')[0]):
    model.fit(X)

# Force GPU
with jax.default_device(jax.devices('gpu')[0]):
    model.fit(X)
```

### 3. Large Datasets
```python
# For datasets that don't fit in memory, use mini-batch mode
model = FCM_JAX(n_clusters=10)
model.fit_minibatch(X, batch_size=10000)
```

### 4. Parallel Training (Multiple GPUs)
```python
model = FCM_JAX(n_clusters=10)
model.fit_multi_device(X)  # Automatically uses all available GPUs
```

## Performance Tips

1. **First iteration is slow**: JIT compilation happens on first call
2. **Reuse models**: Compiled functions are cached
3. **Batch your predictions**: Process multiple samples at once
4. **Use appropriate dtypes**: float32 is faster than float64 on GPU

## Troubleshooting

### Out of Memory on GPU
```python
# Solution 1: Use mini-batch training
model.fit_minibatch(X, batch_size=5000)

# Solution 2: Use float32 instead of float64
X = X.astype(np.float32)
```

### Slower than NumPy
- Check if GPU is actually being used
- Make sure dataset is large enough (JAX has overhead for small data)
- First run includes compilation time

## Feature Comparison

| Feature | NumPy | JAX |
|---------|-------|-----|
| Basic clustering | ✓ | ✓ |
| GPU support | ✗ | ✓ |
| JIT compilation | ✗ | ✓ |
| Automatic differentiation | ✗ | ✓ |
| Mini-batch training | ✗ | ✓ |
| Multi-GPU | ✗ | ✓ |
| Plotting | ✓ | ✓ |
```

#### 5.2 Tutorial Notebooks

**File**: `examples/jax_tutorial.ipynb`

Topics:
1. Basic JAX FCM usage
2. GPU vs CPU comparison
3. Large-scale clustering (100k+ samples)
4. Mini-batch training
5. Hyperparameter tuning with JAX
6. Visualizing convergence
7. Custom distance metrics
8. Saving/loading models

#### 5.3 API Documentation

Update docstrings with:
- Mathematical formulations (LaTeX)
- Complexity analysis
- GPU memory requirements
- Performance characteristics
- Examples

---

### Phase 6: Integration & Deployment (Weeks 13-14)

#### 6.1 Backward Compatibility Layer

**File**: `prosemble/models/__init__.py`

```python
# Allow users to switch between NumPy and JAX
import os

USE_JAX = os.environ.get('PROSEMBLE_USE_JAX', 'auto')

if USE_JAX == 'auto':
    try:
        import jax
        USE_JAX = True
    except ImportError:
        USE_JAX = False
elif USE_JAX in ('true', '1', 'yes'):
    USE_JAX = True
else:
    USE_JAX = False

if USE_JAX:
    from .jax import FCM as FCM
    from .jax import PCM as PCM
    # ... import JAX versions
else:
    from .fcm import FCM
    from .pcm import PCM
    # ... import NumPy versions

__all__ = ['FCM', 'PCM', ...]
```

**Usage**:
```bash
# Use JAX by default
python my_script.py

# Force NumPy
PROSEMBLE_USE_JAX=false python my_script.py

# Force JAX
PROSEMBLE_USE_JAX=true python my_script.py
```

#### 6.2 Dependency Management

**Update `pyproject.toml`**:
```toml
[project]
dependencies = [
    "numpy>=1.20",
    "scikit-learn>=1.0",
    "pandas>=1.3",
    "scipy>=1.7",
    "matplotlib>=3.4"
]

[project.optional-dependencies]
jax = [
    "jax>=0.4.23",
    "jaxlib>=0.4.23",
]

jax-cuda = [
    "jax[cuda12]>=0.4.23",
]

all = [
    "jax>=0.4.23",
    "jaxlib>=0.4.23",
]
```

**Installation**:
```bash
# NumPy only (default)
pip install prosemble

# With JAX (CPU)
pip install prosemble[jax]

# With JAX (GPU)
pip install prosemble[jax-cuda]
```

#### 6.3 CI/CD Updates

**Update `.github/workflows/`**:

```yaml
# .github/workflows/test-jax.yml
name: JAX Tests

on: [push, pull_request]

jobs:
  test-jax-cpu:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -e .[jax]
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        pytest tests/test_*_jax.py --cov=prosemble

    - name: Check parity
      run: |
        pytest tests/test_jax_numpy_parity.py

  test-jax-gpu:
    runs-on: ubuntu-latest
    # Only on main branch to save GPU time
    if: github.ref == 'refs/heads/main'
    container:
      image: nvidia/cuda:12.0.0-cudnn8-runtime-ubuntu22.04

    steps:
    - uses: actions/checkout@v3
    - name: Install JAX GPU
      run: |
        pip install -e .[jax-cuda]
        pip install pytest

    - name: Run GPU tests
      run: |
        pytest tests/test_*_jax.py -k gpu

    - name: Benchmark
      run: |
        python benchmarks/benchmark_gpu.py

  benchmark:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Run benchmarks
      run: |
        python benchmarks/benchmark_all.py
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: benchmarks/results/
```

---

## Risk Assessment & Mitigation

### Risk 1: Breaking Changes
**Probability**: Medium
**Impact**: High
**Mitigation**:
- Maintain NumPy versions alongside JAX
- Provide compatibility layer
- Extensive testing
- Clear migration documentation

### Risk 2: Performance Regression on Small Data
**Probability**: High (JAX has overhead)
**Impact**: Low
**Mitigation**:
- Document minimum dataset size for JAX
- Auto-fallback to NumPy for small data
- Benchmark and document performance characteristics

### Risk 3: GPU Out-of-Memory
**Probability**: Medium
**Impact**: Medium
**Mitigation**:
- Implement mini-batch training
- Provide memory estimation utilities
- Clear error messages with suggestions

### Risk 4: Numerical Differences
**Probability**: Low
**Impact**: Medium
**Mitigation**:
- Extensive numerical stability tests
- Document any known differences
- Configurable tolerance levels

### Risk 5: Community Adoption
**Probability**: Medium
**Impact**: Medium
**Mitigation**:
- Gradual rollout (NumPy remains default)
- Clear benefits communication
- Tutorial notebooks
- Responsive to feedback

---

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **Phase 1: Foundation** | Weeks 1-2 | JAX setup, distance module, tests |
| **Phase 2: Core Algorithms** | Weeks 3-6 | FCM, PCM, FPCM, KFCM in JAX |
| **Phase 3: Optimization** | Weeks 7-8 | Batch processing, multi-device, JIT optimization |
| **Phase 4: Testing** | Weeks 9-10 | Comprehensive test suite, benchmarks |
| **Phase 5: Documentation** | Weeks 11-12 | Migration guide, tutorials, API docs |
| **Phase 6: Integration** | Weeks 13-14 | Backward compatibility, CI/CD, deployment |

**Total Duration**: 14 weeks (~3.5 months)

---

## Success Metrics

### Performance Targets
- [ ] **10-100× speedup** on large datasets (n > 10,000) with GPU
- [ ] **<5% overhead** on small datasets (n < 1,000)
- [ ] **Linear scaling** with number of GPUs in multi-device mode
- [ ] **Sub-second** first prediction after JIT compilation

### Quality Targets
- [ ] **100% API parity** with NumPy versions
- [ ] **<1e-4 numerical difference** in results
- [ ] **>95% test coverage** for JAX code
- [ ] **Zero regression** in existing functionality

### Adoption Targets
- [ ] **Clear documentation** for all features
- [ ] **5+ tutorial notebooks**
- [ ] **Benchmarks** published for all models
- [ ] **Migration guide** with examples

---

## Post-Migration Roadmap

### Short-term (3 months)
1. User feedback collection
2. Bug fixes and optimizations
3. Additional tutorial content
4. Performance tuning

### Medium-term (6 months)
1. Automatic Mixed Precision (AMP) support
2. Distributed training across multiple machines
3. Custom kernel implementations for critical paths
4. Integration with JAX ecosystem (Optax, Flax, etc.)

### Long-term (12 months)
1. Automatic hyperparameter tuning with JAX
2. Differentiable clustering for end-to-end learning
3. Probabilistic programming integration
4. Quantum-inspired algorithms

---

## Appendix

### A. JAX Resources
- Official docs: https://jax.readthedocs.io/
- Performance guide: https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html
- GPU memory management: https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html

### B. Mathematical Formulations

#### Fuzzy C-Means
```
Objective: J = Σᵢ Σⱼ uᵢⱼᵐ ||xᵢ - vⱼ||²

Subject to: Σⱼ uᵢⱼ = 1 ∀i
            uᵢⱼ ∈ [0,1] ∀i,j

Updates:
  vⱼ = Σᵢ(uᵢⱼᵐ xᵢ) / Σᵢ uᵢⱼᵐ
  uᵢⱼ = 1 / Σₖ (||xᵢ-vⱼ|| / ||xᵢ-vₖ||)^(2/(m-1))
```

#### Possibilistic C-Means
```
Objective: J = Σᵢ Σⱼ tᵢⱼᵐ ||xᵢ - vⱼ||² + Σⱼ γⱼ Σᵢ (1-tᵢⱼ)ᵐ

Updates:
  vⱼ = Σᵢ(tᵢⱼᵐ xᵢ) / Σᵢ tᵢⱼᵐ
  tᵢⱼ = 1 / (1 + (||xᵢ-vⱼ||²/γⱼ)^(1/(m-1)))
  γⱼ = K Σᵢ(uᵢⱼᵐ ||xᵢ-vⱼ||²) / Σᵢ uᵢⱼᵐ
```

#### Kernel FCM
```
Feature map: φ: ℝᵈ → ℋ (Hilbert space)
Kernel: k(x,y) = ⟨φ(x), φ(y)⟩

Kernel distance: ||φ(x)-φ(y)||² = k(x,x) - 2k(x,y) + k(y,y)

For Gaussian kernel: k(x,y) = exp(-||x-y||²/(2σ²))

Objective: J = Σᵢ Σⱼ uᵢⱼᵐ ||φ(xᵢ) - φ(vⱼ)||²
```

### C. Code Review Checklist

For each migrated model:
- [ ] All NumPy operations replaced with JAX equivalents
- [ ] No Python loops over data/clusters
- [ ] Functions decorated with `@jit`
- [ ] Uses `chex` for shape assertions
- [ ] Immutable state (NamedTuple)
- [ ] Docstrings include mathematical formulas
- [ ] Unit tests compare with NumPy version
- [ ] Performance benchmark included
- [ ] GPU-compatible (no host-device transfers in loop)
- [ ] Handles edge cases (zero distances, singular matrices)

---

**Document Version**: 1.0
**Last Updated**: January 15, 2026
**Authors**: Analysis based on Prosemble codebase study
**Status**: Ready for Review & Implementation
