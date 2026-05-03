"""
Riemannian Neural Gas on SO(3) manifold example with JAX.

This example demonstrates:
- Neural Gas on curved (non-Euclidean) spaces
- Prototypes live on the SO(3) rotation group
- Updates use exponential/logarithmic maps for geodesic movement
"""

import jax
import jax.numpy as jnp
from prosemble.core.manifolds import SO
from prosemble.models import RiemannianNeuralGas

# Generate synthetic SO(3) data (3x3 rotation matrices)
key = jax.random.PRNGKey(42)
manifold = SO(n=3)

# Sample random rotations near identity
n_samples = 50
keys = jax.random.split(key, n_samples)
X = jnp.stack([manifold.random_point(k) for k in keys])

print(f"Dataset: {X.shape} (rotation matrices on SO(3))")

# Setup Riemannian Neural Gas
model = RiemannianNeuralGas(
    manifold=manifold,
    n_prototypes=5,
    lr_init=0.3,
    lr_final=0.01,
    lambda_init=3.0,
    lambda_final=0.01,
    max_iter=50,
    random_seed=42,
)

# Fit model
print("\nTraining Riemannian Neural Gas...")
model.fit(X)

# Results
print(f"Converged in {model.n_iter_} iterations")
print(f"Final loss: {model.loss_:.4f}")

# Cluster assignments
assignments = model.predict(X)
print(f"\nCluster assignments: {assignments.shape}")
print(f"Unique clusters used: {len(jnp.unique(assignments))}")
print(f"Prototypes shape: {model.prototypes_.shape}")
