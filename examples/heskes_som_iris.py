"""Heskes SOM on Iris dataset.

Demonstrates the Heskes Self-Organizing Map, which uses a modified BMU
definition (neighborhood-weighted) and pure batch updates that guarantee
monotonic energy decrease.
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.models import HeskesSOM

# Load data
dataset = load_iris_jax()
X = dataset.input_data

# Fit Heskes SOM
model = HeskesSOM(
    grid_height=5,
    grid_width=5,
    max_iter=100,
    sigma_final=0.5,
    random_seed=42,
)
model.fit(X)

print(f"Converged in {model.n_iter_} iterations")
print(f"Final energy: {model.loss_:.4f}")
print(f"Prototypes shape: {model.prototypes_.shape}")

# BMU assignments
bmu_coords = model.bmu_map(X)
print(f"BMU coordinates shape: {bmu_coords.shape}")

# Cluster assignments (nearest prototype)
labels = model.predict(X)
print(f"Unique clusters used: {len(jnp.unique(labels))}")

# Verify energy decreases monotonically
diffs = jnp.diff(jnp.array(model.loss_history_[:model.n_iter_]))
print(f"Energy always decreases: {bool(jnp.all(diffs <= 1e-6))}")
