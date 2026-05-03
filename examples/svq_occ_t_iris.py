"""
SVQ-OCC with Tangent Distance (SVQ-OCC-T) example using Iris Data with JAX.

This example demonstrates:
- One-class classification with per-prototype tangent subspaces
- Tangent distance ignores invariance directions for acceptance
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils import train_test_split_jax
from prosemble.models import SVQOCC_T

# Load data
dataset = load_iris_jax()
X, y = dataset.input_data, dataset.labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split_jax(
    X, y, test_size=0.2, random_seed=42
)

target_label = 0
print(f"Dataset: {X.shape}")
print(f"Target class: {target_label}")

# Setup SVQ-OCC-T model
model = SVQOCC_T(
    n_prototypes=3,
    target_label=target_label,
    subspace_dim=1,
    alpha=0.5,
    sigma=0.1,
    lambda_init=3.0,
    lambda_final=0.01,
    max_iter=100,
    lr=0.001,
    random_seed=42,
)

# Fit model
print("\nTraining SVQ-OCC-T...")
model.fit(X_train, y_train)

# Results
print(f"Converged in {model.n_iter_} iterations")
print(f"Final loss: {model.loss_:.4f}")

# Predictions with reject
test_preds = model.predict(X_test)
preds_reject = model.predict_with_reject(
    X_test, upper=0.8, lower=0.2, reject_label=-1
)
n_rejected = int(jnp.sum(preds_reject == -1))
print(f"\nRejected: {n_rejected}/{len(X_test)}")
print(f"Visibility radii: {model.visibility_radii}")
