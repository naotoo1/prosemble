"""
One-Class LGMLVQ (OC-LGMLVQ) example using Iris Data with JAX.

This example demonstrates:
- One-class classification with per-prototype local Omega matrices
- Each prototype learns its own metric, allowing different acceptance
  regions to use different feature weightings
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils import train_test_split_jax
from prosemble.models import OCLGMLVQ

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

# Setup OC-LGMLVQ model
model = OCLGMLVQ(
    n_prototypes=3,
    target_label=target_label,
    latent_dim=2,
    beta=10.0,
    max_iter=100,
    lr=0.001,
    random_seed=42,
)

# Fit model
print("\nTraining OC-LGMLVQ...")
model.fit(X_train, y_train)

# Results
print(f"Converged in {model.n_iter_} iterations")
print(f"Final loss: {model.loss_:.4f}")

# Predictions
test_preds = model.predict(X_test)
proba = model.predict_proba(X_test)
print(f"\nProbabilities shape: {proba.shape}")
print(f"Prototypes shape: {model.prototypes_.shape}")
print(f"Visibility radii: {model.visibility_radii}")
