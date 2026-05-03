"""
Supervised Vector Quantization One-Class Classification (SVQ-OCC) example with JAX.

This example demonstrates:
- One-class classification using Neural Gas ranking + response probabilities
- Heaviside sigmoid activation for crisp acceptance decisions
- Contrastive cost function balancing inlier attraction and outlier repulsion
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils import train_test_split_jax
from prosemble.models import SVQOCC

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

# Setup SVQ-OCC model
model = SVQOCC(
    n_prototypes=3,
    target_label=target_label,
    alpha=0.5,
    sigma=0.1,
    lambda_init=3.0,
    lambda_final=0.01,
    max_iter=100,
    lr=0.01,
    random_seed=42,
)

# Fit model
print("\nTraining SVQ-OCC...")
model.fit(X_train, y_train)

# Results
print(f"Converged in {model.n_iter_} iterations")
print(f"Final loss: {model.loss_:.4f}")

# Predictions
test_preds = model.predict(X_test)
preds_reject = model.predict_with_reject(
    X_test, upper=0.8, lower=0.2, reject_label=-1
)
n_rejected = int(jnp.sum(preds_reject == -1))
print(f"\nRejected: {n_rejected}/{len(X_test)}")

# Decision function and probabilities
scores = model.decision_function(X_test)
proba = model.predict_proba(X_test)
print(f"Decision scores shape: {scores.shape}")
print(f"Probabilities shape: {proba.shape}")
print(f"Visibility radii: {model.visibility_radii}")
