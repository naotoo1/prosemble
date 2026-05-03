"""
One-Class GLVQ (OC-GLVQ) example using Iris Data with JAX.

This example demonstrates:
- One-class classification with learned visibility thresholds
- Prototypes learn both position and acceptance radius theta_k
- Reject option: samples outside all visibility radii are rejected
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils import train_test_split_jax
from prosemble.models import OCGLVQ

# Load data
dataset = load_iris_jax()
X, y = dataset.input_data, dataset.labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split_jax(
    X, y, test_size=0.2, random_seed=42
)

# Use class 0 as the target (inlier) class
target_label = 0
print(f"Dataset: {X.shape}")
print(f"Target class: {target_label}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Setup OC-GLVQ model
model = OCGLVQ(
    n_prototypes=3,
    target_label=target_label,
    beta=10.0,
    max_iter=100,
    lr=0.01,
    random_seed=42,
)

# Fit model
print("\nTraining OC-GLVQ...")
model.fit(X_train, y_train)

# Results
print(f"Converged in {model.n_iter_} iterations")
print(f"Final loss: {model.loss_:.4f}")

# Predictions
test_preds = model.predict(X_test)
print(f"\nPredictions: {test_preds}")

# Reject option
preds_reject = model.predict_with_reject(
    X_test, upper=0.8, lower=0.2, reject_label=-1
)
n_rejected = int(jnp.sum(preds_reject == -1))
print(f"Predictions with reject: {n_rejected} rejected out of {len(X_test)}")

# Decision function and probabilities
scores = model.decision_function(X_test)
proba = model.predict_proba(X_test)
print(f"\nDecision scores shape: {scores.shape}")
print(f"Probabilities shape: {proba.shape}")

# Visibility radii
print(f"\nVisibility radii: {model.visibility_radii}")
print(f"Prototypes shape: {model.prototypes_.shape}")
