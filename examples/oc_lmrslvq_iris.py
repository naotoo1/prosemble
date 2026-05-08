"""
One-Class Localized Matrix RSLVQ (OC-LMRSLVQ) example using Iris Data with JAX.

This example demonstrates:
- One-class classification with per-prototype Omega matrices
- Probabilistic soft-weighting of all prototypes via Gaussian mixture
- Local metric adaptation for novelty detection
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.models import OCLMRSLVQ

# Load data — treat class 0 as target, rest as outliers
dataset = load_iris_jax()
X, y_raw = dataset.input_data, dataset.labels
y = jnp.where(y_raw == 0, 0, 1).astype(jnp.int32)

print(f"Dataset: {X.shape}")
print(f"Target (class 0): {int(jnp.sum(y == 0))}, Outlier: {int(jnp.sum(y == 1))}")

# Setup OC-LMRSLVQ model
model = OCLMRSLVQ(
    sigma=1.0,
    n_prototypes=3,
    max_iter=100,
    lr=0.01,
    target_label=0,
    random_seed=42,
)

# Fit model
print("\nTraining OC-LMRSLVQ...")
model.fit(X, y)

# Results
print(f"Converged in {model.n_iter_} iterations")
print(f"Final loss: {model.loss_:.4f}")
print(f"Loss decreased: {model.loss_history_[0]:.4f} -> {model.loss_history_[-1]:.4f}")

# Predictions
preds = model.predict(X)
target_acc = float(jnp.mean(preds[y == 0] == 0))
outlier_acc = float(jnp.mean(preds[y == 1] == 1))
overall_acc = float(jnp.mean(preds == y))

print(f"\nTarget accuracy:  {target_acc:.2%}")
print(f"Outlier accuracy: {outlier_acc:.2%}")
print(f"Overall accuracy: {overall_acc:.2%}")

# Decision scores
scores = model.decision_function(X)
mean_target = float(jnp.mean(scores[y == 0]))
mean_outlier = float(jnp.mean(scores[y == 1]))
print(f"\nMean target score:  {mean_target:.4f}")
print(f"Mean outlier score: {mean_outlier:.4f}")

# Per-prototype Omega matrices
print(f"\nLocal omegas shape: {model.omegas_.shape}")
print(f"Thetas: {model.thetas_}")

# Predict with reject option
preds_reject = model.predict_with_reject(X, upper=0.7, lower=0.3)
n_accepted = int(jnp.sum(preds_reject == 0))
n_rejected = int(jnp.sum(preds_reject == 1))
n_uncertain = int(jnp.sum(preds_reject == -1))
print(f"\nWith reject option: accepted={n_accepted}, rejected={n_rejected}, uncertain={n_uncertain}")
