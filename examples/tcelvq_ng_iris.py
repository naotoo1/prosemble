"""
Tangent Cross-Entropy LVQ with Neural Gas (TCELVQ-NG) example using Iris Data.

This example demonstrates:
- Multi-class classification with cross-entropy loss over softmax logits
- Per-prototype tangent subspaces learning invariance directions
- Neural Gas rank-based neighborhood cooperation across prototypes
- Calibrated probability estimates via softmax
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.models import TCELVQ_NG

# Load data
dataset = load_iris_jax()
X, y = dataset.input_data, dataset.labels

print(f"Dataset: {X.shape}")
print(f"Classes: {int(jnp.max(y)) + 1}")

# Setup TCELVQ-NG model
model = TCELVQ_NG(
    n_prototypes_per_class=3,
    subspace_dim=2,            # tangent subspace dimension
    gamma_init=5.0,            # initial neighborhood range
    gamma_final=0.01,          # final (narrower)
    max_iter=100,
    lr=0.01,
    random_seed=42,
)

# Fit model
print("\nTraining TCELVQ-NG...")
model.fit(X, y)

# Results
print(f"Converged in {model.n_iter_} iterations")
print(f"Final loss: {model.loss_:.4f}")
print(f"Final gamma: {model.gamma_:.4f}")
print(f"Loss decreased: {model.loss_history_[0]:.4f} -> {model.loss_history_[-1]:.4f}")

# Predictions
preds = model.predict(X)
accuracy = float(jnp.mean(preds == y))
print(f"\nTraining accuracy: {accuracy:.2%}")

# Calibrated probabilities
proba = model.predict_proba(X)
print(f"\nProbabilities shape: {proba.shape}")
print(f"Sample probabilities (first 5):")
for i in range(5):
    print(f"  y={int(y[i])}, pred={int(preds[i])}, proba={proba[i]}")

# Tangent subspace info
print(f"\nOmegas shape: {model.omegas_.shape} (prototypes x features x subspace_dim)")
print(f"Prototype labels: {model.prototype_labels_}")
