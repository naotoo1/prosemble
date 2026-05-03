"""
Supervised Localized Matrix Neural Gas (SLNG) example using Iris Data with JAX.

This example demonstrates:
- Multi-class Neural Gas with per-prototype local Omega matrices
- Each prototype learns its own metric for local discrimination
- Gamma decays during training for annealing
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils import train_test_split_jax
from prosemble.models import SLNG

# Load data
dataset = load_iris_jax()
X, y = dataset.input_data, dataset.labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split_jax(
    X, y, test_size=0.2, random_seed=42
)

print(f"Dataset: {X.shape}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Setup SLNG model
model = SLNG(
    n_prototypes_per_class=2,
    latent_dim=2,
    beta=10.0,
    gamma_init=5.0,
    gamma_final=0.01,
    max_iter=200,
    lr=0.001,
    random_seed=42,
)

# Fit model
print("\nTraining SLNG...")
model.fit(X_train, y_train)

# Results
print(f"Converged in {model.n_iter_} iterations")
print(f"Final loss: {model.loss_:.4f}")

# Predictions
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_acc = float(jnp.mean(train_preds == y_train))
test_acc = float(jnp.mean(test_preds == y_test))

print(f"\nTrain accuracy: {train_acc:.2%}")
print(f"Test accuracy:  {test_acc:.2%}")

print(f"\nPrototypes shape: {model.prototypes_.shape}")
