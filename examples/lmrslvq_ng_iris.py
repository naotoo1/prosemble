"""
Localized Matrix RSLVQ with Neural Gas (LMRSLVQ_NG) example using Iris Data with JAX.

This example demonstrates:
- RSLVQ probabilistic loss with NG neighborhood cooperation
- Per-prototype Omega_k matrices for local metric adaptation
- Gamma decay during training
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils import train_test_split_jax
from prosemble.models import LMRSLVQ_NG

# Load data
dataset = load_iris_jax()
X, y = dataset.input_data, dataset.labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split_jax(
    X, y, test_size=0.2, random_seed=42
)

print(f"Dataset: {X.shape}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Setup LMRSLVQ_NG model
model = LMRSLVQ_NG(
    sigma=1.0,
    latent_dim=2,
    n_prototypes_per_class=2,
    max_iter=100,
    lr=0.01,
    gamma_init=3.0,
    gamma_final=0.01,
    random_seed=42,
)

# Fit model
print("\nTraining LMRSLVQ_NG...")
model.fit(X_train, y_train)

# Results
print(f"Converged in {model.n_iter_} iterations")
print(f"Final loss: {model.loss_:.4f}")
print(f"Final gamma: {model.gamma_:.4f}")

# Predictions
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_acc = float(jnp.mean(train_preds == y_train))
test_acc = float(jnp.mean(test_preds == y_test))

print(f"\nTrain accuracy: {train_acc:.2%}")
print(f"Test accuracy:  {test_acc:.2%}")

# Per-prototype Omegas
print(f"\nPer-prototype Omegas shape: {model.omegas_.shape}")

# Class probabilities
proba = model.predict_proba(X_test[:5])
print(f"\nClass probabilities (first 5 samples):\n{proba}")
