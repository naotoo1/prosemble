"""
Matrix RSLVQ with Neural Gas Cooperation (MRSLVQ_NG) example using Iris Data with JAX.

This example demonstrates:
- RSLVQ probabilistic loss with NG neighborhood cooperation
- Global Omega matrix for metric adaptation
- Omega-projected distances: d(x, w) = (x-w)^T Omega^T Omega (x-w)
- Gamma decay during training
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils import train_test_split_jax
from prosemble.models import MRSLVQ_NG

# Load data
dataset = load_iris_jax()
X, y = dataset.input_data, dataset.labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split_jax(
    X, y, test_size=0.2, random_seed=42
)

print(f"Dataset: {X.shape}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Setup MRSLVQ_NG model with latent_dim for dimensionality reduction
model = MRSLVQ_NG(
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
print("\nTraining MRSLVQ_NG...")
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

# Omega matrix
print(f"\nOmega shape: {model.omega_matrix.shape}")
print(f"Lambda (relevance matrix):\n{model.lambda_matrix}")

# Class probabilities
proba = model.predict_proba(X_test[:5])
print(f"\nClass probabilities (first 5 samples):\n{proba}")
