"""
RSLVQ with Neural Gas Cooperation (RSLVQ_NG) example using Iris Data with JAX.

This example demonstrates:
- RSLVQ probabilistic loss with NG neighborhood cooperation
- Gamma decay: neighborhood range shrinks during training
- When gamma -> 0, recovers standard RSLVQ behavior
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils import train_test_split_jax
from prosemble.models import RSLVQ_NG

# Load data
dataset = load_iris_jax()
X, y = dataset.input_data, dataset.labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split_jax(
    X, y, test_size=0.2, random_seed=42
)

print(f"Dataset: {X.shape} ({X.shape[1]} features)")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Setup RSLVQ_NG model
model = RSLVQ_NG(
    sigma=1.0,
    n_prototypes_per_class=3,
    max_iter=200,
    lr=0.01,
    gamma_init=5.0,       # Start with wide neighborhood cooperation
    gamma_final=0.01,     # End near standard RSLVQ behavior
    random_seed=42,
)

# Fit model
print("\nTraining RSLVQ_NG...")
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

# Class probabilities
proba = model.predict_proba(X_test[:5])
print(f"\nClass probabilities (first 5 samples):\n{proba}")
