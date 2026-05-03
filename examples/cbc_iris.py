"""
Classification-By-Components (CBC) example using Iris Data with JAX.

This example demonstrates:
- Reasoning-based classification with CBC
- Components (unsupervised prototypes) + reasoning matrices
- Evidence-based predictions: positive, negative, and irrelevant evidence per class
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils import train_test_split_jax
from prosemble.models import CBC

# Load data
dataset = load_iris_jax()
X, y = dataset.input_data, dataset.labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split_jax(
    X, y, test_size=0.2, random_seed=42
)

print(f"Dataset: {X.shape}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Classes: {int(jnp.max(y)) + 1}")

# Setup CBC model
model = CBC(
    n_components=6,
    n_classes=3,
    max_iter=100,
    lr=0.01,
    sigma=1.0,
    random_seed=42,
)

# Fit model
print("\nTraining CBC...")
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

# Reasoning matrices
print(f"\nComponents shape: {model.components_.shape}")
print(f"Reasoning matrices shape: {model.reasonings_.shape}")

# Class probabilities (evidence-based)
proba = model.predict_proba(X_test[:5])
print(f"\nClass probabilities (first 5 samples):\n{proba}")
