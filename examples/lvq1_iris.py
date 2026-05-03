"""
LVQ1 (Learning Vector Quantization 1) example using Iris Data with JAX.

This example demonstrates:
- Classic competitive learning (non-gradient)
- Winner-take-all: attract same-class prototype, repel different-class
- Simple, fast, and interpretable
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils import train_test_split_jax
from prosemble.models import LVQ1

# Load data
dataset = load_iris_jax()
X, y = dataset.input_data, dataset.labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split_jax(
    X, y, test_size=0.2, random_seed=42
)

print(f"Dataset: {X.shape}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Setup LVQ1 model
model = LVQ1(
    n_prototypes_per_class=2,
    max_iter=50,
    lr=0.1,
    random_seed=42,
)

# Fit model
print("\nTraining LVQ1...")
model.fit(X_train, y_train)

# Results
print(f"Trained for {model.n_iter_} iterations")

# Predictions
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_acc = float(jnp.mean(train_preds == y_train))
test_acc = float(jnp.mean(test_preds == y_test))

print(f"\nTrain accuracy: {train_acc:.2%}")
print(f"Test accuracy:  {test_acc:.2%}")

# Prototypes
print(f"\nPrototypes shape: {model.prototypes_.shape}")
print(f"Prototype labels: {model.prototype_labels_}")
print(f"\nPrototype positions:\n{model.prototypes_}")
