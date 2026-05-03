"""
Generalized Learning Vector Quantization (GLVQ) example using Iris Data with JAX.

This example demonstrates:
- Supervised prototype-based classification with GLVQ
- Training with GLVQ loss: mu = (d+ - d-)/(d+ + d-)
- Prediction via Winner-Takes-All Competition (WTAC)
- Configurable prototype initialization methods
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils import train_test_split_jax
from prosemble.models import GLVQ

# Load data
dataset = load_iris_jax()
X, y = dataset.input_data, dataset.labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split_jax(
    X, y, test_size=0.2, random_seed=42
)

print(f"Dataset: {X.shape}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Classes: {jnp.unique(y)}")

# Setup GLVQ model with unequal prototype distribution
# Iris has 3 classes (0, 1, 2): assign 3 prototypes to class 0,
# 2 to class 1, and 1 to class 2
#
# Available prototype initializers (pass as string):
#   'stratified_random' - random samples per class (default)
#   'class_mean'        - class centroids
#   'class_conditional_mean' - class centroids replicated per n_per_class
#   'stratified_noise'  - random samples + Gaussian noise
#   'random_normal'     - random normal initialization
#   'uniform'           - random uniform initialization
#   'zeros'             - zero initialization
# Or pass a custom callable: fn(X, y, n_per_class, key) -> (prototypes, labels)
model = GLVQ(
    n_prototypes_per_class={0: 3, 1: 2, 2: 1},
    prototypes_initializer='class_conditional_mean',
    max_iter=100,
    lr=0.01,
    random_seed=42,
)

# Fit model
print("\nTraining GLVQ...")
model.fit(X_train, y_train)

# Results
print(f"Converged in {model.n_iter_} iterations")
print(f"Final loss: {model.loss_:.4f}")
print(f"Loss history (last 5): {model.loss_history_[-5:]}")

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

# Class probabilities
proba = model.predict_proba(X_test[:5])
print(f"\nClass probabilities (first 5 samples):\n{proba}")
