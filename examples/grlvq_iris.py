"""
Generalized Relevance LVQ (GRLVQ) example using Iris Data with JAX.

This example demonstrates:
- Per-feature relevance weighting learned during training
- Relevance profile reveals which features matter most
- Weighted Euclidean distance: d(x,w) = sum_j lambda_j * (x_j - w_j)^2
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils import train_test_split_jax
from prosemble.models import GRLVQ

# Load data
dataset = load_iris_jax()
X, y = dataset.input_data, dataset.labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split_jax(
    X, y, test_size=0.2, random_seed=42
)

print(f"Dataset: {X.shape} ({X.shape[1]} features)")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Setup GRLVQ model
model = GRLVQ(
    n_prototypes_per_class=2,
    max_iter=100,
    lr=0.01,
    random_seed=42,
)

# Fit model
print("\nTraining GRLVQ...")
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

# Learned relevance profile
relevances = model.relevance_profile
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
print("\nLearned feature relevances:")
for name, rel in zip(feature_names, relevances):
    bar = '#' * int(rel * 50)
    print(f"  {name:15s}: {rel:.4f} {bar}")
