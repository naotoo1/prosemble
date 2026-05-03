"""
Supervised Relevance Neural Gas (SRNG) example using Iris Data with JAX.

This example demonstrates:
- Neighborhood cooperation: all same-class prototypes participate in
  learning, weighted by rank (unlike GLVQ which only uses the closest)
- Per-feature relevance weighting learned during training
- Gamma decay: neighborhood range shrinks during training
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils import train_test_split_jax
from prosemble.models import SRNG

# Load data
dataset = load_iris_jax()
X, y = dataset.input_data, dataset.labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split_jax(
    X, y, test_size=0.2, random_seed=42
)

print(f"Dataset: {X.shape} ({X.shape[1]} features)")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Setup SRNG model
model = SRNG(
    n_prototypes_per_class=3,
    max_iter=200,
    lr=0.01,
    gamma_init=5.0,       # Start with wide neighborhood cooperation
    gamma_final=0.01,     # End near standard GRLVQ behavior
    random_seed=42,
)

# Fit model
print("\nTraining SRNG...")
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

# Learned relevance profile
relevances = model.relevance_profile
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
print("\nLearned feature relevances:")
for name, rel in zip(feature_names, relevances):
    bar = '#' * int(rel * 50)
    print(f"  {name:15s}: {rel:.4f} {bar}")
