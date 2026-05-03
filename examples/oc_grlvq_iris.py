"""
One-Class GRLVQ (OC-GRLVQ) example using Iris Data with JAX.

This example demonstrates:
- One-class classification with per-feature relevance weighting
- Learns which features best distinguish target from outliers
- Relevances satisfy: lambda_j >= 0 and sum(lambda) = 1
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.core.utils import train_test_split_jax
from prosemble.models import OCGRLVQ

# Load data
dataset = load_iris_jax()
X, y = dataset.input_data, dataset.labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split_jax(
    X, y, test_size=0.2, random_seed=42
)

target_label = 0
print(f"Dataset: {X.shape} ({X.shape[1]} features)")
print(f"Target class: {target_label}")

# Setup OC-GRLVQ model
model = OCGRLVQ(
    n_prototypes=3,
    target_label=target_label,
    beta=10.0,
    max_iter=100,
    lr=0.01,
    random_seed=42,
)

# Fit model
print("\nTraining OC-GRLVQ...")
model.fit(X_train, y_train)

# Results
print(f"Converged in {model.n_iter_} iterations")
print(f"Final loss: {model.loss_:.4f}")

# Predictions
test_preds = model.predict(X_test)
proba = model.predict_proba(X_test)
print(f"\nProbabilities shape: {proba.shape}")

# Learned relevance profile
relevances = model.relevance_profile
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
print("\nLearned feature relevances:")
for name, rel in zip(feature_names, relevances):
    bar = '#' * int(rel * 50)
    print(f"  {name:15s}: {rel:.4f} {bar}")
