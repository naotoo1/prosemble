"""
Example: Possibilistic Fuzzy C-Means (PFCM) clustering with JAX on Iris dataset.

PFCM combines fuzzy membership (U) and typicality (T) to handle both overlapping
clusters and outliers effectively.
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.models import PFCM

# Load Iris dataset (JAX arrays directly)
dataset = load_iris_jax()
X_jax, y = dataset.input_data, dataset.labels

print("Running PFCM on Iris dataset...")
print(f"Data shape: {X_jax.shape}")
print(f"Number of clusters: 3")
print()

# Create and fit PFCM model with visualization
model = PFCM(
    n_clusters=3,
    fuzzifier=2.0,
    eta=2.0,
    a=1.0,
    b=1.0,
    k=1.0,
    max_iter=100,
    epsilon=1e-5,
    init_method='fcm',
    random_seed=42,
    plot_steps=True,
    show_confidence=True,
    show_pca_variance=True,
    save_plot_path='pfcm_iris_final.png'
)

model.fit(X_jax)

print(f"Training completed in {model.n_iter_} iterations")
print(f"Final objective: {model.objective_:.4f}")
print(f"Centroids shape: {model.centroids_.shape}")
print(f"Membership (U) shape: {model.U_.shape}")
print(f"Typicality (T) shape: {model.T_.shape}")
print()

# Get predictions
labels = model.predict(X_jax)
U = model.predict_proba(X_jax)
T = model.predict_typicality(X_jax)

print(f"Predicted labels shape: {labels.shape}")
print(f"Unique labels: {jnp.unique(labels)}")
print()

# Show membership and typicality statistics
print("Membership (U) statistics:")
print(f"  Mean: {jnp.mean(U):.4f}")
print(f"  Row sums (should be ~1.0): {jnp.mean(jnp.sum(U, axis=1)):.4f}")
print()

print("Typicality (T) statistics:")
print(f"  Mean: {jnp.mean(T):.4f}")
print(f"  Row sums (not necessarily 1.0): {jnp.mean(jnp.sum(T, axis=1)):.4f}")
print()

# Show objective history
objectives = model.get_objective_history()
print(f"Objective function history (first 5): {objectives[:5]}")
print(f"Objective function history (last 5): {objectives[-5:]}")
