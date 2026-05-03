"""PLVQ on Iris dataset.

Demonstrates Probabilistic LVQ: an MLP backbone with Gaussian
mixture soft assignment for probabilistic classification.
"""

import jax.numpy as jnp
from prosemble.datasets import load_iris_jax
from prosemble.models import PLVQ

# Load data
dataset = load_iris_jax()
X = dataset.input_data
y = dataset.labels

# Fit PLVQ
model = PLVQ(
    hidden_sizes=[10],
    latent_dim=2,
    activation='sigmoid',
    sigma=1.0,
    loss_type='rslvq',
    n_prototypes_per_class=2,
    max_iter=200,
    lr=0.01,
    random_seed=42,
)
model.fit(X, y)

print(f"Trained {model.n_iter_} epochs, loss={model.loss_:.4f}")

# Predict
preds = model.predict(X)
accuracy = jnp.mean(preds == y)
print(f"Training accuracy: {accuracy:.2%}")

# Probabilistic predictions
proba = model.predict_proba(X)
print(f"Class probabilities shape: {proba.shape}")
print(f"Mean confidence: {jnp.max(proba, axis=1).mean():.3f}")

# Latent space
latent = model.transform(X)
print(f"Latent shape: {latent.shape}")
