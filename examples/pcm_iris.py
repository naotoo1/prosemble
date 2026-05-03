"""Possibilistic C-Means clustering example using Iris Data with JAX."""

import jax.numpy as jnp
from prosemble.datasets import DATA_JAX
from prosemble.core.utils_jax import train_test_split_jax
from prosemble.models.jax import PCM_JAX

# Load data (JAX arrays directly)
data = DATA_JAX(random=42)
dataset = data.iris
X, y = dataset.input_data, dataset.labels

# Train/test split (JAX-native)
X_train, X_test, y_train, y_test = train_test_split_jax(X, y, test_size=0.2, random_seed=42)

# Setup model
pcm = PCM_JAX(n_clusters=3, fuzzifier=2.0, k=1.0, max_iter=100, epsilon=1e-5,
              init_method='fcm', random_seed=42)

# Fit model
pcm.fit(X_train)

# Results
print(pcm.get_objective_history())
print(pcm.predict(X_train))
print(pcm.predict(X_test))
print(pcm.centroids_)
