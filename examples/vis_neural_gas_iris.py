"""Visualize Neural Gas topology and convergence on Iris dataset."""

import numpy as np
import matplotlib.pyplot as plt
from prosemble.datasets import load_iris_jax
from prosemble.models import NeuralGas
from prosemble.core.vis import plot_neural_gas, plot_prototype_trajectory

# Load data
dataset = load_iris_jax()
X = dataset.input_data
y = dataset.labels

# Fit Neural Gas
model = NeuralGas(
    n_prototypes=10,
    max_iter=100,
    lr_init=0.5,
    lr_final=0.01,
    lambda_init=5.0,
    lambda_final=0.01,
    random_seed=42,
)
model.fit(X)
print(f"Trained {model.n_iter_} iterations, loss={model.loss_:.4f}")
print(f"Unique clusters used: {len(np.unique(np.array(model.predict(X))))}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. Neural Gas topology with data overlay and Hebbian edges
plot_neural_gas(model, X, y, k_edges=2, ax=axes[0])
axes[0].set_title('Neural Gas Topology (PCA)')

# 2. Convergence curve
plot_prototype_trajectory(model.loss_history_, ax=axes[1])
axes[1].set_title('Convergence')

plt.suptitle('Neural Gas — Iris Dataset', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('vis_neural_gas_iris.png', dpi=150, bbox_inches='tight')
print("Saved vis_neural_gas_iris.png")
