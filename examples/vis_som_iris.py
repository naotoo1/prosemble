"""Visualize SOM / Heskes SOM on Iris dataset."""

from prosemble.datasets import load_iris_jax
from prosemble.models import HeskesSOM
from prosemble.core.vis import (
    plot_umatrix, plot_hit_map, plot_component_planes,
    plot_som_grid, plot_som_loss, plot_som_summary,
)

# Load data
dataset = load_iris_jax()
X = dataset.input_data
y = dataset.target_data

# Fit
model = HeskesSOM(
    grid_height=5, grid_width=5, max_iter=100,
    sigma_final=0.5, random_seed=42,
)
model.fit(X)
print(f"Converged in {model.n_iter_} iterations, energy={model.loss_:.2f}")

# Individual plots
fig1 = plot_umatrix(model)
fig2 = plot_hit_map(model, X)
fig3 = plot_component_planes(
    model,
    feature_names=["sepal_len", "sepal_wid", "petal_len", "petal_wid"],
)
fig4 = plot_som_grid(model, X, y)
fig5 = plot_som_loss(model)

# All-in-one summary
fig6 = plot_som_summary(model, X, y)

import matplotlib.pyplot as plt
fig6.savefig('vis_som_iris.png', dpi=150, bbox_inches='tight')
print("Saved vis_som_iris.png")
