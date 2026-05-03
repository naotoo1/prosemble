"""Tests for visualization module."""

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for CI
import matplotlib.pyplot as plt
import numpy as np
import pytest

from prosemble.datasets import load_iris_jax
from prosemble.core.vis import (
    plot_umatrix, plot_hit_map, plot_component_planes,
    plot_som_grid, plot_som_loss, plot_som_summary,
    plot_decision_boundary_2d, plot_prototype_trajectory,
    plot_lvq_summary, plot_neural_gas,
)


@pytest.fixture(scope="module")
def iris():
    dataset = load_iris_jax()
    return dataset.input_data, dataset.labels


@pytest.fixture(scope="module")
def fitted_som(iris):
    from prosemble.models import HeskesSOM
    X, _ = iris
    model = HeskesSOM(
        grid_height=3, grid_width=3, max_iter=20, random_seed=42
    )
    model.fit(X)
    return model


@pytest.fixture(scope="module")
def fitted_glvq(iris):
    from prosemble.models import GLVQ
    X, y = iris
    model = GLVQ(
        n_prototypes_per_class=1, max_iter=20, lr=0.01, random_seed=42
    )
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def fitted_ng(iris):
    from prosemble.models import NeuralGas
    X, _ = iris
    model = NeuralGas(n_prototypes=5, max_iter=20, random_seed=42)
    model.fit(X)
    return model


class TestSOMVisualization:
    def test_umatrix(self, fitted_som):
        fig = plot_umatrix(fitted_som)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_hit_map(self, fitted_som, iris):
        X, _ = iris
        fig = plot_hit_map(fitted_som, X)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_component_planes(self, fitted_som):
        fig = plot_component_planes(fitted_som)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_component_planes_with_names(self, fitted_som):
        fig = plot_component_planes(
            fitted_som,
            feature_names=["a", "b", "c", "d"],
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_som_grid(self, fitted_som, iris):
        X, y = iris
        fig = plot_som_grid(fitted_som, X, y)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_som_grid_no_data(self, fitted_som):
        fig = plot_som_grid(fitted_som)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_som_loss(self, fitted_som):
        fig = plot_som_loss(fitted_som)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_som_summary(self, fitted_som, iris):
        X, y = iris
        fig = plot_som_summary(fitted_som, X, y)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestLVQVisualization:
    def test_decision_boundary(self, fitted_glvq, iris):
        X, y = iris
        fig = plot_decision_boundary_2d(fitted_glvq, X, y, resolution=30)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_loss_trajectory(self, fitted_glvq):
        fig = plot_prototype_trajectory(fitted_glvq.loss_history_)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_lvq_summary(self, fitted_glvq, iris):
        X, y = iris
        fig = plot_lvq_summary(fitted_glvq, X, y)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestNeuralGasVisualization:
    def test_neural_gas_plot(self, fitted_ng, iris):
        X, y = iris
        fig = plot_neural_gas(fitted_ng, X, y)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_neural_gas_no_data(self, fitted_ng):
        fig = plot_neural_gas(fitted_ng)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestVisualizationErrors:
    def test_unfitted_som_raises(self):
        from prosemble.models import HeskesSOM
        model = HeskesSOM(grid_height=2, grid_width=2, max_iter=5)
        with pytest.raises(ValueError, match="not fitted"):
            plot_umatrix(model)

    def test_unfitted_lvq_raises(self):
        from prosemble.models import GLVQ
        model = GLVQ(n_prototypes_per_class=1, max_iter=5)
        with pytest.raises(ValueError, match="not fitted"):
            plot_decision_boundary_2d(model, np.zeros((10, 2)), np.zeros(10))
