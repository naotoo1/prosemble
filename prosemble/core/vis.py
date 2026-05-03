"""
Visualization utilities for prosemble models.

Provides static plotting functions for:
- SOM / Heskes SOM: U-matrix, hit map, component planes, grid with data
- LVQ (GLVQ, GMLVQ, etc.): 2D decision boundaries, prototype plots
- Neural Gas: topology graph, Voronoi regions

All functions accept fitted model objects and return matplotlib Figure objects,
making them composable and easy to save/embed.
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# SOM Visualizations
# ---------------------------------------------------------------------------

def plot_umatrix(model, *, cmap="bone", figsize=(8, 6), ax=None):
    """Plot the U-matrix (unified distance matrix) for a SOM model.

    The U-matrix shows the average distance between each neuron and its
    direct grid neighbours. Dark regions indicate cluster boundaries;
    light regions indicate clusters.

    Parameters
    ----------
    model : KohonenSOM or HeskesSOM
        A fitted SOM model with ``prototypes_``, ``grid_height``, ``grid_width``.
    cmap : str
        Matplotlib colormap name.
    figsize : tuple
        Figure size if creating a new figure.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _check_som(model)
    prototypes = np.asarray(model.prototypes_)
    h, w = model.grid_height, model.grid_width
    grid = prototypes.reshape(h, w, -1)

    umat = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            neighbours = []
            if i > 0:
                neighbours.append(grid[i - 1, j])
            if i < h - 1:
                neighbours.append(grid[i + 1, j])
            if j > 0:
                neighbours.append(grid[i, j - 1])
            if j < w - 1:
                neighbours.append(grid[i, j + 1])
            dists = [np.linalg.norm(grid[i, j] - nb) for nb in neighbours]
            umat[i, j] = np.mean(dists)

    fig, ax = _get_ax(ax, figsize)
    im = ax.imshow(umat, cmap=cmap, interpolation="nearest", origin="upper")
    ax.set_title("U-Matrix")
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    fig.colorbar(im, ax=ax, label="Avg neighbour distance")
    return fig


def plot_hit_map(model, X, *, cmap="YlOrRd", figsize=(8, 6), ax=None):
    """Plot a hit map showing how many data points map to each neuron.

    Parameters
    ----------
    model : KohonenSOM or HeskesSOM
        A fitted SOM model.
    X : array-like of shape (n_samples, n_features)
        Input data.
    cmap : str
        Colormap.
    figsize : tuple
        Figure size.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _check_som(model)
    h, w = model.grid_height, model.grid_width
    labels = np.asarray(model.predict(jnp.asarray(X, dtype=jnp.float32)))
    hits = np.bincount(labels, minlength=h * w).reshape(h, w)

    fig, ax = _get_ax(ax, figsize)
    im = ax.imshow(hits, cmap=cmap, interpolation="nearest", origin="upper")
    # Annotate cells with counts
    for i in range(h):
        for j in range(w):
            ax.text(j, i, str(hits[i, j]), ha="center", va="center",
                    fontsize=8, color="black" if hits[i, j] < hits.max() * 0.7 else "white")
    ax.set_title("Hit Map")
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    fig.colorbar(im, ax=ax, label="Sample count")
    return fig


def plot_component_planes(model, *, feature_names=None, cmap="coolwarm",
                          figsize=None, cols=4):
    """Plot component planes — one heatmap per feature dimension.

    Parameters
    ----------
    model : KohonenSOM or HeskesSOM
        A fitted SOM model.
    feature_names : list of str, optional
        Feature names for subplot titles.
    cmap : str
        Colormap.
    figsize : tuple, optional
        If None, auto-calculated.
    cols : int
        Number of columns in the subplot grid.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _check_som(model)
    prototypes = np.asarray(model.prototypes_)
    h, w = model.grid_height, model.grid_width
    n_features = prototypes.shape[1]
    grid = prototypes.reshape(h, w, n_features)

    rows = int(np.ceil(n_features / cols))
    if figsize is None:
        figsize = (cols * 3, rows * 2.5)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        if idx < n_features:
            name = feature_names[idx] if feature_names else f"Feature {idx}"
            im = ax.imshow(grid[:, :, idx], cmap=cmap, interpolation="nearest",
                           origin="upper")
            ax.set_title(name, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Component Planes", fontsize=12)
    fig.tight_layout()
    return fig


def plot_som_grid(model, X=None, y=None, *, figsize=(8, 8), ax=None):
    """Plot SOM prototypes on a 2D grid with optional data overlay.

    Prototypes are projected to 2D via PCA. Grid connections are drawn.
    If labels ``y`` are provided, data points are coloured by class.

    Parameters
    ----------
    model : KohonenSOM or HeskesSOM
        A fitted SOM model.
    X : array-like, optional
        Data to overlay.
    y : array-like, optional
        Labels for colouring data points.
    figsize : tuple
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _check_som(model)
    prototypes = np.asarray(model.prototypes_)
    h, w = model.grid_height, model.grid_width

    # PCA to 2D
    need_pca = prototypes.shape[1] > 2
    if need_pca:
        pca = PCA(n_components=2)
        all_data = prototypes if X is None else np.vstack([prototypes, np.asarray(X)])
        pca.fit(all_data)
        proto_2d = pca.transform(prototypes)
    else:
        proto_2d = prototypes[:, :2]

    fig, ax = _get_ax(ax, figsize)

    # Draw grid connections
    segments = []
    for i in range(h):
        for j in range(w):
            idx = i * w + j
            if j < w - 1:
                segments.append([proto_2d[idx], proto_2d[idx + 1]])
            if i < h - 1:
                segments.append([proto_2d[idx], proto_2d[idx + w]])
    lc = LineCollection(segments, colors="gray", linewidths=0.8, alpha=0.6)
    ax.add_collection(lc)

    # Overlay data
    if X is not None:
        X_np = np.asarray(X)
        X_2d = pca.transform(X_np) if need_pca else X_np[:, :2]
        if y is not None:
            y_np = np.asarray(y)
            ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_np, cmap="tab10",
                       s=15, alpha=0.4, zorder=1)
        else:
            ax.scatter(X_2d[:, 0], X_2d[:, 1], c="lightblue",
                       s=15, alpha=0.4, zorder=1)

    # Plot prototypes
    ax.scatter(proto_2d[:, 0], proto_2d[:, 1], c="red", s=60,
               edgecolors="black", linewidths=1, zorder=3, label="Prototypes")

    ax.set_title("SOM Grid (PCA projection)" if need_pca else "SOM Grid")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)
    return fig


def plot_som_loss(model, *, figsize=(8, 4), ax=None):
    """Plot the SOM training loss / energy history.

    Parameters
    ----------
    model : KohonenSOM or HeskesSOM
        A fitted SOM model.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _check_som(model)
    history = np.asarray(model.loss_history_)
    if hasattr(model, 'n_iter_'):
        history = history[:model.n_iter_]

    fig, ax = _get_ax(ax, figsize)
    ax.plot(range(1, len(history) + 1), history, "b-", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss / Energy")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)
    return fig


# ---------------------------------------------------------------------------
# LVQ / Supervised Prototype Visualizations
# ---------------------------------------------------------------------------

def plot_decision_boundary_2d(model, X, y, *, resolution=200,
                               cmap="RdYlBu", figsize=(8, 6), ax=None):
    """Plot 2D decision boundaries for a supervised prototype model.

    If data has > 2 features, PCA is used to project to 2D and prototypes
    are shown in the same PCA space.

    Parameters
    ----------
    model : GLVQ, GMLVQ, GRLVQ, CELVQ, CBC, etc.
        A fitted supervised prototype model with ``predict()``.
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
        True labels.
    resolution : int
        Grid resolution for the boundary mesh.
    cmap : str
        Colormap for decision regions.
    figsize : tuple
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _check_supervised(model)
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    prototypes = np.asarray(model.prototypes_)
    proto_labels = np.asarray(model.prototype_labels_)

    need_pca = X_np.shape[1] > 2
    if need_pca:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_np)
        proto_2d = pca.transform(prototypes)
    else:
        X_2d = X_np[:, :2]
        proto_2d = prototypes[:, :2]

    # Create mesh
    margin = 0.5
    x_min, x_max = X_2d[:, 0].min() - margin, X_2d[:, 0].max() + margin
    y_min, y_max = X_2d[:, 1].min() - margin, X_2d[:, 1].max() + margin
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    mesh_points = np.c_[xx.ravel(), yy.ravel()]

    # Predict on mesh
    if need_pca:
        mesh_full = pca.inverse_transform(mesh_points)
    else:
        mesh_full = mesh_points
    mesh_preds = np.asarray(model.predict(jnp.asarray(mesh_full, dtype=jnp.float32)))
    zz = mesh_preds.reshape(xx.shape)

    fig, ax = _get_ax(ax, figsize)

    # Decision regions
    n_classes = len(np.unique(y_np))
    ax.contourf(xx, yy, zz, levels=np.arange(n_classes + 1) - 0.5,
                cmap=cmap, alpha=0.3)
    ax.contour(xx, yy, zz, levels=np.arange(n_classes + 1) - 0.5,
               colors="gray", linewidths=0.5, alpha=0.5)

    # Data points
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_np, cmap=cmap,
                         s=20, edgecolors="black", linewidths=0.3, alpha=0.7)

    # Prototypes
    ax.scatter(proto_2d[:, 0], proto_2d[:, 1], c=proto_labels, cmap=cmap,
               s=200, marker="*", edgecolors="black", linewidths=1.5, zorder=5)

    ax.set_title("Decision Boundaries" + (" (PCA)" if need_pca else ""))
    ax.set_xlabel("PC1" if need_pca else "Feature 1")
    ax.set_ylabel("PC2" if need_pca else "Feature 2")
    ax.grid(True, alpha=0.2)
    return fig


def plot_prototype_trajectory(loss_history, *, figsize=(8, 4), ax=None):
    """Plot training loss curve for a supervised model.

    Parameters
    ----------
    loss_history : array-like
        Loss values per epoch (e.g., ``model.loss_history_``).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    history = np.asarray(loss_history)
    fig, ax = _get_ax(ax, figsize)
    ax.plot(range(1, len(history) + 1), history, "b-", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)
    return fig


def plot_relevance_matrix(model, *, feature_names=None, figsize=(8, 6), ax=None):
    """Plot the learned relevance/omega matrix for GMLVQ or GRLVQ.

    For GRLVQ, plots a bar chart of relevance weights.
    For GMLVQ, plots the full Lambda = Omega^T Omega matrix as a heatmap.

    Parameters
    ----------
    model : GRLVQ or GMLVQ
        A fitted model with ``lambda_matrix_`` or ``relevances_``.
    feature_names : list of str, optional
    figsize : tuple
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = _get_ax(ax, figsize)

    if hasattr(model, "lambda_matrix_"):
        # GMLVQ — full matrix
        lam = np.asarray(model.lambda_matrix_)
        im = ax.imshow(lam, cmap="viridis", interpolation="nearest")
        ax.set_title("Relevance Matrix (Λ = ΩᵀΩ)")
        if feature_names:
            ax.set_xticks(range(len(feature_names)))
            ax.set_xticklabels(feature_names, rotation=45, ha="right")
            ax.set_yticks(range(len(feature_names)))
            ax.set_yticklabels(feature_names)
        fig.colorbar(im, ax=ax)
    elif hasattr(model, "relevances_"):
        # GRLVQ — diagonal
        rel = np.asarray(model.relevances_)
        labels = feature_names or [f"F{i}" for i in range(len(rel))]
        ax.bar(labels, rel, color="steelblue", edgecolor="black")
        ax.set_title("Feature Relevances")
        ax.set_ylabel("Relevance weight")
        if feature_names:
            ax.set_xticklabels(labels, rotation=45, ha="right")
    else:
        ax.text(0.5, 0.5, "No relevance data found",
                transform=ax.transAxes, ha="center", va="center")

    return fig


# ---------------------------------------------------------------------------
# Neural Gas Visualizations
# ---------------------------------------------------------------------------

def plot_neural_gas(model, X=None, y=None, *, k_edges=3,
                    figsize=(8, 8), ax=None):
    """Plot Neural Gas prototypes with competitive Hebbian connections.

    Edges connect prototypes that are first and second BMU for any
    data point (approximate topology).

    Parameters
    ----------
    model : NeuralGas
        A fitted Neural Gas model.
    X : array-like, optional
        Data to overlay and compute edges from.
    y : array-like, optional
        Labels for data colouring.
    k_edges : int
        Connect prototypes that are within k-nearest of each other
        for any sample.
    figsize : tuple
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    _check_fitted(model)
    prototypes = np.asarray(model.prototypes_)

    need_pca = prototypes.shape[1] > 2
    if need_pca:
        pca = PCA(n_components=2)
        all_data = prototypes if X is None else np.vstack([prototypes, np.asarray(X)])
        pca.fit(all_data)
        proto_2d = pca.transform(prototypes)
    else:
        proto_2d = prototypes[:, :2]

    fig, ax = _get_ax(ax, figsize)

    # Compute edges from data
    if X is not None:
        X_np = np.asarray(X)
        X_2d = pca.transform(X_np) if need_pca else X_np[:, :2]

        # Find edges: connect BMU pairs
        dists = np.asarray(
            squared_euclidean_distance_matrix_np(X_np, prototypes)
        )
        ranks = np.argsort(dists, axis=1)
        edges = set()
        for i in range(len(X_np)):
            for k in range(min(k_edges, ranks.shape[1] - 1)):
                a, b = int(ranks[i, k]), int(ranks[i, k + 1])
                edges.add((min(a, b), max(a, b)))

        segments = [[proto_2d[a], proto_2d[b]] for a, b in edges]
        lc = LineCollection(segments, colors="gray", linewidths=0.8, alpha=0.5)
        ax.add_collection(lc)

        # Data overlay
        if y is not None:
            ax.scatter(X_2d[:, 0], X_2d[:, 1], c=np.asarray(y),
                       cmap="tab10", s=15, alpha=0.4, zorder=1)
        else:
            ax.scatter(X_2d[:, 0], X_2d[:, 1], c="lightblue",
                       s=15, alpha=0.4, zorder=1)

    ax.scatter(proto_2d[:, 0], proto_2d[:, 1], c="red", s=80,
               edgecolors="black", linewidths=1.2, zorder=3, label="Prototypes")

    ax.set_title("Neural Gas" + (" (PCA)" if need_pca else ""))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)
    return fig


# ---------------------------------------------------------------------------
# Multi-panel convenience
# ---------------------------------------------------------------------------

def plot_som_summary(model, X, y=None, *, feature_names=None, figsize=(16, 10)):
    """Create a 4-panel SOM summary: U-matrix, hit map, grid, loss.

    Parameters
    ----------
    model : KohonenSOM or HeskesSOM
    X : array-like
    y : array-like, optional
    feature_names : list, optional
    figsize : tuple

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    plot_umatrix(model, ax=axes[0, 0])
    plot_hit_map(model, X, ax=axes[0, 1])
    plot_som_grid(model, X, y, ax=axes[1, 0])
    plot_som_loss(model, ax=axes[1, 1])

    fig.suptitle("SOM Summary", fontsize=14)
    fig.tight_layout()
    return fig


def plot_lvq_summary(model, X, y, *, figsize=(14, 5)):
    """Create a 2-panel LVQ summary: decision boundary + loss curve.

    Parameters
    ----------
    model : GLVQ, GMLVQ, etc.
    X : array-like
    y : array-like
    figsize : tuple

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_decision_boundary_2d(model, X, y, ax=axes[0])

    if hasattr(model, "loss_history_"):
        plot_prototype_trajectory(model.loss_history_, ax=axes[1])
    else:
        axes[1].text(0.5, 0.5, "No loss history", transform=axes[1].transAxes,
                     ha="center", va="center")

    fig.suptitle("LVQ Summary", fontsize=14)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_ax(ax, figsize):
    """Return (fig, ax) — create a new figure if ax is None."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def _check_som(model):
    """Verify model is a fitted SOM."""
    if not hasattr(model, "prototypes_") or model.prototypes_ is None:
        raise ValueError("Model is not fitted. Call .fit() first.")
    if not hasattr(model, "grid_height"):
        raise ValueError("Model does not appear to be a SOM (no grid_height).")


def _check_supervised(model):
    """Verify model is a fitted supervised prototype model."""
    if not hasattr(model, "prototypes_") or model.prototypes_ is None:
        raise ValueError("Model is not fitted. Call .fit() first.")
    if not hasattr(model, "prototype_labels_") or model.prototype_labels_ is None:
        raise ValueError("Model does not have prototype_labels_.")


def _check_fitted(model):
    """Verify model is fitted."""
    if not hasattr(model, "prototypes_") or model.prototypes_ is None:
        raise ValueError("Model is not fitted. Call .fit() first.")


def squared_euclidean_distance_matrix_np(X, Y):
    """Numpy version of squared euclidean distance for visualization."""
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    return XX + YY.T - 2 * X @ Y.T
