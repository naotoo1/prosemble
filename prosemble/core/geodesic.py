"""Geodesic interpolation and boundary visualization for Riemannian models.

Computes geodesic paths between prototypes on Riemannian manifolds for
decision boundary visualization and model interpretation.

On curved manifolds, the decision boundary (locus where d(x, w_i) = d(x, w_j))
is not a hyperplane — it's a curved surface. Geodesic interpolation allows
us to:
1. Visualize the path between prototypes along the manifold.
2. Find the approximate decision boundary point along geodesics.
3. Compute prototype midpoints respecting manifold geometry.

References
----------
.. [1] Absil, P.-A., Mahony, R., & Sepulchre, R. (2008). Optimization
       Algorithms on Matrix Manifolds. Princeton University Press.
"""

import jax.numpy as jnp


def geodesic_interpolation(manifold, point_a, point_b, n_points=50):
    """Compute a geodesic path between two points on a manifold.

    Uses the exponential map to trace the shortest path (geodesic)
    between two manifold points:

    .. math::

        \\gamma(t) = \\text{Exp}_{w_a}(t \\cdot \\text{Log}_{w_a}(w_b)),
        \\quad t \\in [0, 1]

    Parameters
    ----------
    manifold : SO, SPD, or Grassmannian
        Riemannian manifold instance with ``exp_map`` and ``log_map``.
    point_a : array
        Starting point on the manifold.
    point_b : array
        End point on the manifold.
    n_points : int
        Number of interpolation points along the geodesic.
        Default: 50.

    Returns
    -------
    path : array of shape (n_points, ...)
        Points along the geodesic from point_a to point_b.
        Each point lies on the manifold.
    """
    # Compute the tangent vector from a to b
    tangent = manifold.log_map(point_a, point_b)

    # Interpolate along the geodesic
    t_values = jnp.linspace(0.0, 1.0, n_points)
    path = []
    for t in t_values:
        point = manifold.exp_map(point_a, t * tangent)
        path.append(point)

    return jnp.stack(path, axis=0)


def geodesic_midpoint(manifold, point_a, point_b):
    """Compute the geodesic midpoint between two manifold points.

    The midpoint is at t = 0.5 along the geodesic:

    .. math::

        m = \\text{Exp}_{w_a}(0.5 \\cdot \\text{Log}_{w_a}(w_b))

    Parameters
    ----------
    manifold : SO, SPD, or Grassmannian
        Riemannian manifold instance.
    point_a : array
        First manifold point.
    point_b : array
        Second manifold point.

    Returns
    -------
    midpoint : array
        Geodesic midpoint, guaranteed to lie on the manifold.
    """
    tangent = manifold.log_map(point_a, point_b)
    return manifold.exp_map(point_a, 0.5 * tangent)


def decision_boundary_point(manifold, proto_a, proto_b, n_search=100):
    """Find the decision boundary point along the geodesic between prototypes.

    On curved manifolds, the equidistant point (where d(x, w_a) = d(x, w_b))
    is not necessarily at t = 0.5. This function searches along the geodesic
    for the point where distances to both prototypes are equal.

    Parameters
    ----------
    manifold : SO, SPD, or Grassmannian
        Riemannian manifold instance.
    proto_a : array
        First prototype (on manifold).
    proto_b : array
        Second prototype (on manifold).
    n_search : int
        Number of candidate points to evaluate. Default: 100.

    Returns
    -------
    boundary_point : array
        Point on the geodesic where d(point, proto_a) = d(point, proto_b).
    t_boundary : float
        Parameter value t in [0, 1] where the boundary lies.
        t = 0.5 for symmetric manifolds, may differ on curved spaces.
    """
    tangent = manifold.log_map(proto_a, proto_b)
    t_values = jnp.linspace(0.0, 1.0, n_search)

    # Compute distance difference at each point along the geodesic
    def distance_diff(t):
        point = manifold.exp_map(proto_a, t * tangent)
        d_a = manifold.distance(point, proto_a)
        d_b = manifold.distance(point, proto_b)
        return jnp.abs(d_a - d_b)

    diffs = jnp.array([distance_diff(t) for t in t_values])
    best_idx = jnp.argmin(diffs)
    t_boundary = t_values[best_idx]
    boundary_point = manifold.exp_map(proto_a, t_boundary * tangent)

    return boundary_point, float(t_boundary)


def prototype_geodesic_distances(manifold, prototypes, proto_labels):
    """Compute pairwise geodesic distances between prototypes.

    Parameters
    ----------
    manifold : SO, SPD, or Grassmannian
        Riemannian manifold instance.
    prototypes : array of shape (n_prototypes, ...)
        Prototype points on the manifold (flattened).
    proto_labels : array of shape (n_prototypes,)
        Class labels for prototypes.

    Returns
    -------
    distances : array of shape (n_prototypes, n_prototypes)
        Pairwise geodesic distance matrix.
    """
    n = prototypes.shape[0]
    distances = jnp.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = manifold.distance(prototypes[i], prototypes[j])
            distances = distances.at[i, j].set(d)
            distances = distances.at[j, i].set(d)

    return distances


def inter_class_geodesics(manifold, prototypes, proto_labels, n_points=50):
    """Compute geodesic paths between all inter-class prototype pairs.

    Useful for visualizing decision boundaries: the boundary between
    two classes lies somewhere along the geodesic connecting their
    closest prototypes.

    Parameters
    ----------
    manifold : SO, SPD, or Grassmannian
        Riemannian manifold instance.
    prototypes : array of shape (n_prototypes, ...)
        Prototype positions on manifold.
    proto_labels : array of shape (n_prototypes,)
        Class labels.
    n_points : int
        Points per geodesic path. Default: 50.

    Returns
    -------
    geodesics : list of dict
        Each dict contains:
        - 'path': array of shape (n_points, ...) — geodesic path
        - 'proto_a_idx': int — index of first prototype
        - 'proto_b_idx': int — index of second prototype
        - 'class_a': int — class of first prototype
        - 'class_b': int — class of second prototype
        - 'boundary_t': float — approximate boundary location
    """
    import numpy as np

    n = prototypes.shape[0]
    labels_np = np.asarray(proto_labels)
    geodesics = []

    for i in range(n):
        for j in range(i + 1, n):
            if labels_np[i] != labels_np[j]:
                path = geodesic_interpolation(
                    manifold, prototypes[i], prototypes[j], n_points
                )
                _, t_boundary = decision_boundary_point(
                    manifold, prototypes[i], prototypes[j]
                )
                geodesics.append({
                    'path': path,
                    'proto_a_idx': i,
                    'proto_b_idx': j,
                    'class_a': int(labels_np[i]),
                    'class_b': int(labels_np[j]),
                    'boundary_t': t_boundary,
                })

    return geodesics
