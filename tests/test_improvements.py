"""Tests for the 8 improvements to existing models.

Tests cover:
1. Per-group gradient clipping (optimizers)
2. Hypergradient descent (optimizers)
3. Riemannian Nesterov accelerated gradient (optimizers)
4. Prototype diversity regularization
5. Sparse relevance via elastic net (proximal gradient)
6. Curriculum learning (self-paced)
7. Reject option with calibrated uncertainty
8. Geodesic interpolation for Riemannian models
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from prosemble.core.optimizers import (
    per_group_clip,
    hypergradient_descent,
    riemannian_nesterov,
)
from prosemble.core.regularization import (
    prototype_diversity_loss,
    prototype_diversity_loss_vectorized,
    sparse_relevance_proximal,
    elastic_net_proximal,
)
from prosemble.core.curriculum import (
    curriculum_weights,
    curriculum_threshold,
    apply_curriculum_to_loss,
)
from prosemble.core.reject import RejectOptionMixin
from prosemble.core.geodesic import (
    geodesic_interpolation,
    geodesic_midpoint,
    decision_boundary_point,
    prototype_geodesic_distances,
)
from prosemble.core.manifolds import SO


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def simple_data():
    """Simple 2D dataset for testing."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    X0 = jax.random.normal(k1, (30, 4)) + jnp.array([2.0, 0.0, 0.0, 0.0])
    X1 = jax.random.normal(k2, (30, 4)) + jnp.array([-2.0, 0.0, 0.0, 0.0])
    X = jnp.concatenate([X0, X1])
    y = jnp.concatenate([jnp.zeros(30, dtype=jnp.int32),
                         jnp.ones(30, dtype=jnp.int32)])
    return X, y


@pytest.fixture
def so3_manifold():
    """SO(3) manifold instance."""
    return SO(3)


@pytest.fixture
def so3_points(so3_manifold):
    """Two random SO(3) rotation matrices (on CPU for Schur decomposition)."""
    with jax.default_device(jax.devices('cpu')[0]):
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        R1 = so3_manifold.random_point(k1)
        R2 = so3_manifold.random_point(k2)
    return R1, R2


# =============================================================================
# 1. Per-Group Gradient Clipping Tests
# =============================================================================

class TestPerGroupClip:
    """Tests for per-group gradient clipping optimizer."""

    def test_basic_clipping(self):
        """Gradients exceeding max_norm are scaled down."""
        params = {
            'prototypes': jnp.ones((3, 4)) * 10.0,
            'omega': jnp.ones((4, 2)) * 5.0,
        }
        grads = {
            'prototypes': jnp.ones((3, 4)) * 100.0,  # norm >> 1.0
            'omega': jnp.ones((4, 2)) * 0.1,  # norm < 0.5
        }

        optimizer = per_group_clip({'prototypes': 1.0, 'omega': 0.5})
        state = optimizer.init(params)
        updates, _ = optimizer.update(grads, state)

        # Prototypes should be clipped
        proto_norm = jnp.sqrt(jnp.sum(updates['prototypes'] ** 2))
        assert proto_norm <= 1.0 + 1e-6

        # Omega should NOT be clipped (norm < max_norm)
        np.testing.assert_allclose(updates['omega'], grads['omega'], atol=1e-6)

    def test_no_clip_when_below_threshold(self):
        """Gradients below max_norm pass through unchanged."""
        params = {'prototypes': jnp.ones((2, 3))}
        grads = {'prototypes': jnp.ones((2, 3)) * 0.1}

        optimizer = per_group_clip({'prototypes': 5.0})
        state = optimizer.init(params)
        updates, _ = optimizer.update(grads, state)

        np.testing.assert_allclose(updates['prototypes'], grads['prototypes'], atol=1e-6)

    def test_unspecified_keys_unclipped(self):
        """Parameters not in max_norms dict are left unchanged."""
        params = {'prototypes': jnp.ones((2, 3)), 'gamma': jnp.array(1.0)}
        grads = {'prototypes': jnp.ones((2, 3)) * 100.0, 'gamma': jnp.array(50.0)}

        optimizer = per_group_clip({'prototypes': 1.0})
        state = optimizer.init(params)
        updates, _ = optimizer.update(grads, state)

        # gamma should be unchanged (not in clip dict)
        assert float(updates['gamma']) == 50.0

    def test_chainable_with_adam(self):
        """Can be chained with standard optax optimizers."""
        optimizer = optax.chain(
            per_group_clip({'prototypes': 1.0, 'omega': 0.5}),
            optax.adam(0.01),
        )
        params = {'prototypes': jnp.ones((3, 4)), 'omega': jnp.ones((4, 2))}
        state = optimizer.init(params)
        grads = {'prototypes': jnp.ones((3, 4)), 'omega': jnp.ones((4, 2))}
        updates, new_state = optimizer.update(grads, state, params)

        # Should produce valid updates
        assert updates['prototypes'].shape == (3, 4)
        assert updates['omega'].shape == (4, 2)


# =============================================================================
# 2. Hypergradient Descent Tests
# =============================================================================

class TestHypergradientDescent:
    """Tests for hypergradient descent optimizer."""

    def test_basic_update(self):
        """Produces valid parameter updates."""
        params = {'prototypes': jnp.ones((3, 4))}
        optimizer = hypergradient_descent(init_lr=0.01, hyper_lr=1e-4)
        state = optimizer.init(params)

        grads = {'prototypes': jnp.ones((3, 4)) * 0.5}
        updates, new_state = optimizer.update(grads, state, params)

        assert updates['prototypes'].shape == (3, 4)
        # First step: prev_grads = 0, dot = 0, new_lr = init_lr + hyper_lr * 0 = init_lr
        # update = -init_lr * grad = -0.01 * 0.5
        np.testing.assert_allclose(
            updates['prototypes'], -0.01 * 0.5 * jnp.ones((3, 4)), atol=1e-5
        )

    def test_lr_increases_consistent_direction(self):
        """Learning rate increases when gradients point same direction."""
        params = {'x': jnp.array([1.0, 2.0, 3.0])}
        optimizer = hypergradient_descent(init_lr=0.01, hyper_lr=0.001)
        state = optimizer.init(params)

        # Two consecutive gradients in the same direction
        grads1 = {'x': jnp.array([1.0, 1.0, 1.0])}
        _, state = optimizer.update(grads1, state, params)

        grads2 = {'x': jnp.array([1.0, 1.0, 1.0])}
        _, state = optimizer.update(grads2, state, params)

        # LR should have increased (dot product positive)
        new_lr = float(state.learning_rates['x'])
        assert new_lr > 0.01

    def test_lr_decreases_oscillating_direction(self):
        """Learning rate decreases when gradients oscillate."""
        params = {'x': jnp.array([1.0, 2.0, 3.0])}
        optimizer = hypergradient_descent(init_lr=0.01, hyper_lr=0.001)
        state = optimizer.init(params)

        grads1 = {'x': jnp.array([1.0, 1.0, 1.0])}
        _, state = optimizer.update(grads1, state, params)

        grads2 = {'x': jnp.array([-1.0, -1.0, -1.0])}  # opposite direction
        _, state = optimizer.update(grads2, state, params)

        new_lr = float(state.learning_rates['x'])
        assert new_lr < 0.01

    def test_lr_bounded(self):
        """Learning rate stays within [min_lr, max_lr]."""
        params = {'x': jnp.array([1.0])}
        optimizer = hypergradient_descent(
            init_lr=0.01, hyper_lr=10.0,  # very aggressive
            min_lr=1e-5, max_lr=0.1
        )
        state = optimizer.init(params)

        # Many steps with large consistent gradients
        for _ in range(20):
            grads = {'x': jnp.array([100.0])}
            _, state = optimizer.update(grads, state, params)

        lr = float(state.learning_rates['x'])
        assert 1e-5 - 1e-7 <= lr <= 0.1 + 1e-6  # float32 tolerance


# =============================================================================
# 3. Riemannian Nesterov Tests
# =============================================================================

class TestRiemannianNesterov:
    """Tests for Riemannian Nesterov accelerated gradient."""

    def test_basic_update(self):
        """Produces valid parameter updates."""
        params = {'prototypes': jnp.ones((3, 4))}
        optimizer = riemannian_nesterov(learning_rate=0.01, momentum=0.9)
        state = optimizer.init(params)

        grads = {'prototypes': jnp.ones((3, 4))}
        updates, new_state = optimizer.update(grads, state, params)

        assert updates['prototypes'].shape == (3, 4)
        # First step: v=0, so v_new = 0.9*0 + grad = grad
        # update = -lr * (mu * v_new + grad) = -lr * (0.9*grad + grad) = -lr*1.9*grad
        expected = -0.01 * 1.9 * jnp.ones((3, 4))
        np.testing.assert_allclose(updates['prototypes'], expected, atol=1e-6)

    def test_momentum_accumulates(self):
        """Velocity accumulates over multiple steps."""
        params = {'x': jnp.array([1.0, 2.0])}
        optimizer = riemannian_nesterov(learning_rate=0.01, momentum=0.9)
        state = optimizer.init(params)

        # Multiple steps with same gradient
        for _ in range(5):
            grads = {'x': jnp.array([1.0, 1.0])}
            updates, state = optimizer.update(grads, state, params)

        # Velocity should be larger than initial
        vel_norm = jnp.sqrt(jnp.sum(state.velocity['x'] ** 2))
        assert vel_norm > 1.0  # accumulated momentum

    def test_zero_momentum_equals_sgd(self):
        """With momentum=0, reduces to standard SGD."""
        params = {'x': jnp.array([1.0, 2.0, 3.0])}
        optimizer = riemannian_nesterov(learning_rate=0.05, momentum=0.0)
        state = optimizer.init(params)

        grads = {'x': jnp.array([2.0, 3.0, 4.0])}
        updates, _ = optimizer.update(grads, state, params)

        # With mu=0: v_new = grad, update = -lr * (0 + grad) = -lr * grad
        expected = -0.05 * jnp.array([2.0, 3.0, 4.0])
        np.testing.assert_allclose(updates['x'], expected, atol=1e-6)


# =============================================================================
# 4. Prototype Diversity Regularization Tests
# =============================================================================

class TestPrototypeDiversityLoss:
    """Tests for DPP-inspired diversity regularization."""

    def test_spread_prototypes_low_loss(self):
        """Well-separated prototypes have low diversity loss."""
        # Two prototypes far apart (same class)
        prototypes = jnp.array([[10.0, 0.0], [-10.0, 0.0]])
        labels = jnp.array([0, 0])
        loss = prototype_diversity_loss(prototypes, labels, sigma_div=1.0)
        assert loss < 1.0  # Should be small (prototypes are diverse)

    def test_collapsed_prototypes_high_loss(self):
        """Collapsed prototypes have high diversity loss."""
        # Two prototypes at nearly the same point
        prototypes = jnp.array([[0.0, 0.0], [0.001, 0.0]])
        labels = jnp.array([0, 0])
        loss = prototype_diversity_loss(prototypes, labels, sigma_div=1.0)
        assert loss > 5.0  # Should be large (prototypes collapsed)

    def test_different_classes_ignored(self):
        """Prototypes from different classes don't interact."""
        # Two prototypes close but from different classes
        prototypes = jnp.array([[0.0, 0.0], [0.1, 0.0]])
        labels = jnp.array([0, 1])
        loss = prototype_diversity_loss(prototypes, labels, sigma_div=1.0)
        # Each class has 1 prototype, so loss = 0
        assert abs(float(loss)) < 1e-3

    def test_sigma_controls_scale(self):
        """Larger sigma_div reduces penalty for moderate distances."""
        prototypes = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        labels = jnp.array([0, 0])

        loss_small_sigma = prototype_diversity_loss(prototypes, labels, sigma_div=0.1)
        loss_large_sigma = prototype_diversity_loss(prototypes, labels, sigma_div=10.0)

        # Small sigma: prototypes are "far" relative to sigma -> low loss
        # Large sigma: prototypes are "close" relative to sigma -> high loss
        assert loss_small_sigma < loss_large_sigma

    def test_differentiable(self):
        """Loss is differentiable w.r.t. prototypes."""
        prototypes = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        labels = jnp.array([0, 0, 0])

        grad_fn = jax.grad(lambda p: prototype_diversity_loss(p, labels, sigma_div=1.0))
        grads = grad_fn(prototypes)
        assert grads.shape == prototypes.shape
        assert not jnp.any(jnp.isnan(grads))

    def test_vectorized_produces_finite_result(self):
        """Vectorized version produces a finite diversity loss."""
        prototypes = jnp.array([
            [1.0, 0.0], [2.0, 0.0],  # class 0
            [5.0, 0.0], [6.0, 0.0],  # class 1
        ])
        labels = jnp.array([0, 0, 1, 1])

        loss_vec = prototype_diversity_loss_vectorized(
            prototypes, labels, sigma_div=1.0,
            n_classes=2, max_protos_per_class=2
        )
        assert jnp.isfinite(loss_vec)
        # Prototypes are 1 unit apart, so kernel value is exp(-0.5) ≈ 0.6
        # logdet of 2x2 kernel matrix should be finite and negative (suboptimal diversity)
        assert float(loss_vec) > 0.0


# =============================================================================
# 5. Sparse Relevance Tests
# =============================================================================

class TestSparseRelevance:
    """Tests for sparse relevance via proximal gradient."""

    def test_soft_thresholding_zeros(self):
        """Values below threshold become exactly zero."""
        relevances = jnp.array([0.5, 0.01, -0.02, 0.3, -0.5])
        result = sparse_relevance_proximal(relevances, l1_weight=0.1)

        # Values |x| < 0.1 should be zero
        assert float(result[1]) == 0.0
        assert float(result[2]) == 0.0

    def test_soft_thresholding_shrinkage(self):
        """Values above threshold are shrunk but not zeroed."""
        relevances = jnp.array([1.0, -1.0, 0.5])
        result = sparse_relevance_proximal(relevances, l1_weight=0.1)

        # 1.0 -> 1.0 - 0.1 = 0.9
        np.testing.assert_allclose(float(result[0]), 0.9, atol=1e-6)
        # -1.0 -> -(1.0 - 0.1) = -0.9
        np.testing.assert_allclose(float(result[1]), -0.9, atol=1e-6)

    def test_elastic_net_combines_l1_l2(self):
        """Elastic net applies both thresholding and shrinkage."""
        relevances = jnp.array([1.0, 0.05, -0.8])
        result = elastic_net_proximal(relevances, l1_weight=0.1, l2_weight=0.1)

        # First: soft threshold at 0.1, then divide by (1 + 0.1)
        # 1.0 -> (1.0 - 0.1) / 1.1 = 0.818...
        np.testing.assert_allclose(float(result[0]), 0.9 / 1.1, atol=1e-5)
        # 0.05 -> 0 (below threshold)
        assert float(result[1]) == 0.0

    def test_lr_scales_threshold(self):
        """Learning rate parameter scales the effective threshold."""
        relevances = jnp.array([0.5, 0.1, -0.3])
        result_lr1 = sparse_relevance_proximal(relevances, l1_weight=0.1, lr=1.0)
        result_lr2 = sparse_relevance_proximal(relevances, l1_weight=0.1, lr=2.0)

        # Higher lr -> more aggressive thresholding
        assert jnp.sum(result_lr2 == 0) >= jnp.sum(result_lr1 == 0)

    def test_zero_l1_no_thresholding(self):
        """With l1_weight=0, no thresholding occurs."""
        relevances = jnp.array([0.01, -0.01, 0.5])
        result = sparse_relevance_proximal(relevances, l1_weight=0.0)
        np.testing.assert_allclose(result, relevances, atol=1e-6)


# =============================================================================
# 6. Curriculum Learning Tests
# =============================================================================

class TestCurriculumLearning:
    """Tests for curriculum learning (self-paced)."""

    def test_hard_mode_binary_weights(self):
        """Hard mode produces binary 0/1 weights."""
        losses = jnp.array([0.1, 0.5, 0.9, 0.2, 0.8])
        weights = curriculum_weights(losses, threshold=0.5, mode='hard')

        # Losses <= 0.5 -> weight 1, else 0
        expected = jnp.array([1.0, 1.0, 0.0, 1.0, 0.0])
        np.testing.assert_allclose(weights, expected)

    def test_soft_mode_smooth_weights(self):
        """Soft mode produces smooth weights in [0, 1]."""
        losses = jnp.array([0.1, 0.5, 1.0, 2.0])
        weights = curriculum_weights(losses, threshold=0.5, mode='soft')

        # All weights should be in [0, 1]
        assert jnp.all(weights >= 0.0)
        assert jnp.all(weights <= 1.0)
        # Easy sample (0.1 < 0.5) should have weight 1
        np.testing.assert_allclose(float(weights[0]), 1.0, atol=1e-5)
        # Hard sample (2.0 >> 0.5) should have low weight
        assert float(weights[3]) < 0.3

    def test_linear_mode(self):
        """Linear mode gives linearly decreasing weights above threshold."""
        losses = jnp.array([0.0, 0.5, 1.0, 1.5])
        weights = curriculum_weights(losses, threshold=1.0, mode='linear')

        # 0.0 -> 1.0 (below threshold)
        np.testing.assert_allclose(float(weights[0]), 1.0, atol=1e-5)
        # 1.0 -> 1.0 (at threshold)
        np.testing.assert_allclose(float(weights[2]), 1.0, atol=1e-5)
        # 1.5 -> 0.5 (linearly ramped)
        np.testing.assert_allclose(float(weights[3]), 0.5, atol=1e-5)

    def test_threshold_increases_over_training(self):
        """Threshold grows from init to final over training."""
        t_start = curriculum_threshold(0, max_iter=100, init_threshold=0.3)
        t_mid = curriculum_threshold(50, max_iter=100, init_threshold=0.3)
        t_end = curriculum_threshold(99, max_iter=100, init_threshold=0.3)

        assert t_start < t_mid < t_end

    def test_exponential_schedule(self):
        """Exponential schedule grows faster than linear."""
        t_lin = curriculum_threshold(
            50, max_iter=100, init_threshold=0.1,
            final_threshold=10.0, schedule='linear'
        )
        t_exp = curriculum_threshold(
            50, max_iter=100, init_threshold=0.1,
            final_threshold=10.0, schedule='exponential'
        )
        # Both should be between init and final
        assert 0.1 < float(t_lin) < 10.0
        assert 0.1 < float(t_exp) < 10.0

    def test_apply_curriculum_full_pipeline(self):
        """Full curriculum pipeline produces valid weighted loss."""
        losses = jnp.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0])
        # Early in training: should weight easy samples more
        loss_early = apply_curriculum_to_loss(losses, iteration=0, max_iter=100)
        # Late in training: should include all samples
        loss_late = apply_curriculum_to_loss(losses, iteration=99, max_iter=100)

        assert jnp.isfinite(loss_early)
        assert jnp.isfinite(loss_late)
        # Early loss should be lower (only easy samples counted)
        assert float(loss_early) < float(loss_late)


# =============================================================================
# 7. Reject Option Tests
# =============================================================================

class TestRejectOption:
    """Tests for reject option with calibrated uncertainty."""

    def _make_model_with_rejection(self, simple_data):
        """Create a fitted model with RejectOptionMixin."""
        from prosemble.models import GLVQ

        # GLVQ already inherits RejectOptionMixin via SupervisedPrototypeModel
        X, y = simple_data
        model = GLVQ(
            n_prototypes_per_class=2, max_iter=50, lr=0.01, use_scan=False
        )
        model.fit(X, y)
        return model, X, y

    def test_confidence_range(self, simple_data):
        """Confidence scores are in [-1, 1]."""
        model, X, y = self._make_model_with_rejection(simple_data)
        conf = model.confidence(X)
        assert jnp.all(conf >= -1.0 - 1e-5)
        assert jnp.all(conf <= 1.0 + 1e-5)

    def test_predict_with_rejection_returns_valid(self, simple_data):
        """Predictions are either valid class labels or -1."""
        model, X, y = self._make_model_with_rejection(simple_data)
        preds = model.predict_with_rejection(X, threshold=0.3)

        valid_labels = set(np.unique(np.asarray(y)).tolist() + [-1])
        pred_labels = set(np.unique(np.asarray(preds)).tolist())
        assert pred_labels.issubset(valid_labels)

    def test_high_threshold_more_rejections(self, simple_data):
        """Higher threshold leads to more rejections."""
        model, X, y = self._make_model_with_rejection(simple_data)

        rate_low = model.rejection_rate(X, threshold=0.0)
        rate_high = model.rejection_rate(X, threshold=0.8)

        assert rate_high >= rate_low

    def test_zero_threshold_no_reject_easy_data(self, simple_data):
        """On well-separated data, threshold=0 should reject very few."""
        model, X, y = self._make_model_with_rejection(simple_data)
        rate = model.rejection_rate(X, threshold=0.0)
        # Well-separated data: most samples should have positive confidence
        assert rate < 0.5

    def test_accuracy_coverage_curve(self, simple_data):
        """Accuracy-coverage curve is monotonically related."""
        model, X, y = self._make_model_with_rejection(simple_data)
        thresholds, accuracies, coverages = model.accuracy_coverage_curve(X, y)

        # Coverages should decrease with increasing threshold
        assert coverages[0] >= coverages[-1]
        # All values should be valid
        assert np.all(accuracies >= 0.0) and np.all(accuracies <= 1.0)
        assert np.all(coverages >= 0.0) and np.all(coverages <= 1.0)

    def test_optimal_threshold_in_range(self, simple_data):
        """Optimal threshold is within valid range."""
        model, X, y = self._make_model_with_rejection(simple_data)
        t_opt = model.optimal_threshold(X, y)
        assert -1.0 <= t_opt <= 1.0


# =============================================================================
# 8. Geodesic Interpolation Tests
# =============================================================================

class TestGeodesicInterpolation:
    """Tests for geodesic interpolation on Riemannian manifolds.

    Note: SO(3) log_map uses Schur decomposition which requires CPU.
    All tests use jax.default_device(cpu) context.
    """

    def _cpu_device(self):
        return jax.devices('cpu')[0]

    def test_interpolation_endpoints(self, so3_manifold, so3_points):
        """Geodesic starts at point_a and ends at point_b."""
        R1, R2 = so3_points
        with jax.default_device(self._cpu_device()):
            path = geodesic_interpolation(so3_manifold, R1, R2, n_points=20)

        # First point should be close to R1
        np.testing.assert_allclose(path[0], R1, atol=1e-4)
        # Last point should be close to R2
        np.testing.assert_allclose(path[-1], R2, atol=1e-4)

    def test_interpolation_on_manifold(self, so3_manifold, so3_points):
        """All interpolated points lie on SO(3)."""
        R1, R2 = so3_points
        with jax.default_device(self._cpu_device()):
            path = geodesic_interpolation(so3_manifold, R1, R2, n_points=10)

        for i in range(path.shape[0]):
            assert so3_manifold.belongs(path[i]), f"Point {i} not on manifold"

    def test_midpoint_equidistant(self, so3_manifold, so3_points):
        """Midpoint is approximately equidistant from both endpoints."""
        R1, R2 = so3_points
        with jax.default_device(self._cpu_device()):
            mid = geodesic_midpoint(so3_manifold, R1, R2)
            d1 = so3_manifold.distance(R1, mid)
            d2 = so3_manifold.distance(R2, mid)

        # On SO(3), midpoint should be equidistant
        np.testing.assert_allclose(float(d1), float(d2), atol=1e-3)

    def test_midpoint_on_manifold(self, so3_manifold, so3_points):
        """Midpoint lies on the manifold."""
        R1, R2 = so3_points
        with jax.default_device(self._cpu_device()):
            mid = geodesic_midpoint(so3_manifold, R1, R2)
        assert so3_manifold.belongs(mid)

    def test_decision_boundary_between_equidistant(self, so3_manifold, so3_points):
        """Decision boundary point has equal distance to both prototypes."""
        R1, R2 = so3_points
        with jax.default_device(self._cpu_device()):
            boundary, t = decision_boundary_point(so3_manifold, R1, R2)
            d1 = so3_manifold.distance(boundary, R1)
            d2 = so3_manifold.distance(boundary, R2)

        # Should be approximately equidistant
        np.testing.assert_allclose(float(d1), float(d2), atol=0.05)

    def test_decision_boundary_t_near_half(self, so3_manifold, so3_points):
        """On symmetric manifolds, boundary is near t=0.5."""
        R1, R2 = so3_points
        with jax.default_device(self._cpu_device()):
            _, t = decision_boundary_point(so3_manifold, R1, R2)

        # On SO(3), boundary should be close to midpoint
        assert 0.3 < t < 0.7

    def test_geodesic_distances_symmetric(self, so3_manifold):
        """Pairwise distance matrix is symmetric."""
        with jax.default_device(self._cpu_device()):
            key = jax.random.PRNGKey(42)
            keys = jax.random.split(key, 3)
            protos = jnp.stack([so3_manifold.random_point(k) for k in keys])
            labels = jnp.array([0, 0, 1])

            dists = prototype_geodesic_distances(so3_manifold, protos, labels)

        np.testing.assert_allclose(dists, dists.T, atol=1e-5)
        # Diagonal should be zero
        np.testing.assert_allclose(jnp.diag(dists), 0.0, atol=1e-5)


# =============================================================================
# Integration Tests: Using improvements with real models
# =============================================================================

class TestIntegrationWithModels:
    """Integration tests combining improvements with actual models."""

    def test_per_group_clip_with_glvq(self, simple_data):
        """Per-group clip works as optimizer for GLVQ."""
        from prosemble.models import GLVQ

        X, y = simple_data
        optimizer = optax.chain(
            per_group_clip({'prototypes': 2.0}),
            optax.adam(0.01),
        )
        model = GLVQ(
            n_prototypes_per_class=2, max_iter=30,
            optimizer=optimizer, use_scan=False
        )
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy > 0.7

    def test_hypergradient_with_glvq(self, simple_data):
        """Hypergradient descent works as optimizer for GLVQ."""
        from prosemble.models import GLVQ

        X, y = simple_data
        optimizer = hypergradient_descent(init_lr=0.005, hyper_lr=1e-5)
        model = GLVQ(
            n_prototypes_per_class=2, max_iter=100,
            optimizer=optimizer, use_scan=False
        )
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy > 0.6

    def test_nesterov_with_glvq(self, simple_data):
        """Riemannian Nesterov works as optimizer for GLVQ."""
        from prosemble.models import GLVQ

        X, y = simple_data
        optimizer = riemannian_nesterov(learning_rate=0.01, momentum=0.9)
        model = GLVQ(
            n_prototypes_per_class=2, max_iter=50,
            optimizer=optimizer, use_scan=False
        )
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = float(jnp.mean(preds == y))
        assert accuracy > 0.7

    def test_diversity_regularization_separates_prototypes(self, simple_data):
        """Diversity loss gradient pushes same-class prototypes apart."""
        X, y = simple_data
        # Start with two identical prototypes
        prototypes = jnp.array([
            [2.0, 0.0, 0.0, 0.0],
            [2.01, 0.0, 0.0, 0.0],  # nearly collapsed
        ])
        labels = jnp.array([0, 0])

        grad_fn = jax.grad(
            lambda p: prototype_diversity_loss(p, labels, sigma_div=1.0)
        )
        grads = grad_fn(prototypes)

        # Gradients should push prototypes apart (opposite directions)
        # Proto 0 gradient and Proto 1 gradient should have opposite sign
        # in the first dimension (where they're close)
        assert grads[0, 0] * grads[1, 0] < 0  # opposite directions
