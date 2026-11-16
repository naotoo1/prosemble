"""Temperature calibration of the soft models."""

import os
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from prosemble.core.manifolds import SPD
from prosemble.models import (
    DKLMRSLVQ, DKRSLVQ, OCWRSLVQ, RiemannianRSLVQ, WMRSLVQ, WRSLVQ,
)
from prosemble.models.soft_mixin import _fit_temperature


@pytest.fixture(scope='module')
def wine():
    X, y = load_wine(return_X_y=True)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    # Labels that are not 0..n-1, to check that they are encoded.
    return train_test_split(X, y + 4, test_size=0.4, stratify=y, random_state=0)


def _nll(model, X, y):
    proba = np.asarray(model.predict_proba(X), dtype=np.float64)
    return float(-np.mean(np.log(proba[np.arange(len(y)), np.searchsorted(
        np.asarray(model.classes_), y)])))


class TestFitTemperature:

    def _scores(self, temperature, n=4000, seed=0):
        """Scores of a model whose posteriors are calibrated at ``temperature``."""
        rng = np.random.default_rng(seed)
        logits = rng.normal(size=(n, 3)) * 3.0
        z = logits / temperature
        p = np.exp(z - z.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        labels = np.array([rng.choice(3, p=row) for row in p])
        return logits, labels

    @pytest.mark.parametrize('temperature', [0.4, 1.0, 2.5])
    def test_recovers_the_temperature_that_generated_the_labels(self, temperature):
        logits, labels = self._scores(temperature)
        fitted = _fit_temperature(logits, labels, np.arange(3), 3)
        assert fitted == pytest.approx(temperature, rel=0.1)

    def test_minimises_the_loss_of_the_library_class_posterior(self):
        """With several prototypes per class, against a brute-force search."""
        from prosemble.core.losses import class_log_posterior
        rng = np.random.default_rng(1)
        logits = rng.normal(size=(500, 5)) * 2.0
        prototype_labels = np.array([0, 0, 1, 2, 2])
        labels = rng.integers(0, 3, size=500)
        labels[:300] = prototype_labels[np.argmax(logits[:300], axis=1)]

        def nll(t):
            lp = np.asarray(class_log_posterior(jnp.asarray(logits / t), jnp.asarray(prototype_labels), 3))
            return -np.mean(lp[np.arange(500), labels])

        grid = np.exp(np.linspace(np.log(0.05), np.log(20), 400))
        brute = grid[int(np.argmin([nll(t) for t in grid]))]
        fitted = _fit_temperature(logits, labels, prototype_labels, 3)
        assert fitted == pytest.approx(brute, rel=0.03)
        assert nll(fitted) <= nll(brute) + 1e-6


class TestCalibrate:

    @pytest.mark.parametrize('cls, k', [(DKLMRSLVQ, 1), (WRSLVQ, 3), (WMRSLVQ, 2)])
    def test_cross_fitted_temperature_does_not_raise_the_held_out_loss(self, wine, cls, k):
        Xa, Xb, ya, yb = wine
        model = cls(n_prototypes_per_class=k, max_iter=80, random_seed=0)
        model.fit(Xa, ya)
        before = _nll(model, Xb, yb)
        labels = np.asarray(model.predict(Xb))
        model.calibrate(Xa, ya, cv=3)
        assert float(model.temperature_) > 0
        assert _nll(model, Xb, yb) <= before + 0.02
        np.testing.assert_array_equal(np.asarray(model.predict(Xb)), labels)

    def test_held_out_temperature_minimises_the_loss_on_that_data(self, wine):
        Xa, Xb, ya, yb = wine
        model = WRSLVQ(n_prototypes_per_class=2, max_iter=60, random_seed=0)
        model.fit(Xa, ya)
        model.calibrate(Xb, yb)
        best = _nll(model, Xb, yb)
        fitted = float(model.temperature_)
        for factor in (0.7, 1.4):
            model.temperature_ = jnp.asarray(fitted * factor)
            assert _nll(model, Xb, yb) >= best - 1e-6

    def test_temperature_is_a_rescaling_of_sigma(self, wine):
        Xa, Xb, ya, _ = wine
        model = DKRSLVQ(max_iter=40, random_seed=0)
        model.fit(Xa, ya)
        model.temperature_ = jnp.asarray(2.0)
        calibrated = np.asarray(model.predict_proba(Xb))
        model.temperature_ = None
        model.sigma_ = model.sigma_ * np.sqrt(2.0)
        np.testing.assert_allclose(np.asarray(model.predict_proba(Xb)), calibrated, atol=1e-5)

    def test_rejection_uses_the_calibrated_confidence_and_the_trained_labels(self, wine):
        Xa, Xb, ya, _ = wine
        model = WRSLVQ(n_prototypes_per_class=3, max_iter=60, random_seed=0)
        model.fit(Xa, ya)
        model.temperature_ = jnp.asarray(20.0)          # nearly uniform posteriors
        rejected = np.asarray(model.predict_with_rejection(Xb, confidence=0.5))
        assert np.all(rejected == -1)
        accepted = np.asarray(model.predict_with_rejection(Xb, confidence=0.0))
        np.testing.assert_array_equal(accepted, np.asarray(model.predict(Xb)))

    def test_refit_drops_the_temperature(self, wine):
        Xa, _, ya, _ = wine
        model = WRSLVQ(max_iter=20, random_seed=0)
        model.fit(Xa, ya).calibrate(Xa, ya, cv=2)
        assert model.temperature_ is not None
        model.fit(Xa, ya)
        assert model.temperature_ is None

    def test_save_load_keeps_the_temperature(self, wine):
        Xa, Xb, ya, _ = wine
        model = DKRSLVQ(max_iter=20, random_seed=0)
        model.fit(Xa, ya).calibrate(Xa, ya, cv=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model.npz')
            model.save(path)
            loaded = DKRSLVQ.load(path)
        assert float(loaded.temperature_) == pytest.approx(float(model.temperature_))
        np.testing.assert_allclose(np.asarray(loaded.predict_proba(Xb)),
                                   np.asarray(model.predict_proba(Xb)), rtol=1e-5)

    def test_riemannian_model(self):
        import jax
        spd = SPD(3)
        keys = jax.random.split(jax.random.PRNGKey(0), 40)
        X = jnp.stack([spd.random_point(k) for k in keys]).reshape(40, -1)
        y = jnp.array([0] * 20 + [1] * 20)
        model = RiemannianRSLVQ(manifold=spd, max_iter=10)
        model.fit(X, y).calibrate(X, y, cv=2)
        assert float(model.temperature_) > 0

    @pytest.mark.parametrize('cv', [1, 0, 2.5, 'all'])
    def test_invalid_cv(self, wine, cv):
        Xa, _, ya, _ = wine
        model = WRSLVQ(max_iter=5).fit(Xa, ya)
        with pytest.raises(ValueError, match='cv'):
            model.calibrate(Xa, ya, cv=cv)

    def test_not_fitted(self, wine):
        Xa, _, ya, _ = wine
        with pytest.raises(Exception, match='fit'):
            WRSLVQ().calibrate(Xa, ya)

    def test_one_class_models_refuse(self, wine):
        Xa, _, ya, _ = wine
        model = OCWRSLVQ(max_iter=5).fit(Xa)
        with pytest.raises(NotImplementedError, match='nu'):
            model.calibrate(Xa, ya)
