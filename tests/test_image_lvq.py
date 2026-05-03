"""Tests for Image LVQ models (CNN backbone)."""

import jax
import jax.numpy as jnp
import pytest

from prosemble.models import ImageGLVQ, ImageGMLVQ, ImageGTLVQ, ImageCBC


@pytest.fixture(scope="module")
def small_images():
    """Synthetic 8x8x1 grayscale images, 3 classes."""
    key = jax.random.PRNGKey(42)
    n_per_class = 20
    images = []
    labels = []
    for c in range(3):
        key, subkey = jax.random.split(key)
        # Each class has different mean intensity
        imgs = jax.random.normal(subkey, (n_per_class, 8, 8, 1)) + c * 2.0
        images.append(imgs)
        labels.append(jnp.full(n_per_class, c, dtype=jnp.int32))
    X = jnp.concatenate(images, axis=0)
    y = jnp.concatenate(labels, axis=0)
    # Flatten for model input (models reshape internally)
    return X.reshape(X.shape[0], -1), y


# ---------------------------------------------------------------------------
# ImageGLVQ
# ---------------------------------------------------------------------------

class TestImageGLVQ:
    def test_fit(self, small_images):
        X, y = small_images
        model = ImageGLVQ(
            input_shape=(8, 8, 1), channels=[4], kernel_sizes=[3],
            latent_dim=8,
            n_prototypes_per_class=1, max_iter=20,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        assert model.prototypes_ is not None
        assert model.backbone_params_ is not None

    def test_predict(self, small_images):
        X, y = small_images
        model = ImageGLVQ(
            input_shape=(8, 8, 1), channels=[4], kernel_sizes=[3],
            latent_dim=8,
            n_prototypes_per_class=1, max_iter=30,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (60,)
        assert jnp.all(preds >= 0) and jnp.all(preds < 3)

    def test_predict_proba(self, small_images):
        X, y = small_images
        model = ImageGLVQ(
            input_shape=(8, 8, 1), channels=[4], kernel_sizes=[3],
            latent_dim=8,
            n_prototypes_per_class=1, max_iter=30,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (60, 3)
        assert jnp.allclose(jnp.sum(proba, axis=1), 1.0, atol=1e-5)

    def test_transform(self, small_images):
        X, y = small_images
        model = ImageGLVQ(
            input_shape=(8, 8, 1), channels=[4], kernel_sizes=[3],
            latent_dim=8,
            n_prototypes_per_class=1, max_iter=20,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        latent = model.transform(X)
        assert latent.shape == (60, 8)

    def test_loss_decreases(self, small_images):
        X, y = small_images
        model = ImageGLVQ(
            input_shape=(8, 8, 1), channels=[4], kernel_sizes=[3],
            latent_dim=8,
            n_prototypes_per_class=1, max_iter=50,
            lr=0.01, random_seed=42, use_scan=False,
        )
        model.fit(X, y)
        assert float(model.loss_history_[-1]) < float(model.loss_history_[0])

    def test_multi_layer_cnn(self, small_images):
        X, y = small_images
        model = ImageGLVQ(
            input_shape=(8, 8, 1), channels=[4, 8], kernel_sizes=[3, 3],
            latent_dim=8,
            n_prototypes_per_class=1, max_iter=20,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        assert model.prototypes_ is not None

    def test_reproducibility(self, small_images):
        X, y = small_images
        m1 = ImageGLVQ(
            input_shape=(8, 8, 1), channels=[4], kernel_sizes=[3],
            latent_dim=8, n_prototypes_per_class=1, max_iter=15,
            lr=0.01, random_seed=42,
        )
        m1.fit(X, y)
        m2 = ImageGLVQ(
            input_shape=(8, 8, 1), channels=[4], kernel_sizes=[3],
            latent_dim=8, n_prototypes_per_class=1, max_iter=15,
            lr=0.01, random_seed=42,
        )
        m2.fit(X, y)
        assert jnp.allclose(
            m1.predict(X), m2.predict(X)
        )


# ---------------------------------------------------------------------------
# ImageGMLVQ
# ---------------------------------------------------------------------------

class TestImageGMLVQ:
    def test_fit(self, small_images):
        X, y = small_images
        model = ImageGMLVQ(
            input_shape=(8, 8, 1), channels=[4], kernel_sizes=[3],
            latent_dim=8,
            n_prototypes_per_class=1, max_iter=20,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        assert model.prototypes_ is not None
        assert model.omega_ is not None

    def test_predict(self, small_images):
        X, y = small_images
        model = ImageGMLVQ(
            input_shape=(8, 8, 1), channels=[4], kernel_sizes=[3],
            latent_dim=8,
            n_prototypes_per_class=1, max_iter=30,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (60,)

    def test_lambda_matrix(self, small_images):
        X, y = small_images
        model = ImageGMLVQ(
            input_shape=(8, 8, 1), channels=[4], kernel_sizes=[3],
            latent_dim=8,
            n_prototypes_per_class=1, max_iter=20,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        lam = model.lambda_matrix
        assert lam.shape == (8, 8)

    def test_loss_decreases(self, small_images):
        X, y = small_images
        model = ImageGMLVQ(
            input_shape=(8, 8, 1), channels=[4], kernel_sizes=[3],
            latent_dim=8,
            n_prototypes_per_class=1, max_iter=50,
            lr=0.01, random_seed=42, use_scan=False,
        )
        model.fit(X, y)
        assert float(model.loss_history_[-1]) < float(model.loss_history_[0])


# ---------------------------------------------------------------------------
# ImageGTLVQ
# ---------------------------------------------------------------------------

class TestImageGTLVQ:
    def test_fit(self, small_images):
        X, y = small_images
        model = ImageGTLVQ(
            input_shape=(8, 8, 1), channels=[4], kernel_sizes=[3],
            latent_dim=8, subspace_dim=2,
            n_prototypes_per_class=1, max_iter=20,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        assert model.prototypes_ is not None
        assert model.omegas_ is not None
        assert model.omegas_.shape == (3, 8, 2)

    def test_predict(self, small_images):
        X, y = small_images
        model = ImageGTLVQ(
            input_shape=(8, 8, 1), channels=[4], kernel_sizes=[3],
            latent_dim=8, subspace_dim=2,
            n_prototypes_per_class=1, max_iter=30,
            lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (60,)

    def test_loss_decreases(self, small_images):
        X, y = small_images
        model = ImageGTLVQ(
            input_shape=(8, 8, 1), channels=[4], kernel_sizes=[3],
            latent_dim=8, subspace_dim=2,
            n_prototypes_per_class=1, max_iter=50,
            lr=0.01, random_seed=42, use_scan=False,
        )
        model.fit(X, y)
        assert float(model.loss_history_[-1]) < float(model.loss_history_[0])


# ---------------------------------------------------------------------------
# ImageCBC
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_images_01():
    """Synthetic 8x8x1 images in [0, 1] range for CBC."""
    key = jax.random.PRNGKey(42)
    n_per_class = 20
    images = []
    labels = []
    for c in range(3):
        key, subkey = jax.random.split(key)
        imgs = jax.random.uniform(subkey, (n_per_class, 8, 8, 1)) * 0.3 + c * 0.3
        images.append(imgs)
        labels.append(jnp.full(n_per_class, c, dtype=jnp.int32))
    X = jnp.concatenate(images, axis=0)
    y = jnp.concatenate(labels, axis=0)
    return X.reshape(X.shape[0], -1), y


class TestImageCBC:
    def test_fit(self, small_images_01):
        X, y = small_images_01
        model = ImageCBC(
            input_shape=(8, 8, 1), channels=[4], kernel_sizes=[3],
            latent_dim=8, n_components=5, n_classes=3,
            sigma=1.0, max_iter=20, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        assert model.components_ is not None
        assert model.reasonings_ is not None
        assert model.backbone_params_ is not None

    def test_predict(self, small_images_01):
        X, y = small_images_01
        model = ImageCBC(
            input_shape=(8, 8, 1), channels=[4], kernel_sizes=[3],
            latent_dim=8, n_components=5, n_classes=3,
            sigma=1.0, max_iter=30, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (60,)
        assert jnp.all(preds >= 0) and jnp.all(preds < 3)

    def test_predict_proba(self, small_images_01):
        X, y = small_images_01
        model = ImageCBC(
            input_shape=(8, 8, 1), channels=[4], kernel_sizes=[3],
            latent_dim=8, n_components=5, n_classes=3,
            sigma=1.0, max_iter=30, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (60, 3)

    def test_components_clamped(self, small_images_01):
        """Components should be clamped to [0, 1] after training."""
        X, y = small_images_01
        model = ImageCBC(
            input_shape=(8, 8, 1), channels=[4], kernel_sizes=[3],
            latent_dim=8, n_components=5, n_classes=3,
            sigma=1.0, max_iter=30, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        comps = model.components_
        assert jnp.all(comps >= 0.0)
        assert jnp.all(comps <= 1.0)

    def test_loss_decreases(self, small_images_01):
        X, y = small_images_01
        model = ImageCBC(
            input_shape=(8, 8, 1), channels=[4], kernel_sizes=[3],
            latent_dim=8, n_components=5, n_classes=3,
            sigma=1.0, max_iter=50, lr=0.01, random_seed=42,
            use_scan=False,
        )
        model.fit(X, y)
        assert float(model.loss_history_[-1]) < float(model.loss_history_[0])

    def test_transform(self, small_images_01):
        X, y = small_images_01
        model = ImageCBC(
            input_shape=(8, 8, 1), channels=[4], kernel_sizes=[3],
            latent_dim=8, n_components=5, n_classes=3,
            sigma=1.0, max_iter=20, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        latent = model.transform(X)
        assert latent.shape == (60, 8)
