"""Tests for prosemble.core.onnx_export — ONNX model export."""

import pytest
import numpy as np
import numpy.testing as npt
import jax.numpy as jnp

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from prosemble.models import GLVQ, GMLVQ, NeuralGas, FCM, KFCM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_supervised_data(n=60, d=4, n_classes=3, seed=42):
    np.random.seed(seed)
    X = jnp.array(np.random.randn(n, d).astype(np.float32))
    y = jnp.array(np.tile(np.arange(n_classes), n // n_classes))
    return X, y


def _make_oc_data(n=60, d=4, seed=42):
    """One-class data: label 1 = target (normal), label 0 = outlier."""
    np.random.seed(seed)
    # 40 normal samples (class 1), 20 outliers (class 0)
    X_normal = np.random.randn(40, d).astype(np.float32)
    X_outlier = np.random.randn(20, d).astype(np.float32) + 5.0
    X = jnp.array(np.concatenate([X_normal, X_outlier], axis=0))
    y = jnp.array([1]*40 + [0]*20)
    return X, y


def _onnx_predict(model, X, batch_size=None):
    """Run ONNX inference and return predictions."""
    if batch_size is None:
        batch_size = X.shape[0]
    onnx_model = model.export_onnx(batch_size=batch_size)
    sess = ort.InferenceSession(onnx_model.SerializeToString())
    return sess.run(None, {'X': np.asarray(X)})[0]


# ---------------------------------------------------------------------------
# Existing tests: GLVQ, GMLVQ, NeuralGas, FCM
# ---------------------------------------------------------------------------

@pytest.fixture
def fitted_glvq():
    X, y = _make_supervised_data(30, 4)
    model = GLVQ(n_prototypes_per_class=1, max_iter=5, random_seed=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def fitted_gmlvq():
    X, y = _make_supervised_data(30, 4)
    model = GMLVQ(n_prototypes_per_class=1, max_iter=5, random_seed=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def fitted_ng():
    np.random.seed(42)
    X = jnp.array(np.random.randn(30, 4).astype(np.float32))
    model = NeuralGas(n_prototypes=3, max_iter=5, random_seed=42)
    model.fit(X)
    return model, X


class TestGLVQExport:

    def test_export_returns_model_proto(self, fitted_glvq):
        model, X, y = fitted_glvq
        onnx_model = model.export_onnx()
        assert isinstance(onnx_model, onnx.ModelProto)

    def test_export_saves_to_file(self, fitted_glvq, tmp_path):
        model, X, y = fitted_glvq
        path = str(tmp_path / "glvq.onnx")
        model.export_onnx(path=path)
        assert (tmp_path / "glvq.onnx").exists()

    def test_numerical_match(self, fitted_glvq):
        model, X, y = fitted_glvq
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X, batch_size=30)
        npt.assert_array_equal(jax_preds, ort_preds)

    def test_single_sample(self, fitted_glvq):
        model, X, y = fitted_glvq
        x_single = X[:1]
        jax_pred = np.asarray(model.predict(x_single))
        ort_pred = _onnx_predict(model, x_single, batch_size=1)
        npt.assert_array_equal(jax_pred, ort_pred)


class TestGMLVQExport:

    def test_omega_distance_detected(self, fitted_gmlvq):
        from prosemble.core.onnx_export import _identify_model_type
        model, X, y = fitted_gmlvq
        model_type, dist_type = _identify_model_type(model)
        assert model_type == 'supervised'
        assert dist_type == 'omega'

    def test_export_returns_model_proto(self, fitted_gmlvq):
        model, X, y = fitted_gmlvq
        onnx_model = model.export_onnx()
        assert isinstance(onnx_model, onnx.ModelProto)

    def test_numerical_match(self, fitted_gmlvq):
        model, X, y = fitted_gmlvq
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X, batch_size=30)
        npt.assert_array_equal(jax_preds, ort_preds)


class TestNeuralGasExport:

    def test_unsupervised_export(self, fitted_ng):
        model, X = fitted_ng
        onnx_model = model.export_onnx()
        assert isinstance(onnx_model, onnx.ModelProto)

    def test_numerical_match(self, fitted_ng):
        model, X = fitted_ng
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X, batch_size=30)
        npt.assert_array_equal(jax_preds, ort_preds)


class TestFCMExport:

    @pytest.fixture
    def fitted_fcm(self):
        np.random.seed(42)
        X = jnp.array(np.random.randn(30, 4).astype(np.float32))
        model = FCM(n_clusters=3, max_iter=20, random_seed=42)
        model.fit(X)
        return model, X

    def test_export_returns_model_proto(self, fitted_fcm):
        model, X = fitted_fcm
        onnx_model = model.export_onnx()
        assert isinstance(onnx_model, onnx.ModelProto)

    def test_numerical_match(self, fitted_fcm):
        model, X = fitted_fcm
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X, batch_size=30)
        npt.assert_array_equal(jax_preds, ort_preds)

    def test_export_saves_to_file(self, fitted_fcm, tmp_path):
        model, X = fitted_fcm
        path = str(tmp_path / "fcm.onnx")
        model.export_onnx(path=path)
        assert (tmp_path / "fcm.onnx").exists()


class TestKernelFuzzyRejection:

    def test_kfcm_raises(self):
        from prosemble.core.onnx_export import _check_model_exportable
        model = KFCM(n_clusters=3, max_iter=1)
        with pytest.raises(NotImplementedError, match="kernel fuzzy"):
            _check_model_exportable(model)


class TestUnsupportedModels:

    def test_riemannian_raises(self):
        from prosemble.core.manifolds import SO
        from prosemble.models import RiemannianSRNG
        model = RiemannianSRNG(manifold=SO(3), n_prototypes_per_class=1, max_iter=1)
        from prosemble.core.onnx_export import _check_model_exportable
        with pytest.raises(NotImplementedError, match="Riemannian"):
            _check_model_exportable(model)

    def test_riemannian_ng_raises(self):
        from prosemble.core.manifolds import SO
        from prosemble.models import RiemannianNeuralGas
        model = RiemannianNeuralGas(manifold=SO(3), n_prototypes=2, max_iter=1)
        from prosemble.core.onnx_export import _check_model_exportable
        with pytest.raises(NotImplementedError, match="RiemannianNeuralGas"):
            _check_model_exportable(model)


class TestOnnxImportGuard:

    def test_export_onnx_importable(self):
        from prosemble.core import export_onnx
        assert callable(export_onnx)


# ---------------------------------------------------------------------------
# NEW: Local omega / tangent supervised models
# ---------------------------------------------------------------------------

class TestLocalOmegaExport:
    """LGMLVQ: per-prototype Omega, ||Omega_k (x - w_k)||^2, WTAC."""

    @pytest.fixture
    def fitted_lgmlvq(self):
        from prosemble.models import LGMLVQ
        X, y = _make_supervised_data(60, 4)
        model = LGMLVQ(
            n_prototypes_per_class=1, max_iter=10, lr=0.01,
            latent_dim=3, random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_model_type_detected(self, fitted_lgmlvq):
        from prosemble.core.onnx_export import _identify_model_type
        model, X = fitted_lgmlvq
        model_type, dist_type = _identify_model_type(model)
        assert model_type == 'supervised'
        assert dist_type == 'local_omega'

    def test_export_returns_model_proto(self, fitted_lgmlvq):
        model, X = fitted_lgmlvq
        onnx_model = model.export_onnx()
        assert isinstance(onnx_model, onnx.ModelProto)

    def test_numerical_match(self, fitted_lgmlvq):
        model, X = fitted_lgmlvq
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)


class TestTangentExport:
    """GTLVQ: per-prototype tangent subspace, orthogonal complement distance."""

    @pytest.fixture
    def fitted_gtlvq(self):
        from prosemble.models import GTLVQ
        X, y = _make_supervised_data(60, 4)
        model = GTLVQ(
            n_prototypes_per_class=1, max_iter=10, lr=0.01,
            subspace_dim=2, random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_model_type_detected(self, fitted_gtlvq):
        from prosemble.core.onnx_export import _identify_model_type
        model, X = fitted_gtlvq
        model_type, dist_type = _identify_model_type(model)
        assert model_type == 'supervised'
        assert dist_type == 'tangent'

    def test_export_returns_model_proto(self, fitted_gtlvq):
        model, X = fitted_gtlvq
        onnx_model = model.export_onnx()
        assert isinstance(onnx_model, onnx.ModelProto)

    def test_numerical_match(self, fitted_gtlvq):
        model, X = fitted_gtlvq
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)


class TestSLNGExport:
    """SLNG: same local omega distance as LGMLVQ, NG training only."""

    @pytest.fixture
    def fitted_slng(self):
        from prosemble.models import SLNG
        X, y = _make_supervised_data(60, 4)
        model = SLNG(
            n_prototypes_per_class=1, max_iter=10, lr=0.01,
            latent_dim=3, random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_numerical_match(self, fitted_slng):
        model, X = fitted_slng
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)


class TestSTNGExport:
    """STNG: same tangent distance as GTLVQ, NG training only."""

    @pytest.fixture
    def fitted_stng(self):
        from prosemble.models import STNG
        X, y = _make_supervised_data(60, 4)
        model = STNG(
            n_prototypes_per_class=1, max_iter=10, lr=0.01,
            subspace_dim=2, random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_numerical_match(self, fitted_stng):
        model, X = fitted_stng
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)


class TestLMRSLVQExport:
    """LMRSLVQ: local omega + RSLVQ loss."""

    @pytest.fixture
    def fitted_lmrslvq(self):
        from prosemble.models import LMRSLVQ
        X, y = _make_supervised_data(60, 4)
        model = LMRSLVQ(
            n_prototypes_per_class=1, max_iter=10, lr=0.01,
            latent_dim=3, random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_numerical_match(self, fitted_lmrslvq):
        model, X = fitted_lmrslvq
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)


# ---------------------------------------------------------------------------
# NEW: One-class hard nearest models
# ---------------------------------------------------------------------------

class TestOCGLVQExport:
    """OCGLVQ: squared Euclidean + hard nearest decision."""

    @pytest.fixture
    def fitted_ocglvq(self):
        from prosemble.models import OCGLVQ
        X, y = _make_oc_data()
        model = OCGLVQ(
            n_prototypes=3, max_iter=10, lr=0.01,
            random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_model_type_detected(self, fitted_ocglvq):
        from prosemble.core.onnx_export import _identify_model_type
        model, X = fitted_ocglvq
        model_type, dist_type = _identify_model_type(model)
        assert model_type == 'oc_hard_nearest'
        assert dist_type == 'squared_euclidean'

    def test_export_returns_model_proto(self, fitted_ocglvq):
        model, X = fitted_ocglvq
        onnx_model = model.export_onnx()
        assert isinstance(onnx_model, onnx.ModelProto)

    def test_numerical_match(self, fitted_ocglvq):
        model, X = fitted_ocglvq
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)

    def test_single_sample(self, fitted_ocglvq):
        model, X = fitted_ocglvq
        x_single = X[:1]
        jax_pred = np.asarray(model.predict(x_single))
        ort_pred = _onnx_predict(model, x_single, batch_size=1)
        npt.assert_array_equal(jax_pred, ort_pred)


class TestOCGMLVQExport:
    """OCGMLVQ: global omega + hard nearest decision."""

    @pytest.fixture
    def fitted_ocgmlvq(self):
        from prosemble.models import OCGMLVQ
        X, y = _make_oc_data()
        model = OCGMLVQ(
            n_prototypes=3, max_iter=10, lr=0.01,
            random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_model_type_detected(self, fitted_ocgmlvq):
        from prosemble.core.onnx_export import _identify_model_type
        model, X = fitted_ocgmlvq
        model_type, dist_type = _identify_model_type(model)
        assert model_type == 'oc_hard_nearest'
        assert dist_type == 'omega'

    def test_numerical_match(self, fitted_ocgmlvq):
        model, X = fitted_ocgmlvq
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)


class TestOCGRLVQExport:
    """OCGRLVQ: relevance-weighted + hard nearest decision."""

    @pytest.fixture
    def fitted_ocgrlvq(self):
        from prosemble.models import OCGRLVQ
        X, y = _make_oc_data()
        model = OCGRLVQ(
            n_prototypes=3, max_iter=10, lr=0.01,
            random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_model_type_detected(self, fitted_ocgrlvq):
        from prosemble.core.onnx_export import _identify_model_type
        model, X = fitted_ocgrlvq
        model_type, dist_type = _identify_model_type(model)
        assert model_type == 'oc_hard_nearest'
        assert dist_type == 'relevance'

    def test_numerical_match(self, fitted_ocgrlvq):
        model, X = fitted_ocgrlvq
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)


class TestOCLGMLVQExport:
    """OCLGMLVQ: local omega + hard nearest decision."""

    @pytest.fixture
    def fitted_oclgmlvq(self):
        from prosemble.models import OCLGMLVQ
        X, y = _make_oc_data()
        model = OCLGMLVQ(
            n_prototypes=3, max_iter=10, lr=0.01,
            latent_dim=3, random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_model_type_detected(self, fitted_oclgmlvq):
        from prosemble.core.onnx_export import _identify_model_type
        model, X = fitted_oclgmlvq
        model_type, dist_type = _identify_model_type(model)
        assert model_type == 'oc_hard_nearest'
        assert dist_type == 'local_omega'

    def test_numerical_match(self, fitted_oclgmlvq):
        model, X = fitted_oclgmlvq
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)


class TestOCGTLVQExport:
    """OCGTLVQ: tangent + hard nearest decision."""

    @pytest.fixture
    def fitted_ocgtlvq(self):
        from prosemble.models import OCGTLVQ
        X, y = _make_oc_data()
        model = OCGTLVQ(
            n_prototypes=3, max_iter=10, lr=0.01,
            subspace_dim=2, random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_model_type_detected(self, fitted_ocgtlvq):
        from prosemble.core.onnx_export import _identify_model_type
        model, X = fitted_ocgtlvq
        model_type, dist_type = _identify_model_type(model)
        assert model_type == 'oc_hard_nearest'
        assert dist_type == 'tangent'

    def test_numerical_match(self, fitted_ocgtlvq):
        model, X = fitted_ocgtlvq
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)


# ---------------------------------------------------------------------------
# NEW: One-class Gaussian soft models
# ---------------------------------------------------------------------------

class TestOCRSLVQExport:
    """OCRSLVQ: squared Euclidean + Gaussian soft decision."""

    @pytest.fixture
    def fitted_ocrslvq(self):
        from prosemble.models import OCRSLVQ
        X, y = _make_oc_data()
        model = OCRSLVQ(
            n_prototypes=3, max_iter=10, lr=0.01,
            sigma=1.0, random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_model_type_detected(self, fitted_ocrslvq):
        from prosemble.core.onnx_export import _identify_model_type
        model, X = fitted_ocrslvq
        model_type, dist_type = _identify_model_type(model)
        assert model_type == 'oc_gaussian_soft'
        assert dist_type == 'squared_euclidean'

    def test_numerical_match(self, fitted_ocrslvq):
        model, X = fitted_ocrslvq
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)


class TestOCMRSLVQExport:
    """OCMRSLVQ: global omega + Gaussian soft decision."""

    @pytest.fixture
    def fitted_ocmrslvq(self):
        from prosemble.models import OCMRSLVQ
        X, y = _make_oc_data()
        model = OCMRSLVQ(
            n_prototypes=3, max_iter=10, lr=0.01,
            sigma=1.0, random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_model_type_detected(self, fitted_ocmrslvq):
        from prosemble.core.onnx_export import _identify_model_type
        model, X = fitted_ocmrslvq
        model_type, dist_type = _identify_model_type(model)
        assert model_type == 'oc_gaussian_soft'
        assert dist_type == 'omega'

    def test_numerical_match(self, fitted_ocmrslvq):
        model, X = fitted_ocmrslvq
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)


# ---------------------------------------------------------------------------
# NEW: One-class Gaussian+NG models
# ---------------------------------------------------------------------------

class TestOCRSLVQNGExport:
    """OCRSLVQ_NG: squared Euclidean + Gaussian+NG soft decision."""

    @pytest.fixture
    def fitted_ocrslvq_ng(self):
        from prosemble.models import OCRSLVQ_NG
        X, y = _make_oc_data()
        model = OCRSLVQ_NG(
            n_prototypes=3, max_iter=10, lr=0.01,
            sigma=1.0, random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_model_type_detected(self, fitted_ocrslvq_ng):
        from prosemble.core.onnx_export import _identify_model_type
        model, X = fitted_ocrslvq_ng
        model_type, dist_type = _identify_model_type(model)
        assert model_type == 'oc_gaussian_ng'
        assert dist_type == 'squared_euclidean'

    def test_numerical_match(self, fitted_ocrslvq_ng):
        model, X = fitted_ocrslvq_ng
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)


# ---------------------------------------------------------------------------
# NEW: SVQ-OCC models
# ---------------------------------------------------------------------------

class TestSVQOCCExport:
    """SVQOCC: squared Euclidean + Gaussian response."""

    @pytest.fixture
    def fitted_svqocc(self):
        from prosemble.models import SVQOCC
        X, y = _make_oc_data()
        model = SVQOCC(
            n_prototypes=3, max_iter=10, lr=0.01,
            response_type='gaussian', random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_model_type_detected(self, fitted_svqocc):
        from prosemble.core.onnx_export import _identify_model_type
        model, X = fitted_svqocc
        model_type, dist_type = _identify_model_type(model)
        assert model_type == 'svqocc'
        assert dist_type == 'squared_euclidean'

    def test_export_returns_model_proto(self, fitted_svqocc):
        model, X = fitted_svqocc
        onnx_model = model.export_onnx()
        assert isinstance(onnx_model, onnx.ModelProto)

    def test_numerical_match(self, fitted_svqocc):
        model, X = fitted_svqocc
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)


class TestSVQOCCStudentTExport:
    """SVQOCC with Student-t response."""

    @pytest.fixture
    def fitted_svqocc_t_response(self):
        from prosemble.models import SVQOCC
        X, y = _make_oc_data()
        model = SVQOCC(
            n_prototypes=3, max_iter=10, lr=0.01,
            response_type='student_t', random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_numerical_match(self, fitted_svqocc_t_response):
        model, X = fitted_svqocc_t_response
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)


class TestSVQOCCUniformExport:
    """SVQOCC with uniform response."""

    @pytest.fixture
    def fitted_svqocc_uniform(self):
        from prosemble.models import SVQOCC
        X, y = _make_oc_data()
        model = SVQOCC(
            n_prototypes=3, max_iter=10, lr=0.01,
            response_type='uniform', random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_numerical_match(self, fitted_svqocc_uniform):
        model, X = fitted_svqocc_uniform
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)


class TestSVQOCCOmegaExport:
    """SVQOCC_M: global omega + SVQ-OCC response."""

    @pytest.fixture
    def fitted_svqocc_m(self):
        from prosemble.models import SVQOCC_M
        X, y = _make_oc_data()
        model = SVQOCC_M(
            n_prototypes=3, max_iter=10, lr=0.01,
            response_type='gaussian', random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_model_type_detected(self, fitted_svqocc_m):
        from prosemble.core.onnx_export import _identify_model_type
        model, X = fitted_svqocc_m
        model_type, dist_type = _identify_model_type(model)
        assert model_type == 'svqocc'
        assert dist_type == 'omega'

    def test_numerical_match(self, fitted_svqocc_m):
        model, X = fitted_svqocc_m
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)


class TestSVQOCCTangentExport:
    """SVQOCC_T: tangent + SVQ-OCC response."""

    @pytest.fixture
    def fitted_svqocc_t(self):
        from prosemble.models import SVQOCC_T
        X, y = _make_oc_data()
        model = SVQOCC_T(
            n_prototypes=3, max_iter=10, lr=0.01,
            subspace_dim=2, response_type='gaussian', random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_model_type_detected(self, fitted_svqocc_t):
        from prosemble.core.onnx_export import _identify_model_type
        model, X = fitted_svqocc_t
        model_type, dist_type = _identify_model_type(model)
        assert model_type == 'svqocc'
        assert dist_type == 'tangent'

    def test_numerical_match(self, fitted_svqocc_t):
        model, X = fitted_svqocc_t
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)


# ---------------------------------------------------------------------------
# NEW: Encoder models (MLP backbone)
# ---------------------------------------------------------------------------

def _make_image_data(n=40, input_shape=(8, 8, 1), n_classes=2, seed=42):
    """Generate synthetic flat image data for CNN encoder model tests."""
    np.random.seed(seed)
    d = int(np.prod(input_shape))
    X = np.random.randn(n, d).astype(np.float32)
    y = np.tile(np.arange(n_classes), n // n_classes)[:n]
    return jnp.array(X), jnp.array(y)


class TestSiameseGLVQExport:
    """SiameseGLVQ: MLP encoder + squared Euclidean + WTAC."""

    @pytest.fixture
    def fitted_siamese_glvq(self):
        from prosemble.models import SiameseGLVQ
        X, y = _make_supervised_data(40, 4, n_classes=2)
        model = SiameseGLVQ(
            n_prototypes_per_class=1, max_iter=10, lr=0.01,
            hidden_sizes=[8], latent_dim=3, random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_model_type_detected(self, fitted_siamese_glvq):
        from prosemble.core.onnx_export import _identify_model_type
        model, X = fitted_siamese_glvq
        model_type, dist_type = _identify_model_type(model)
        assert model_type == 'supervised'
        assert dist_type == 'squared_euclidean'

    def test_export_returns_model_proto(self, fitted_siamese_glvq):
        model, X = fitted_siamese_glvq
        onnx_model = model.export_onnx()
        assert isinstance(onnx_model, onnx.ModelProto)

    def test_numerical_match(self, fitted_siamese_glvq):
        model, X = fitted_siamese_glvq
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)

    def test_single_sample(self, fitted_siamese_glvq):
        model, X = fitted_siamese_glvq
        x_single = X[:1]
        jax_pred = np.asarray(model.predict(x_single))
        ort_pred = _onnx_predict(model, x_single, batch_size=1)
        npt.assert_array_equal(jax_pred, ort_pred)


class TestSiameseGMLVQExport:
    """SiameseGMLVQ: MLP encoder + omega distance + WTAC."""

    @pytest.fixture
    def fitted_siamese_gmlvq(self):
        from prosemble.models import SiameseGMLVQ
        X, y = _make_supervised_data(40, 4, n_classes=2)
        model = SiameseGMLVQ(
            n_prototypes_per_class=1, max_iter=10, lr=0.01,
            hidden_sizes=[8], latent_dim=3, random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_model_type_detected(self, fitted_siamese_gmlvq):
        from prosemble.core.onnx_export import _identify_model_type
        model, X = fitted_siamese_gmlvq
        model_type, dist_type = _identify_model_type(model)
        assert model_type == 'supervised'
        assert dist_type == 'omega'

    def test_numerical_match(self, fitted_siamese_gmlvq):
        model, X = fitted_siamese_gmlvq
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)


class TestSiameseGTLVQExport:
    """SiameseGTLVQ: MLP encoder + tangent distance + WTAC."""

    @pytest.fixture
    def fitted_siamese_gtlvq(self):
        from prosemble.models import SiameseGTLVQ
        X, y = _make_supervised_data(40, 4, n_classes=2)
        model = SiameseGTLVQ(
            n_prototypes_per_class=1, max_iter=10, lr=0.01,
            hidden_sizes=[8], latent_dim=3, subspace_dim=2,
            random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_model_type_detected(self, fitted_siamese_gtlvq):
        from prosemble.core.onnx_export import _identify_model_type
        model, X = fitted_siamese_gtlvq
        model_type, dist_type = _identify_model_type(model)
        assert model_type == 'supervised'
        assert dist_type == 'tangent'

    def test_numerical_match(self, fitted_siamese_gtlvq):
        model, X = fitted_siamese_gtlvq
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)


# ---------------------------------------------------------------------------
# NEW: Encoder models (CNN backbone)
# ---------------------------------------------------------------------------

class TestImageGLVQExport:
    """ImageGLVQ: CNN encoder + squared Euclidean + WTAC."""

    @pytest.fixture
    def fitted_image_glvq(self):
        from prosemble.models import ImageGLVQ
        input_shape = (8, 8, 1)
        X, y = _make_image_data(40, input_shape, n_classes=2)
        model = ImageGLVQ(
            n_prototypes_per_class=1, max_iter=10, lr=0.01,
            input_shape=input_shape, channels=[4], kernel_sizes=[3],
            latent_dim=3, random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_model_type_detected(self, fitted_image_glvq):
        from prosemble.core.onnx_export import _identify_model_type
        model, X = fitted_image_glvq
        model_type, dist_type = _identify_model_type(model)
        assert model_type == 'supervised'
        assert dist_type == 'squared_euclidean'

    def test_export_returns_model_proto(self, fitted_image_glvq):
        model, X = fitted_image_glvq
        onnx_model = model.export_onnx()
        assert isinstance(onnx_model, onnx.ModelProto)

    def test_numerical_match(self, fitted_image_glvq):
        model, X = fitted_image_glvq
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)

    def test_single_sample(self, fitted_image_glvq):
        model, X = fitted_image_glvq
        x_single = X[:1]
        jax_pred = np.asarray(model.predict(x_single))
        ort_pred = _onnx_predict(model, x_single, batch_size=1)
        npt.assert_array_equal(jax_pred, ort_pred)


class TestImageGMLVQExport:
    """ImageGMLVQ: CNN encoder + omega distance + WTAC."""

    @pytest.fixture
    def fitted_image_gmlvq(self):
        from prosemble.models import ImageGMLVQ
        input_shape = (8, 8, 1)
        X, y = _make_image_data(40, input_shape, n_classes=2)
        model = ImageGMLVQ(
            n_prototypes_per_class=1, max_iter=10, lr=0.01,
            input_shape=input_shape, channels=[4], kernel_sizes=[3],
            latent_dim=3, random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_model_type_detected(self, fitted_image_gmlvq):
        from prosemble.core.onnx_export import _identify_model_type
        model, X = fitted_image_gmlvq
        model_type, dist_type = _identify_model_type(model)
        assert model_type == 'supervised'
        assert dist_type == 'omega'

    def test_numerical_match(self, fitted_image_gmlvq):
        model, X = fitted_image_gmlvq
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)


class TestImageGTLVQExport:
    """ImageGTLVQ: CNN encoder + tangent distance + WTAC."""

    @pytest.fixture
    def fitted_image_gtlvq(self):
        from prosemble.models import ImageGTLVQ
        input_shape = (8, 8, 1)
        X, y = _make_image_data(40, input_shape, n_classes=2)
        model = ImageGTLVQ(
            n_prototypes_per_class=1, max_iter=10, lr=0.01,
            input_shape=input_shape, channels=[4], kernel_sizes=[3],
            latent_dim=3, subspace_dim=2, random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_model_type_detected(self, fitted_image_gtlvq):
        from prosemble.core.onnx_export import _identify_model_type
        model, X = fitted_image_gtlvq
        model_type, dist_type = _identify_model_type(model)
        assert model_type == 'supervised'
        assert dist_type == 'tangent'

    def test_numerical_match(self, fitted_image_gtlvq):
        model, X = fitted_image_gtlvq
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)


# ---------------------------------------------------------------------------
# NEW: LVQMLN and PLVQ (MLP encoder, prototypes in latent space)
# ---------------------------------------------------------------------------

class TestLVQMLNExport:
    """LVQMLN: MLP encoder + squared Euclidean + WTAC (latent protos)."""

    @pytest.fixture
    def fitted_lvqmln(self):
        from prosemble.models import LVQMLN
        X, y = _make_supervised_data(40, 4, n_classes=2)
        model = LVQMLN(
            n_prototypes_per_class=1, max_iter=10, lr=0.01,
            hidden_sizes=[8], latent_dim=3, random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_model_type_detected(self, fitted_lvqmln):
        from prosemble.core.onnx_export import _identify_model_type
        model, X = fitted_lvqmln
        model_type, dist_type = _identify_model_type(model)
        assert model_type == 'supervised'
        assert dist_type == 'squared_euclidean'

    def test_export_returns_model_proto(self, fitted_lvqmln):
        model, X = fitted_lvqmln
        onnx_model = model.export_onnx()
        assert isinstance(onnx_model, onnx.ModelProto)

    def test_numerical_match(self, fitted_lvqmln):
        model, X = fitted_lvqmln
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)


class TestPLVQExport:
    """PLVQ: MLP encoder + Gaussian mixture soft assignment."""

    @pytest.fixture
    def fitted_plvq(self):
        from prosemble.models import PLVQ
        X, y = _make_supervised_data(40, 4, n_classes=2)
        model = PLVQ(
            n_prototypes_per_class=2, max_iter=10, lr=0.01,
            hidden_sizes=[8], latent_dim=3, sigma=1.0,
            random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_model_type_detected(self, fitted_plvq):
        from prosemble.core.onnx_export import _identify_model_type
        model, X = fitted_plvq
        model_type, dist_type = _identify_model_type(model)
        assert model_type == 'plvq'
        assert dist_type == 'squared_euclidean'

    def test_export_returns_model_proto(self, fitted_plvq):
        model, X = fitted_plvq
        onnx_model = model.export_onnx()
        assert isinstance(onnx_model, onnx.ModelProto)

    def test_numerical_match(self, fitted_plvq):
        model, X = fitted_plvq
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)


# ---------------------------------------------------------------------------
# NEW: CBC models (reasoning matrices)
# ---------------------------------------------------------------------------

class TestCBCExport:
    """CBC: squared Euclidean + Gaussian similarity + CBCC reasoning."""

    @pytest.fixture
    def fitted_cbc(self):
        from prosemble.models import CBC
        X, y = _make_supervised_data(40, 4, n_classes=2)
        model = CBC(
            n_components=3, n_classes=2, sigma=1.0,
            max_iter=10, lr=0.01, random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_model_type_detected(self, fitted_cbc):
        from prosemble.core.onnx_export import _identify_model_type
        model, X = fitted_cbc
        model_type, dist_type = _identify_model_type(model)
        assert model_type == 'cbc'
        assert dist_type == 'squared_euclidean'

    def test_export_returns_model_proto(self, fitted_cbc):
        model, X = fitted_cbc
        onnx_model = model.export_onnx()
        assert isinstance(onnx_model, onnx.ModelProto)

    def test_numerical_match(self, fitted_cbc):
        model, X = fitted_cbc
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)


class TestImageCBCExport:
    """ImageCBC: CNN encoder + Gaussian similarity + CBCC reasoning."""

    @pytest.fixture
    def fitted_image_cbc(self):
        from prosemble.models import ImageCBC
        input_shape = (8, 8, 1)
        X, y = _make_image_data(40, input_shape, n_classes=2)
        model = ImageCBC(
            n_components=3, n_classes=2, sigma=1.0,
            max_iter=10, lr=0.01,
            input_shape=input_shape, channels=[4], kernel_sizes=[3],
            latent_dim=3, random_seed=42,
        )
        model.fit(X, y)
        return model, X

    def test_model_type_detected(self, fitted_image_cbc):
        from prosemble.core.onnx_export import _identify_model_type
        model, X = fitted_image_cbc
        model_type, dist_type = _identify_model_type(model)
        assert model_type == 'cbc'
        assert dist_type == 'squared_euclidean'

    def test_export_returns_model_proto(self, fitted_image_cbc):
        model, X = fitted_image_cbc
        onnx_model = model.export_onnx()
        assert isinstance(onnx_model, onnx.ModelProto)

    def test_numerical_match(self, fitted_image_cbc):
        model, X = fitted_image_cbc
        jax_preds = np.asarray(model.predict(X))
        ort_preds = _onnx_predict(model, X)
        npt.assert_array_equal(jax_preds, ort_preds)
