"""
Tests for kernel-based clustering models.
"""
import numpy as np
import pytest
from prosemble.models.kfcm import KFCM
from prosemble.models.kpcm import KPCM
from prosemble.models.kafcm import KAFCM


class TestKFCM:
    """Tests for Kernel Fuzzy C-Means."""

    def test_kfcm_initialization(self, simple_2d_data):
        """Test KFCM initialization."""
        X, _ = simple_2d_data
        
        model = KFCM(
            data=X,
            c=3,
            m=2,
            num_iter=50,
            epsilon=0.00001,
            ord='fro',
            kernel='rbf',
            sigma=1.0
        )
        assert model.num_clusters == 3
        assert model.kernel == 'rbf'

    def test_kfcm_fit_predict(self, simple_2d_data):
        """Test KFCM fit and predict."""
        X, _ = simple_2d_data
        
        model = KFCM(
            data=X,
            c=3,
            m=2,
            num_iter=50,
            epsilon=0.00001,
            ord='fro',
            kernel='rbf',
            sigma=1.0,
            plot_steps=False
        )
        model.fit()
        
        labels = model.predict()
        assert labels.shape[0] == X.shape[0]
        assert set(labels).issubset({0, 1, 2})

    def test_kfcm_centroids(self, simple_2d_data):
        """Test KFCM learned centroids."""
        X, _ = simple_2d_data
        
        model = KFCM(
            data=X,
            c=3,
            m=2,
            num_iter=50,
            epsilon=0.00001,
            ord='fro',
            kernel='rbf',
            sigma=1.0,
            plot_steps=False
        )
        model.fit()
        
        centroids = model.final_centroids()
        assert centroids.shape == (3, 2)


class TestKPCM:
    """Tests for Kernel Possibilistic C-Means."""

    def test_kpcm_initialization(self, simple_2d_data):
        """Test KPCM initialization."""
        X, _ = simple_2d_data
        
        # KPCM needs initial centroids
        from prosemble.models.kfcm import KFCM
        kfcm = KFCM(
            data=X,
            c=3,
            m=2,
            num_iter=30,
            epsilon=0.00001,
            ord='fro',
            kernel='rbf',
            sigma=1.0,
            plot_steps=False
        )
        kfcm.fit()
        init_centroids = kfcm.final_centroids()
        
        model = KPCM(
            data=X,
            c=3,
            m=2,
            num_iter=50,
            epsilon=0.00001,
            ord='fro',
            kernel='rbf',
            sigma=1.0,
            set_prototypes=init_centroids
        )
        assert model.num_clusters == 3

    def test_kpcm_fit_predict(self, simple_2d_data):
        """Test KPCM fit and predict."""
        X, _ = simple_2d_data
        
        from prosemble.models.kfcm import KFCM
        kfcm = KFCM(
            data=X,
            c=3,
            m=2,
            num_iter=30,
            epsilon=0.00001,
            ord='fro',
            kernel='rbf',
            sigma=1.0,
            plot_steps=False
        )
        kfcm.fit()
        init_centroids = kfcm.final_centroids()
        
        model = KPCM(
            data=X,
            c=3,
            m=2,
            num_iter=50,
            epsilon=0.00001,
            ord='fro',
            kernel='rbf',
            sigma=1.0,
            set_prototypes=init_centroids,
            plot_steps=False
        )
        model.fit()
        
        labels = model.predict()
        assert labels.shape[0] == X.shape[0]


class TestKAFCM:
    """Tests for Kernel Adaptive Fuzzy C-Means."""

    def test_kafcm_initialization(self, simple_2d_data):
        """Test KAFCM initialization."""
        X, _ = simple_2d_data
        
        model = KAFCM(
            data=X,
            c=3,
            m=2,
            num_iter=50,
            epsilon=0.00001,
            ord='fro',
            kernel='rbf',
            sigma=1.0
        )
        assert model.num_clusters == 3

    def test_kafcm_fit_predict(self, simple_2d_data):
        """Test KAFCM fit and predict."""
        X, _ = simple_2d_data
        
        model = KAFCM(
            data=X,
            c=3,
            m=2,
            num_iter=50,
            epsilon=0.00001,
            ord='fro',
            kernel='rbf',
            sigma=1.0,
            plot_steps=False
        )
        model.fit()
        
        labels = model.predict()
        assert labels.shape[0] == X.shape[0]
