"""
Tests for BGPC (Basic Graded Possibilistic Clustering).
"""
import numpy as np
import pytest
from prosemble.models.bgpc import BGPC


class TestBGPCBasic:
    """Basic tests for BGPC."""

    def test_bgpc_initialization(self, simple_2d_data):
        """Test BGPC initialization."""
        X, _ = simple_2d_data
        
        model = BGPC(
            data=X,
            c=3,
            num_iter=50,
            a_f=0.8,
            b_f=0.1,
            epsilon=0.00001,
            ord='fro',
            plot_steps=False
        )
        assert model.num_clusters == 3

    def test_bgpc_fit(self, simple_2d_data):
        """Test BGPC fitting."""
        X, _ = simple_2d_data
        
        model = BGPC(
            data=X,
            c=3,
            num_iter=50,
            a_f=0.8,
            b_f=0.1,
            epsilon=0.00001,
            ord='fro',
            plot_steps=False
        )
        model.fit()
        
        # Should have learned centroids
        assert len(model.fit_cent) > 0

    def test_bgpc_predict(self, simple_2d_data):
        """Test BGPC prediction."""
        X, _ = simple_2d_data
        
        model = BGPC(
            data=X,
            c=3,
            num_iter=50,
            a_f=0.8,
            b_f=0.1,
            epsilon=0.00001,
            ord='fro',
            plot_steps=False
        )
        model.fit()
        
        labels = model.predict()
        assert labels.shape[0] == X.shape[0]

    def test_bgpc_predict_new(self, iris_train_test_split):
        """Test BGPC prediction on new data."""
        X_train, X_test, y_train, _ = iris_train_test_split
        
        model = BGPC(
            data=X_train,
            c=3,
            num_iter=50,
            a_f=0.8,
            b_f=0.1,
            epsilon=0.00001,
            ord='fro',
            plot_steps=False
        )
        model.fit()
        
        predictions = model.predict_new(X_test)
        assert len(predictions) == len(X_test)
