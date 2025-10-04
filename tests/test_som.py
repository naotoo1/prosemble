"""
Tests for Self-Organizing Map (SOM).
"""
import numpy as np
import pytest
from prosemble.models.som import SOM


class TestSOMBasic:
    """Basic tests for SOM."""

    def test_som_initialization(self, simple_2d_data):
        """Test SOM initialization."""
        X, _ = simple_2d_data
        
        model = SOM(
            data=X,
            c=9,  # 3x3 grid
            num_iter=100,
            epsilon=0.00001,
            ord='fro'
        )
        assert model.num_clusters == 9

    def test_som_fit(self, simple_2d_data):
        """Test SOM fitting."""
        X, _ = simple_2d_data
        
        model = SOM(
            data=X,
            c=9,
            num_iter=50,
            epsilon=0.00001,
            ord='fro',
            plot_steps=False
        )
        model.fit()
        
        # Check that weights were learned
        # SOM should have learned prototypes
        assert hasattr(model, 'fit_cent')

    def test_som_predict(self, simple_2d_data):
        """Test SOM prediction."""
        X, _ = simple_2d_data
        
        model = SOM(
            data=X,
            c=9,
            num_iter=50,
            epsilon=0.00001,
            ord='fro',
            plot_steps=False
        )
        model.fit()
        
        labels = model.predict()
        assert labels.shape[0] == X.shape[0]
