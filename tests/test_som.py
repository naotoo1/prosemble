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
            c=9,  # Not used but required parameter
            num_iter=100,
            init_lr=0.5,
            sigma=1.0
        )
        assert model.number_cluster == 9
        assert model.num_iter == 100
        assert model.learning_rate == 0.5

    def test_som_fit(self, simple_2d_data):
        """Test SOM fitting."""
        X, _ = simple_2d_data
        
        model = SOM(
            data=X,
            c=9,
            num_iter=50,  # Use small number for faster tests
            init_lr=0.5,
            sigma=1.0
        )
        som_map = model.fit()
        
        # Check that SOM map was created
        assert som_map is not None
        assert som_map.shape[2] == X.shape[1]  # Should match input dimensions

    def test_som_get_cluster_and_predict(self, simple_2d_data):
        """Test SOM clustering and prediction."""
        X, y = simple_2d_data
        
        model = SOM(
            data=X,
            c=9,
            num_iter=50,
            init_lr=0.5,
            sigma=1.0
        )
        model.fit()
        
        # Get cluster map
        label_map, final_map = model.get_cluster(y)
        assert label_map.shape == (model.grid, model.grid)
        assert final_map.shape == (model.grid, model.grid)
        
        # Test prediction
        predictions = model.predict(X[:10])
        assert len(predictions) == 10

    def test_som_grid_size(self, simple_2d_data):
        """Test SOM grid size calculation."""
        X, _ = simple_2d_data
        
        model = SOM(
            data=X,
            c=9,
            num_iter=50,
            init_lr=0.5,
            sigma=1.0
        )
        
        # Grid size should be calculated based on data
        expected_grid = int(np.sqrt(5 * np.sqrt(X.shape[0])))
        assert model.grid == expected_grid

    def test_som_default_num_iter(self, simple_2d_data):
        """Test SOM with default num_iter."""
        X, _ = simple_2d_data
        
        model = SOM(
            data=X,
            c=9,
            num_iter=None,  # Should use default
            init_lr=0.5,
            sigma=1.0
        )
        
        # Default should be 500 * grid * grid
        expected_iter = 500 * model.grid * model.grid
        assert model.num_iter == expected_iter
