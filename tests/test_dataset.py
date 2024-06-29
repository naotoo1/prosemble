"""prosemble dataset test suite"""

import unittest
import numpy as np
from prosemble.datasets import DATA


class TestWDBC(unittest.TestCase):
    def setUp(self):
        self.data = DATA().breast_cancer
        self.input_features = self.data.input_data
        self.labels = self.data.labels

    def test_size(self):
        self.assertEqual(len(self.input_features), 569)
        self.assertEqual(len(self.labels), 569)

    def test_dimensions(self):
        self.assertEqual(self.input_features.shape[1], 30)

    def test_unique_labels(self):
        self.assertEqual(len(np.unique(self.labels)), 2)

    def test_dims_selection(self):
        self.input_features = self.input_features[:, [2, 3]]
        self.assertEqual(self.input_features.shape[1], 2)

    def tearDown(self):
        del self.input_features , self.labels 