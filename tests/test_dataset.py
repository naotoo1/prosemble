"""prosemble dataset test suite"""

import unittest
import numpy as np
from prosemble import datasets


class TestWDBC(unittest.TestCase):
    def setUp(self):
        self.data,self.labels = datasets.DATA().breast_cancer

    def test_size(self):
        self.assertEqual(len(self.data), 569)
        self.assertEqual(len(self.labels), 569)

    def test_dimensions(self):
        self.assertEqual(self.data.shape[1], 30)

    def test_unique_labels(self):
        self.assertEqual(len(np.unique(self.labels)), 2)

    def test_dims_selection(self):
        self.input_features = self.data[:, [0, 1]]
        self.assertEqual(self.input_features.shape[1], 2)


class TestS1(unittest.TestCase):
    def setUp(self):
        self.data,self.labels = datasets.DATA().S_1

    def test_size(self):
        self.assertEqual(len(self.data), 150)
        self.assertEqual(len(self.labels), 150)

    def test_dimensions(self):
        self.assertEqual(self.data.shape[1], 2)

    def test_unique_labels(self):
        self.assertEqual(len(np.unique(self.labels)), 2)

    def test_dims_selection(self):
        self.input_features = self.data[:, [0, 1]]
        self.assertEqual(self.data.shape[1], 2)




class TestS2(unittest.TestCase):
    def setUp(self):
        self.data,self.labels = datasets.DATA().S_2

    def test_size(self):
        self.assertEqual(len(self.data), 200)
        self.assertEqual(len(self.labels), 200)

    def test_dimensions(self):
        self.assertEqual(self.data.shape[1], 2)

    def test_unique_labels(self):
        self.assertEqual(len(np.unique(self.labels)), 2)

    def test_dims_selection(self):
        self.input_features = self.data[:, [0, 1]]
        self.assertEqual(self.data.shape[1], 2)



class TestMnits(unittest.TestCase):
    def setUp(self):
        self.data,self.labels = datasets.DATA().mnist

    def test_size(self):
        self.assertEqual(len(self.data), 70000)
        self.assertEqual(len(self.labels), 70000)

    def test_dimensions(self):
        self.assertEqual(self.data.shape[1], 28)

    def test_unique_labels(self):
        self.assertEqual(len(np.unique(self.labels)), 10)

