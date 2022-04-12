from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from neuralib.layers.activations import ReLU, Sigmoid, Tanh
from tests.utils.numerical_grad import numerical_grad


class ActivationsTest(TestCase):

    def setUp(self):
        self.X = np.array([
            [1, -2, 1],
            [-1, 1, 5]], dtype=float)

        self.layers = [Sigmoid(), ReLU(), Tanh()]

    def test_backward(self):
        for layer in self.layers:
            layer.forward(self.X)
            d_X = layer.backward()

            def forward_as_func_of_X(X):
                return layer.forward(X)

            assert_array_almost_equal(
                numerical_grad(forward_as_func_of_X, self.X),
                d_X
            )

class ReLUTest(TestCase):

    def setUp(self):
        self.X = np.array([
            [1, -2, 1],
            [-2, 1, 5]], dtype=float)

        self.layer = ReLU()

    def test_forward(self):
        self.layer.forward(self.X)
        assert_array_equal(
            self.layer.forward(self.X),
            np.array([
                [1, 0, 1],
                [0, 1, 5]], dtype=float)
        )

class TanhTest(TestCase):

    def setUp(self):
        self.X = np.array([
            [1, -2, 1],
            [-2, 1, 5]], dtype=float)

        self.layer = Tanh()

    def test_forward(self):
        self.layer.forward(self.X)
        assert_array_almost_equal(
            self.layer.forward(self.X),
            np.array([
                [0.76159416, -0.96402758, 0.76159416],
                [-0.96402758, 0.76159416, 1.0]], dtype=float), 
                decimal=4
        )

