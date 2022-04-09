from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from neuralib.layers.activations import Sigmoid
from tests.utils.numerical_grad import numerical_grad


class SigmoidTest(TestCase):

    def setUp(self):
        self.X = np.array([
            [1, -2, 1],
            [0, 1, 5]], dtype=float)

        self.layer = Sigmoid()

    # def test_forward_with_ground_truths(self):
    #     out = self.layer.forward(self.X)

    #     self.assertAlmostEqual(loss, self.expected_out, places=6)

    def test_backward(self):
        self.layer.forward(self.X)
        d_X = self.layer.backward()

        layer = self.layer

        def forward_as_func_of_X(X):
            return layer.forward(X)

        assert_array_almost_equal(
            numerical_grad(forward_as_func_of_X, self.X),
            d_X
        )