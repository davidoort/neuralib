from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal, assert_almost_equal

from neuralib.layers.losses import MSE
from tests.utils.numerical_grad import numerical_grad


class MSETest(TestCase):

    def setUp(self):
        self.y = np.array([
            [1, -2, 1],
            [0, 1, 5]], dtype=float)
        self.y_true = np.array([
            [1, -2, 1],
            [1, -1, 5]], dtype=float)


        self.layer = MSE()

    def test_forward_with_ground_truths(self):
        res, err = self.layer.forward(self.y, self.y)
        assert_array_equal(res, np.array([[0, 0, 0], [0, 0, 0]]))
        assert_equal(err, 0)

        res, err = self.layer.forward(self.y, self.y_true)
        assert_array_equal(res, np.array([[0, 0, 0], [-1, 2, 0]]))
        assert_almost_equal(err, (1**2 + 2**2) / 2 / 2)


    def test_backward(self):
        self.layer.forward(self.y, self.y_true)
        d_X = self.layer.backward()

        layer = self.layer

        def forward_as_func_of_X(X):
            _, err = layer.forward(X, self.y_true)
            return err

        assert_array_almost_equal(
            numerical_grad(forward_as_func_of_X, self.y),
            d_X
        )