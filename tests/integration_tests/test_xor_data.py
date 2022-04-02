from unittest import TestCase

import numpy as np

from nnlib import Model
from nnlib.layers import FullyConnected
from nnlib.layers.activations import Sigmoid
from nnlib.layers.losses import MSE
from tests.utils import xor_data


class XorDataTest(TestCase):

    def test_prediction_on_init_weights(self):
        n = 50   # number of examples (data points)
        d = 2    # number of features (dimensionality of the data)
        h = 50   # number of neurons in the hidden layer
        k = 2    # label dimensionality 

        np.random.seed(0)
        X, y = xor_data(num_examples=n)

        model = Model()
        model.add(FullyConnected(input_size=d, output_size=h))
        model.add(Sigmoid())
        model.add(FullyConnected(input_size=h, output_size=k))
        model.add(MSE())

        y_pred = model.predict(X)

        # Check that y_pred is not None and that it has the right shape.
        self.assertIsNotNone(y_pred)
        self.assertEqual(y_pred.shape, (n, k))

   