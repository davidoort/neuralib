from unittest import TestCase

import warnings
import pytest

import numpy as np

from neuralib import Model
from neuralib.layers import Linear
from neuralib.layers.activations import Sigmoid
from neuralib.layers.layers import GradLayer
from neuralib.layers.losses import MSE
from neuralib.optimizers import VGD
from neuralib.utils import xor_data


class XorDataTest(TestCase):

    # Create custom model
    def setUp(self) -> None:
        # self.batch_dim = 4   # number of examples (data points)
        self.input_dim = 2    # number of features (dimensionality of the data)
        self.hidden_dim = 50   # number of neurons in the hidden layer
        self.target_dim = 1   # label dimensionality 

        self.X = np.array([[0,0], [0,1], [1,0], [1,1]])
        self.y = np.array([ [0],   [1],   [1],   [0]])
        # self.X, self.y = xor_data(num_examples=self.batch_dim)

        # in one line
        self.model = Model([Linear(input_size=self.input_dim, output_size=self.hidden_dim), Sigmoid(), Linear(input_size=self.hidden_dim, output_size=self.target_dim), MSE()])

    def test_prediction_on_init_weights(self):

        y_pred = self.model.predict(self.X)

        # Check that y_pred is not None and that it has the right shape.
        self.assertIsNotNone(y_pred)
        self.assertEqual(y_pred.shape, (self.y.shape[0], self.target_dim))

    def test_training_acc_on_custom_model(self):
        # Train the model
        self.model.train(self.X, self.y, batch_size=4, epochs=10000, optimizer=VGD(lr=0.1))


        # Test on training data
        y_pred = self.model.predict(self.X)

        # Check that the model has learned the XOR function
        self.assertTrue(np.allclose(y_pred, self.y))

    @pytest.mark.filterwarnings("ignore")
    def test_xor_without_activation(self):
        model = self.model
        
        # Pop the activation layer
        model.pop(1)

        print(model.layers)
        # Train the model
        model.train(self.X, self.y, batch_size=4, epochs=10000, optimizer=VGD(lr=0.1))

        # Test on training data
        y_pred = model.predict(self.X)

        # Check that the model has learned the XOR function
        self.assertFalse(np.allclose(y_pred, self.y))

    def test_get_params(self):
        params = self.model.get_params()
        self.assertEqual(len(params), len([l for l in self.model.layers if isinstance(l, GradLayer)]))

        # Compare total number of params. This currently only works because the layers are linear
        pred_n_params = sum([p['n_params'] for p in params])  
        n_params = sum([p['weights'].size + p['biases'].size for p in params])
        self.assertEqual(pred_n_params, n_params)

   