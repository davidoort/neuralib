from unittest import TestCase

import warnings
import pytest

import numpy as np

from neuralib import SequentialModel
from neuralib.architectures import MLP
from neuralib.layers import Linear
from neuralib.layers.activations import Sigmoid
from neuralib.layers.layers import GradLayer, Identity
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
        self.manual_model = SequentialModel([Linear(input_size=self.input_dim, output_size=self.hidden_dim), Sigmoid(), Linear(input_size=self.hidden_dim, output_size=self.target_dim), MSE()])
    
        self.mlp_model = MLP(input_size=self.input_dim, hidden_size=self.hidden_dim, output_size=self.target_dim, activations=[Sigmoid(), Identity()], loss=MSE())

        # TODO: add more variations to this list (e.g. iterate over different activations, loss functions, etc.)
        self.models = [self.manual_model, self.mlp_model]
    def test_prediction_on_init_weights(self):
        for model in self.models:
            y_pred = model.predict(self.X)

            # Check that y_pred is not None and that it has the right shape.
            self.assertIsNotNone(y_pred)
            self.assertEqual(y_pred.shape, (self.y.shape[0], self.target_dim))

    def test_training_acc_on_custom_model(self):
        for model in self.models:
            # Train the model
            model.train(self.X, self.y, batch_size=4, epochs=10000, optimizer=VGD(lr=0.1))


            # Test on training data
            y_pred = model.predict(self.X)

            # Check that the model has learned the XOR function
            self.assertTrue(np.allclose(y_pred, self.y))

    @pytest.mark.filterwarnings("ignore")
    def test_xor_without_activation(self):
        for model in self.models:

            # Pop the activation layer
            model.pop(1)

            print(model.layers)
            # Train the model
            model.train(self.X, self.y, batch_size=4, epochs=10000, optimizer=VGD(lr=0.1))

            # Test on training data
            y_pred = model.predict(self.X)

            # Check that the model has learned the XOR function
            self.assertFalse(np.allclose(y_pred, self.y))

   