from unittest import TestCase

import numpy as np

from neuralib import Model
from neuralib.layers import Linear
from neuralib.layers.activations import Sigmoid
from neuralib.layers.losses import MSE
from neuralib.optimizers import SGD
from tests.utils import xor_data


class XorDataTest(TestCase):

    # Create custom model
    def setUp(self) -> None:
        self.batch_dim = 50   # number of examples (data points)
        self.input_dim = 2    # number of features (dimensionality of the data)
        self.hidden_dim = 50   # number of neurons in the hidden layer
        self.target_dim = 2    # label dimensionality 

        np.random.seed(0)
        self.X, self.y = xor_data(num_examples=self.batch_dim)

        self.model = Model()
        self.model.add(Linear(input_size=self.input_dim, output_size=self.hidden_dim))
        self.model.add(Sigmoid())
        self.model.add(Linear(input_size=self.hidden_dim, output_size=self.target_dim))
        self.model.add(MSE())

    def test_prediction_on_init_weights(self):

        y_pred = self.model.predict(self.X)

        # Check that y_pred is not None and that it has the right shape.
        self.assertIsNotNone(y_pred)
        self.assertEqual(y_pred.shape, (self.batch_dim, self.target_dim))

    # def test_training_acc_on_custom_model(self):
    #     # Train the model
    #     self.model.train(self.X, self.y, batch_size=2, epochs=100, optimizer=SGD(lr=0.1))


    #     X_test = np.array([[0,0], [0,1], [1,0], [1,1]])
    #     y_test = np.array([[0], [1], [1], [0]])
    #     y_pred = self.model.predict(X_test)

    #     # Check that the model has learned the XOR function
    #     self.assertTrue(np.allclose(y_pred, y_test))



   