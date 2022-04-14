from unittest import TestCase

import numpy as np

from neuralib import SequentialModel
from neuralib.architectures import MLP
from neuralib.layers import Linear
from neuralib.layers.activations import Sigmoid
from neuralib.layers.layers import GradLayer, Identity
from neuralib.layers.losses import MSE

class MLPTest(TestCase):

    # Create custom MLP models
    def setUp(self) -> None:
        self.input_size = 2    # number of features (dimensionality of the data)
        self.hidden_size = 50   # number of neurons in the hidden layer
        self.target_size = 1   # label dimensionality 

        # manually create shallow architecture
        self.manual_model = SequentialModel([Linear(input_size=self.input_size, output_size=self.hidden_size), 
                                   Sigmoid(), 
                                   Linear(input_size=self.hidden_size, output_size=self.target_size), 
                                   MSE()])

        self.mlp_model = MLP(input_size=self.input_size, hidden_size=self.hidden_size, output_size=self.target_size, activations=[Sigmoid(), Identity()], loss=MSE())
    
    def test_model_matching(self):
        # Test that the manual model matches the MLP model
        self.assertEqual(self.manual_model.layers, self.mlp_model.layers)

    def test_get_params(self):
        params = self.manual_model.get_params()
        self.assertEqual(len(params), len([l for l in self.manual_model.layers if isinstance(l, GradLayer)]))

        # Compare total number of params. This currently only works because the layers are linear
        pred_n_params = sum([p['n_params'] for p in params])  
        n_params = sum([p['weights'].size + p['biases'].size for p in params])
        self.assertEqual(pred_n_params, n_params)
