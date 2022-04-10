from copy import deepcopy
from unittest import TestCase

import numpy as np

from neuralib.architectures import Model
from neuralib.layers import MSE, Sigmoid, Linear

from numpy.testing import assert_equal

class ValidationTest(TestCase):

    def setUp(self):
        self.model = Model()
        self.model.add(MSE())

    def test_loss_validation(self):
        assert_equal(self.model.validate(), True)

    def test_with_invalid_order_linear(self):
        model = deepcopy(self.model)
        model.add(Linear(input_size=1, output_size=1))
        assert_equal(model.validate(), False)

    def test_with_invalid_order_activation(self):
        model = deepcopy(self.model)
        model.add(Sigmoid())
        assert_equal(model.validate(), False)
    
    def test_with_valid_order_activation(self):
        self.model.add_front(Sigmoid())
        assert_equal(self.model.validate(), True)