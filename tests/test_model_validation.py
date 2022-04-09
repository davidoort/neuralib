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