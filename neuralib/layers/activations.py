import numpy as np
from neuralib.layers.layers import ComputationalLayer

class Sigmoid(ComputationalLayer):
    def __init__(self):
        pass
    def forward(self, input):
        """
        Computes the sigmoid function sigm(input) = 1/(1+exp(-input))
        """
        super().forward(input)
        return self._sigmoid(input)
        
    def backward(self, grad_top = None):
        """
        Computes the derivative of sigmoid funtion. sigmoid(y) * (1.0 - sigmoid(y)). 
        The way we implemented this requires that the input y is already sigmoided
        """
        if grad_top is None:
            return self._d_sigmoid(self._input_cache)

        # Verify that the input cache is of the same shape as grad_top
        assert self._input_cache.shape == grad_top.shape, "Input cache and grad_top must have the same shape"

        # TODO: Verify that element-wise multiplication is correct here
        return self._d_sigmoid(self._input_cache) * grad_top

    def _sigmoid(self, x: np.array) -> np.array:
        """Compute the sigmoid function sigm(x) = 1/(1+exp(-x)) in element-wise fashion.

        Args:
            x (np.array): input matrix/vector

        Returns:
            np.array: output matrix/vector of the same size as the input
        """
        return 1/(1+np.exp(-x))

    def _d_sigmoid(self, x: np.array) -> np.array:
        """Computes the derivative of sigmoid funtion. sigmoid(y) * (1.0 - sigmoid(y)). 

        Args:
            x (np.array): input matrix/vector

        Returns:
            np.array: output matrix/vector of the same size as the input
        """
        return np.multiply(self._sigmoid(x), 1.0-self._sigmoid(x))
