import numpy as np
from neuralib.layers.layers import ComputationalLayer

class Sigmoid(ComputationalLayer):
    def __init__(self):
        super().__init__()
        
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

    def __eq__(self, other) -> bool:
        if super().__eq__(other):
            if isinstance(other, Sigmoid):
                return True
            else:
                return False
        return False

class ReLU(ComputationalLayer):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        """
        Computes the ReLU function ReLU(input) = max(0, input)
        """
        super().forward(input)
        return self._relu(input)
        
    def backward(self, grad_top = None):
        """
        Performs a backward pass of the ReLU function.
        """
        if grad_top is None:
            return self._d_relu(self._input_cache)

        # Verify that the input cache is of the same shape as grad_top
        assert self._input_cache.shape == grad_top.shape, "Input cache and grad_top must have the same shape"

        return self._d_relu(self._input_cache) * grad_top

    def _relu(self, x: np.array) -> np.array:
        """Compute the ReLU function ReLU(x) = max(0, x) in element-wise fashion.

        Args:
            x (np.array): input matrix/vector

        Returns:
            np.array: output matrix/vector of the same size as the input
        """
        return np.maximum(0, x)

    def _d_relu(self, x: np.array) -> np.array:
        """Computes the derivative of ReLU funtion. 

        Args:
            x (np.array): input matrix/vector

        Returns:
            np.array: output matrix/vector of the same size as the input
        """

        # The derivative is 0 when the input is < 0, and 1 otherwise
        d_relu_x = self._relu(np.sign(x)) 

        # Assert that no element in d_relu_x is < 0 or > 1
        assert np.all(d_relu_x >= 0) and np.all(d_relu_x <= 1), "d_relu_x must be between 0 and 1"
        return d_relu_x

    def __eq__(self, other) -> bool:
        if super().__eq__(other):
            if isinstance(other, ReLU):
                return True
            else:
                return False
        return False

class Tanh(ComputationalLayer):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        """
        Computes the Tanh function Tanh(input) = (exp(input) - exp(-input))/(exp(input) + exp(-input))
        """
        super().forward(input)
        return self._tanh(input)
        
    def backward(self, grad_top = None):
        """
        Computes the derivative of Tanh funtion. Tanh(y) * (1.0 - Tanh(y)). 
        The way we implemented this requires that the input y is already Tanh
        """
        if grad_top is None:
            return self._d_tanh(self._input_cache)

        # Verify that the input cache is of the same shape as grad_top
        assert self._input_cache.shape == grad_top.shape, "Input cache and grad_top must have the same shape"

        return self._d_tanh(self._input_cache) * grad_top

    def _tanh(self, x: np.array) -> np.array:
        """Compute the Tanh function Tanh(x) = (exp(x) - exp(-x))/(exp(x) + exp(-x)) in element-wise fashion.

        Args:
            x (np.array): input matrix/vector

        Returns:
            np.array: output matrix/vector of the same size as the input
        """
        return np.tanh(x)

    def _d_tanh(self, x: np.array) -> np.array:
        """Computes the derivative of Tanh funtion.

        Args:
            x (np.array): input matrix/vector

        Returns:
            np.array: output matrix/vector of the same size as the input
        """
        return 1.0 - np.power(np.tanh(x), 2)

    def __eq__(self, other) -> bool:
        if super().__eq__(other):
            if isinstance(other, Tanh):
                return True
            else:
                return False
        return False