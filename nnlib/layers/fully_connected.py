import numpy as np
from nnlib.layers import GradLayer

class FullyConnected(GradLayer):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__(input_size, output_size)


    def forward(self, inputs):
        return inputs @ self.weights + self.biases

    def backward(self, inputs, gradients):
        self.dweights  = []
        self.dbiases = []
        return []
    

# Can also be called linear or feed-forward layer
def dense(inputs, weights):
    """A simple dense layer."""
    return np.matmul(inputs, weights)