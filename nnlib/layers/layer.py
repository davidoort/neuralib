import numpy as np
from nnlib.utils import initialize_weights
from abc import ABC,abstractmethod
from nnlib.optimizers import Optimizer

class ComputationalLayer(ABC):
    '''
    This layer does not have parameters, so it does not need to be updated.
    The forward method is used to compute the output of the layer.
    The backward method is used to compute the gradient of the layer inputs with respect to the layer outputs.
    '''
    def __init__(self) -> None:
        # self.grad = False
        pass

    @abstractmethod
    def forward(self, inputs):
        # TODO: add docstring? 
        # TODO: add shape checks
        pass
    
    @abstractmethod
    def backward(self, inputs, gradients):
        # TODO: define a general interface for backpropagation
        pass
    
        

class GradLayer(ComputationalLayer):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        # self.grad = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = initialize_weights(input_size, output_size)
        self.biases = initialize_weights(1, output_size)
        self.dweights = np.zeros(self.weights.shape)
        self.dbiases = np.zeros(self.biases.shape)

    def update(self, optimizer: Optimizer) -> None:
        self.weights += optimizer.step(self.dweights)
        self.biases += optimizer.step(self.dbiases)
