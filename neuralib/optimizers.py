import numpy as np
from abc import ABC,abstractmethod

class Optimizer(ABC):
    def __init__(self, lr: float) -> None:
        self.lr = lr

    @abstractmethod
    def step(self, dweights: np.array) -> np.array:
        """Abstract method to update the weights based on the gradient of the loss with respect to the weights.

        Args:
            dweights (np.array): The gradient of the loss with respect to the weights (tensor).

        Returns:
            np.array: The increment of the weights (tensor).
        """
        pass

class VGD(Optimizer):
    '''
    Stochastic Gradient Descent
    '''
    def step(self, dweights: np.array) -> np.array:
        # Vanilla Gradient Descent update rule
        return -self.lr * dweights