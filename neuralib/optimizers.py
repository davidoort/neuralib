import numpy as np
from abc import ABC,abstractmethod

class Optimizer(ABC):
    def __init__(self, lr) -> None:
        self.lr = lr

    @abstractmethod
    def step(self, dweights):
        pass

class SGD(Optimizer):
    def step(self, dweights):
        return -self.lr * dweights