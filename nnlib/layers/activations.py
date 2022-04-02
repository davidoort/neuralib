import numpy as np
from nnlib.layers.layers import ComputationalLayer

class Sigmoid(ComputationalLayer):
    def __init__(self):
        pass
    def forward(self, x):
        """
        Computes the sigmoid function sigm(input) = 1/(1+exp(-input))
        """
        return 1/(1+np.exp(-x))
        
    def backward(self, x, sigmoided: bool = False):
        """
        Computes the derivative of sigmoid funtion. sigmoid(y) * (1.0 - sigmoid(y)). 
        The way we implemented this requires that the input y is already sigmoided
        """
        if sigmoided:
            return np.multiply(x, 1.0-x)
        else:
            return np.multiply(self.forward(x), 1.0-self.forward(x)) 

# def sigmoid(x):
#   """
#   Computes the sigmoid function sigm(input) = 1/(1+exp(-input))
#   """
#   return 1/(1+np.exp(-x))

# def d_sigmoid(x, sigmoided: bool=True):
#   """
#   Computes the derivative of sigmoid funtion. sigmoid(y) * (1.0 - sigmoid(y)). 
#   The way we implemented this requires that the input y is already sigmoided
#   """
#   if sigmoided:
#     return np.multiply(x, 1.0-x)
#   else:
#     return np.multiply(sigmoid(x), 1.0-sigmoid(x)) 
