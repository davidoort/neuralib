import numpy as np
from neuralib.layers.layers import ComputationalLayer
from typing import Union
class Loss(ComputationalLayer):
    def __init__(self) -> None:
        super().__init__()
    #     self.targets = None

    # # This is a static method where you pass the class
    # @classmethod
    # def from_targets(cls, targets):
    #     # Calls the constructor of the class
    #     loss = cls()
    #     loss.add_targets(targets)
    #     return loss

    # Static method allows you to call the method without instantiating an instance of the class without passing the instance as an argument
    # @staticmethod
    # def loss(y_pred, y):
    #    pass

    # # This function is so that you can call an instance of mse like instance(y_pred, y_true)
    # def __call__(self, y_pred, y_true):
    #     return self.loss(y_pred, y_true)

    # def add_targets(self, targets):
    #     self.targets = targets

    def forward(self, y_pred, y_true):
        assert (y_pred.shape == y_true.shape), "y_pred and targets must have same shape"

    
    def backward(self, y_pred, y_true):
        assert (y_pred.shape == y_true.shape), "y_pred and targets must have same shape"


# TODO: Write docs about expected input and output shapes
class MSE(Loss):
    def __init__(self) -> None:
        super().__init__()

    # Static method allows you to call the method without instantiating an instance of the class without passing the instance as an argument
    # @staticmethod
    # def loss(y_pred, y):
    #     super().loss(y_pred, y)
    #     residual = y_pred - y
    #     error = np.sum(residual**2)/(2*np.size(y_pred))
    #     return residual, error
    
    def forward(self, y_pred, y_true) -> Union[np.array, float]:
        super().forward(y_pred, y_true)
        residual = y_pred - y_true
        error = np.sum(residual**2)/(2*np.size(y_pred))
        return residual, error  

    def backward(self, y_pred, y_true) -> np.array:
        # TODO: Remember that I am not not normalizing th gradient by the batch size (will probably have to use np.mean instead of a dot product or sum somewhere)
        super().backward(y_pred, y_true)
        d_loss_d_y_pred = y_pred - y_true
        return d_loss_d_y_pred
        # return (y_pred - y_true)/np.size(y_pred)

