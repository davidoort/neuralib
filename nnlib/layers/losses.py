import numpy as np
from nnlib.layers.layer import ComputationalLayer

class Loss(ComputationalLayer):
    def __init__(self) -> None:
        super().__init__()
        self.targets = None

    def add_targets(self, targets):
        self.targets = targets

    def forward(self, y_pred):
        assert (self.targets is not None), "Targets not set, cannot compute loss"
        assert (y_pred.shape == self.targets.shape), "y_pred and targets must have same shape"

# TODO: Write docs about expected input and output shapes
class MSE(Loss):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, y_pred):
        super().forward(y_pred)
        residual = y_pred - self.targets
        error = np.sum(residual**2)/(2*np.size(y_pred))
        return residual, error

    def backward(self, y_pred):
        return (y_pred - self.targets)/np.size(y_pred)
    

# def mse(y_pred, y):
#     residual = y_pred - y
#     error = np.sum(residual**2)/(2*np.size(y_pred))
#     return residual, error