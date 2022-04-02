import numpy as np
from neuralib.layers.layers import ComputationalLayer

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
    
    # This is a static method where you pass the class
    @classmethod
    def from_targets(cls, targets):
        # Calls the constructor of the class
        mse = cls()
        mse.add_targets(targets)
        return mse

    # TODO: now that this method (mse, or loss) exists, forward and __call__ can be in the abstract class?  
    # Static method allows you to call the method without instantiating an instance of the class without passing the instance as an argument
    @staticmethod
    def mse(y_pred, y):
        residual = y_pred - y
        error = np.sum(residual**2)/(2*np.size(y_pred))
        return residual, error

    # This function is so that you can call an instance of mse like instance(y_pred, y_true)
    def __call__(self, y_pred, y_true):
        return self.mse(y_pred, y_true)
    
    def forward(self, y_pred):
        super().forward(y_pred)
        return self.mse(y_pred, self.targets)   

    def backward(self, y_pred):
        return (y_pred - self.targets)/np.size(y_pred)

