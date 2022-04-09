import numpy as np
from neuralib.layers.layers import ComputationalLayer
from typing import Union
class Loss(ComputationalLayer):
    def __init__(self) -> None:
        super().__init__()
        self._pred_cache = None
        self._targets_cache = None

    def forward(self, y_pred, y_true):
        assert (y_pred.shape == y_true.shape), "y_pred and targets must have same shape"
        self._pred_cache = y_pred
        self._targets_cache = y_true

    
    def backward(self, y_pred, y_true):
        assert (y_pred.shape == y_true.shape), "y_pred and targets must have same shape"


# TODO: Write docs about expected input and output shapes
class MSE(Loss):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, y_pred, y_true) -> Union[np.array, float]:
        super().forward(y_pred, y_true)

        residual = y_pred - y_true
        error = np.sum(residual**2)/(2*y_pred.shape[0])
        assert(error >= 0), "Error must be non-negative"

        return residual, error  

    def backward(self) -> np.array:
        assert (self._pred_cache is not None and self._targets_cache is not None), "Forward pass required before backward pass"
        super().backward(self._pred_cache, self._targets_cache)
        d_mse_d_y_pred = (self._pred_cache - self._targets_cache) / self._pred_cache.shape[0]
        # clear_cache()
        return d_mse_d_y_pred

