from cmath import isnan
import numpy as np
from neuralib.layers.layers import ComputationalLayer
from typing import Union
class Loss(ComputationalLayer):
    '''
    Generic Loss Class (inherits from ComputationalLayer since it is a layer without parameters)
    '''
    def __init__(self) -> None:
        super().__init__()
        self._pred_cache = None
        self._targets_cache = None

    def forward(self, y_pred: np.array, y_true: np.array):
        """Forward pass of the loss layer (i.e. compute the loss)

        Args:
            y_pred (n_samples x n_output_model): Predicted outputs of the model (inputs to the loss layer)
            y_true (n_samples x n_output_model): True outputs of the model (inputs to the loss layer)
        """
        assert (y_pred.shape == y_true.shape), "y_pred and targets must have same shape"
        self._pred_cache = y_pred
        self._targets_cache = y_true

    
    def backward(self) -> np.array:
        """Backward pass of the loss layer (i.e. compute the gradient of the loss wrt prediction)
        Label and prediction are stored in the cache.       

        Returns:
            np.array: Gradient of the loss with respect to the prediction (tensor)
        """
        assert (self._targets_cache.shape == self._pred_cache.shape), "y_pred and targets must have same shape"

class MSE(Loss):
    '''
    Mean Squared Error Loss Class (inherits from Loss class)
    '''
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, y_pred, y_true) -> Union[np.array, float]:
        """Compute the residual and MSE loss

        Args:
            y_pred (n_samples x n_output_model): Predicted outputs of the model (inputs to the loss layer)
            y_true (n_samples x n_output_model): True outputs of the model (inputs to the loss layer)

        Returns:
            Union[np.array, float]: residual and MSE loss
        """
        super().forward(y_pred, y_true)

        residual = y_pred - y_true
        try:
            error = np.sum(residual**2)/(2*y_pred.shape[0])
        except OverflowError as err:
            # Set error to infinity 
            print('Overflowed after ', error, err)
            error = np.inf
        except:
            raise
        if isnan(error):
            error = np.inf
            
        assert(error >= 0), "Error must be non-negative"

        return residual, error  

    def backward(self) -> np.array:
        """Backward pass of the loss layer (i.e. compute the gradient of the loss wrt prediction)       
        Label and prediction are stored in the cache.       

        Returns:
            np.array: Gradient of the MSE error wrt y_pred (tensor)
        """
        assert (self._pred_cache is not None and self._targets_cache is not None), "Forward pass required before backward pass"
        super().backward()
        d_mse_d_y_pred = (self._pred_cache - self._targets_cache) / self._pred_cache.shape[0]
        return d_mse_d_y_pred

