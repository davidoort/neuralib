import numpy as np
from neuralib.utils import initialize_weights
from abc import ABC,abstractmethod
from neuralib.optimizers import Optimizer

class ComputationalLayer(ABC):
    '''
    This layer does not have parameters, so it does not need to be updated.
    The forward method is used to compute the output of the layer.
    The backward method is used to compute the gradient of the layer inputs with respect to the layer outputs.
    '''
    def __init__(self) -> None:
        self._input_cache = None
        pass

    @abstractmethod
    def forward(self, inputs: np.array) -> np.array:
        self._input_cache = inputs
        # TODO: add docstring? 
        pass
    
    @abstractmethod
    def backward(self, gradients_top):
        """Backward pass of the layer.

        Args:
            gradients_top (n_model_outputs x n_layer_outputs x n_samples): Gradients of the top layer outputs (loss) with respect to the layer outputs.
        
        Inputs could explicitly be passed to the backward method, but it is not required, 
        if the forward method has been called before and they are stored in a cache.
        """
        pass


class GradLayer(ComputationalLayer):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = initialize_weights(input_size, output_size)
        self.biases = initialize_weights(1, output_size)
        self.dweights = np.zeros(self.weights.shape)
        self.dbiases = np.zeros(self.biases.shape)

    def update(self, optimizer: Optimizer) -> None:
        self.weights += optimizer.step(self.dweights)
        self.biases += optimizer.step(self.dbiases)

    # def zero_grad(self) -> None:
    #     self.dweights = np.zeros(self.weights.shape)
    #     self.dbiases = np.zeros(self.biases.shape)


class Linear(GradLayer):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__(input_size, output_size)

    # Could change the order of multiplication but this left to right convention is used in the Machine Perception course at ETH
    def forward(self, inputs: np.array) -> np.array:
        super().forward(inputs)
        return inputs @ self.weights + self.biases

    def backward(self, gradients_top) -> np.array:
        """_summary_

        Args:
            gradients_top (_type_): _description_

        Returns:
            np.array: _description_

        It's important to distingush 4 types of gradients:
        - d_loss_d_layer_outputs            (gradients_top):    Gradients of the loss with respect to the layer outputs.
        - d_loss_d_layer_inputs             (gradients_prop):   Gradients of the loss with respect to the layer inputs. This is what is returned and propagated to the previous layer.
        - d_loss_d_layer_weights            (self.dweights):    Gradients of the loss with respect to the layer weights. What is needed to update the weights.
        - d_loss_d_layer_biases             (self.dbiases):     Gradients of the loss with respect to the layer biases. What is needed to update the biases.
        """


        self.dweights  = []
        self.dbiases = []
        return []
    