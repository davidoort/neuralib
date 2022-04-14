import typing
import numpy as np
from neuralib.utils import initialize_weights_uniform
from abc import ABC,abstractmethod
from neuralib.optimizers import Optimizer

class ComputationalLayer(ABC):
    '''
    This layer does not have parameters, so it does not need to be updated.
    The forward method is used to compute the output of the layer.
    The backward method is used to compute the gradient of the layer inputs with respect to the layer outputs.
    '''
    _input_cache: np.ndarray

    def __init__(self, input_size: int = None, output_size: int = None) -> None:
        """Layers that change the input size or output size should pass the input and output size to the constructor. 
        This is used during validation of the architecture.

        Args:
            input_size (int, optional): Input size of the layer. Defaults to None.
            output_size (int, optional): Output size of the layer. Defaults to None.
        """
        self._input_cache = None
        self.input_size = input_size
        self.output_size = output_size


    @abstractmethod
    def forward(self, inputs: np.array) -> np.array:
        """The forward pass (computation) of the layer. Inputs are stored in a cache for the backward pass.

        Args:
            inputs (n_samples x n_inputs): Inputs to the layer. 

        Returns:
            np.array: (n_samples x n_outputs) Output of the layer.
        """
        self._input_cache = inputs # n_samples x n_inputs
    
    @abstractmethod
    def backward(self, gradients_top: np.array) -> np.array:
        """Backward pass of the layer.

        Args:
            gradients_top (n_model_outputs x n_layer_outputs): Gradients of the top layer outputs (loss) with respect to the layer outputs.
        
        Inputs to the layer required for the backward pass are stored in the _input_cache.
        """
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        if self.input_size==other.input_size and self.output_size==other.output_size and isinstance(other, ComputationalLayer):
            return True
        else:
            return False


class GradLayer(ComputationalLayer):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__(input_size, output_size)
        self.weights = initialize_weights_uniform(input_size, output_size) # n_inputs x n_outputs
        self.biases = initialize_weights_uniform(1, output_size) # 1 x n_outputs
        self.d_weights = np.zeros(self.weights.shape) # n_inputs x n_outputs = weights.shape
        self.d_biases = np.zeros(self.biases.shape) # 1 x n_outputs = biases.shape

    def update(self, optimizer: Optimizer) -> None:
        """
        Update the weights and biases of the layer using the gradients computed 
        during the backward pass and an optimizer of choice.

        Args:
            optimizer (Optimizer): The optimizer step method is called to calculate 
            the increment of the weights and biases based on their gradient.
        """
        self.weights += optimizer.step(self.d_weights)
        self.biases += optimizer.step(self.d_biases)

    def get_params(self) -> typing.Dict[str, np.array]:
        """
        Method that returns all the parameters of the layer (weights and biases) in a dictionary
        as well as the number of parameters in the layer.
        """ 
        return {'weights': self.weights, 'biases': self.biases, 'n_params': self.weights.size + self.biases.size}

    def __eq__(self, other) -> bool:
        if super().__eq__(other):
            if isinstance(other, GradLayer):
                return True
            else:
                return False
        return False
            
class Identity(ComputationalLayer):
    '''
    Auxiliary class that can be used as an "identity" placeholder for layers.
    '''
    def __init__(self, input_size: int = None, output_size: int = None) -> None:
        super().__init__(input_size, output_size)

    def forward(self, inputs: np.array) -> np.array:
        # This layer does not modify the inputs, so we can just return them.
        return inputs

    def backward(self, gradients_top: np.array) -> np.array:
        # This layer does not modify the inputs, so the gradient of the output with respect to the inputs is just np.ones(inputs.shape)
        # so we can just return the gradient of the top layers.
        return gradients_top

    def __eq__(self, other) -> bool:
        if super().__eq__(other):
            if isinstance(other, Identity):
                return True
            else:
                return False
        return False
class Linear(GradLayer):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__(input_size, output_size)

    # Could change the order of multiplication but this left to right convention is used in the Machine Perception course at ETH
    def forward(self, inputs: np.array) -> np.array:
        """The forward pass of the linear layer performs a matrix multiplication between the inputs and the weights and adds the biases.

        Args:
            inputs (n_samples x n_inputs): Inputs to the linear layer.

        Returns:
            (n_samples x n_outputs): Layer outputs.
        """
        super().forward(inputs)
        return inputs @ self.weights + self.biases

    def backward(self, gradients_top) -> np.array:
        """This is the backward pass of the layer. It calculates the gradient of the top of the computational graph (e.g. loss) with respect to the layer inputs, weights and biases.

        Args:
            gradients_top (np.array): n_sample x n_outputs

        Returns:
            np.array: the gradient of the top of the computational graph with respect to the layer inputs, which is passed to the previous layer.

        It's important to distingush 4 types of gradients:
        - d_loss_d_layer_outputs            (gradients_top):    Gradients of the loss with respect to the layer outputs.
        - d_loss_d_layer_inputs             (gradients_prop):   Gradients of the loss with respect to the layer inputs. This is what is returned and propagated to the previous layer.
        - d_loss_d_layer_weights            (self.d_weights):    Gradients of the loss with respect to the layer weights. What is needed to update the weights.
        - d_loss_d_layer_biases             (self.d_biases):     Gradients of the loss with respect to the layer biases. What is needed to update the biases.
        """
        super().backward(gradients_top)

        self.d_weights = self._input_cache.T @ gradients_top  # Shape of d_weights is the same as the shape of weights so n_inputs x n_outputs 
        self.d_biases = np.ones(shape=[1, gradients_top.shape[0]]) @ gradients_top # Shape of biases is 1 x n_outputs
        return gradients_top @ self.weights.T # Shape of gradients_prop is n_samples x n_inputs

    def __eq__(self, other) -> bool:
        if super().__eq__(other):
            if isinstance(other, Linear):
                return True
            else:
                return False
        return False