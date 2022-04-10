from matplotlib import pyplot as plt
import numpy as np
from neuralib.layers import ComputationalLayer, Loss
from abc import ABC,abstractmethod
from neuralib.layers.layers import GradLayer
from neuralib.optimizers import Optimizer, SGD
from typing import Union, List  

class Architecture(ABC):
    def __init__(self) -> None:
        self.layers = []

    @abstractmethod
    def train(self, X: np.array, y: np.array, batch_size: int, epochs: int, optimizer: Optimizer = SGD(lr=0.1)) -> None:
        """Train the architecture.

        Args:
            X (n_samples x n_features): Training data.
            y (n_samples x label_dim): Training labels.
            batch_size (int): Number of samples per batch.
            epochs (int): Number of epochs.
            optimizer (Optimizer, optional): Optimizer used to update architecture parameters. Defaults to SGD(lr=0.1).
        """
        assert(self.validate()), "Model is not valid"


    def predict(self, input: np.array, labels: np.array = None):
        """Predict the output of the network.

        Args:
            input (n_samples x n_features): Input data.
            labels (n_samples x label_dim, optional): Labels. Defaults to None.

        Returns:
            n_samples x label_dim: Predicted labels.
        """

        assert(self.validate()), "Model is not valid"
        if labels is None:
            return self._forward(input)[0]
        return self._forward(input, labels)

    @abstractmethod
    def _forward(self, input: np.array):
        """Abstract private method to compute the forward pass of the architecture.

        Args:
            input (n_samples x n_inputs_architecture): Inputs to the architecture.
        """
        pass

    @abstractmethod
    def _backward(self) -> None:
        """
        Abstract private method to do a full backward pass of the architecture. Used in the public train method.
        """
        pass
    
    def validate(self) -> bool:
        """Validate the general network architecture.

        Returns:
            bool: Whether the architecture is generally valid. More checks might be done in the subclasses.
        """

        # Check that the architecture has at least one layer
        if len(self.layers) == 0:
            return False

        # Check that the architecture has only one loss layer and that it is the last layer
        loss_layers = [layer for layer in self.layers[:-1] if isinstance(layer, Loss)]
        # Check that there is only one loss layer in self.layers and that it is at the end of the self.layers list
        if not isinstance(self.layers[-1], Loss) or loss_layers:
            return False
        
        return True
    



class Model(Architecture):
    '''
    General model architecture, can be customized by adding sequential layers
    '''
    training_loss: List[float]

    def __init__(self, layers: List[ComputationalLayer] = []) -> None:
        super().__init__()
        self.training_loss = []
        self.layers = layers

    def add(self, layer: ComputationalLayer) -> None:
        # Add a layer to the end of the model
        self.layers.append(layer)

    def add_front(self, layer: ComputationalLayer) -> None:
        # Add a layer to the front of the model
        self.layers.insert(0, layer)
    
    def pop(self, index: int = -1) -> ComputationalLayer:
        # Remove the last layer by default
        return self.layers.pop(index)

    def validate(self) -> bool:
        if not super().validate():
            return False
        
        # Check that the dimensions between layers match
        # Filter out the loss layer and the layers which have input_size of None (those don't change the dimensions of the inputs)
        layers = [l for l in self.layers if l.input_size is not None]
        for i in range(len(layers)-1):
            if layers[i].output_size:
                if layers[i].output_size != layers[i+1].input_size:
                    return False
            
        
        return True

  
    def train(self, X: np.array, y: np.array, batch_size: int, epochs: int, optimizer: Optimizer = SGD(lr=0.1)) -> None:
        """
        Standard training script that iterates over the training data in batches, 
        performs a forward and backward pass on all layers and updates the parameters of each layer.

        Args:
            X (n_samples x n_features): Training data.
            y (n_samples x label_dim): Training labels.
            batch_size (int): Number of samples per batch.
            epochs (int): Number of epochs.
            optimizer (Optimizer, optional): Optimizer used to update architecture parameters. Defaults to SGD(lr=0.1).
        """
        super().train(X, y, batch_size, epochs, optimizer)
        for _ in range(epochs):
            # TODO: Implement batch randomization
            for i in range(0, X.shape[0], batch_size):
                # Do a forward pass using the current batch, labels are also passed and stored in a cache for the backward pass. 
                # This way the forward pass includes the loss calculation.
                _, loss = self._forward(X[i:i+batch_size], y[i:i+batch_size])

                self.training_loss.append(loss)

                # Do a backward pass using the current batch
                self._backward()

                # Update the weights and biases of each layer
                for layer in [l for l in self.layers if isinstance(l, GradLayer)]:
                    layer.update(optimizer)

        
    
    def _forward(self, inputs: np.array, targets: np.array = None) -> Union[np.array, np.array]:
        """Private method to compute the forward pass of the model.

        Args:
            inputs (np.array): Inputs to the model.
            targets (np.array, optional): Targets, which if passed also allow the loss to be computed. Defaults to None.

        Returns:
            Union[np.array, np.array]: Output of the forward pass (predictions) and the loss (if targets are passed).
        """
        output = inputs
        loss = None
        # Run a recursive loop through the layers of the model
        for layer in self.layers:
            # Check if layer is a subclass of Loss(). 
            # Assuming that there is only one loss layer and it's the last layer in the model
            if isinstance(layer, Loss):
                if targets is not None:
                    _, loss = layer.forward(output, targets)
                continue
            output = layer.forward(output)
         
        return output, loss

    def _backward(self) -> None:
        """
        This should update the dweights and dbiases of each GradLayer in the model
        """
        super()._backward()
        for layer in reversed(self.layers):
            if isinstance(layer, Loss):
                gradient = layer.backward()
                continue
            gradient = layer.backward(gradient)

    def plot_progress(self):
        # Plot training error progression over time
        training_errors = np.asarray(self.training_loss)
        plt.plot(training_errors)
        plt.xlabel('Epochs')
        plt.ylabel('Training Error')

# TODO: implement parametrizable MLP model using Model class
# class MLP(Model):
