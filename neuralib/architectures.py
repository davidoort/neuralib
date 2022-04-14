import warnings
from matplotlib import pyplot as plt
import numpy as np
from neuralib.layers import ComputationalLayer, Loss
from abc import ABC,abstractmethod
from neuralib.layers.activations import Sigmoid
from neuralib.layers.layers import GradLayer, Identity, Linear
from neuralib.layers.losses import MSE
from neuralib.metrics import ScalarMetric
from neuralib.optimizers import Optimizer, VGD
import typing
from typing import Union, List  

class Architecture(ABC):
    def __init__(self) -> None:
        self.layers = []

    @abstractmethod
    def train(self, X: np.array, y: np.array, batch_size: int, epochs: int, optimizer: Optimizer = VGD(lr=0.1)) -> None:
        """Train the architecture.

        Args:
            X (n_samples x n_features): Training data.
            y (n_samples x label_dim): Training labels.
            batch_size (int): Number of samples per batch.
            epochs (int): Number of epochs.
            optimizer (Optimizer, optional): Optimizer used to update architecture parameters. Defaults to VGD(lr=0.1).
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
    def _forward(self, input: np.array, targets: np.array = None):
        """Abstract private method to compute the forward pass of the architecture.

        Args:
            input (n_samples x n_inputs_architecture): Inputs to the architecture.
            targets (np.array, optional): Targets, which if passed also allow the loss to be computed. Defaults to None.

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
    
    @abstractmethod
    def get_params(self) -> List[typing.Dict[str, np.array]]:
        """
        Return a list of the parameters of each layer in the model.
        """
        pass

class SequentialModel(Architecture):
    '''
    General model architecture, can be customized by adding sequential layers
    '''
    training_loss: List[float]

    def __init__(self, layers: List[ComputationalLayer] = [], metrics: List[ScalarMetric] = None, random_seed: int = None) -> None:
        super().__init__()
        if random_seed is not None:
            np.random.seed(random_seed)
        self.training_loss = []
        self.test_loss = []
        self.layers = layers
        self.metrics = metrics

        if len(self.layers) != 0:
            self.validate()

        if len(self.layers) != 0:
            self.validate()

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

  
    def train(self, X_train: np.array, y_train: np.array, batch_size: int, epochs: int, optimizer: Optimizer = VGD(lr=0.1), X_test: np.array = None, y_test: np.array = None) -> None:
        """
        Standard training script that iterates over the training data in batches, 
        performs a forward and backward pass on all layers and updates the parameters of each layer.

        Args:
            X (n_samples x n_features): Training data.
            y (n_samples x label_dim): Training labels.
            batch_size (int): Number of samples per batch.
            epochs (int): Number of epochs.
            optimizer (Optimizer, optional): Optimizer used to update architecture parameters. Defaults to VGD(lr=0.1).
            X_test (n_test_samples x n_features, optional): Test data which can be used to evaluate the model during training. Defaults to None.
            y_test (n_test_samples x label_dim, optional): Test labels which can be used to calculate the test loss progress during training. Defaults to None.
        """
        super().train(X_train, y_train, batch_size, epochs, optimizer)
        for i in range(epochs):
            # Shuffle data. Using this method instead of sklearn.utils.shuffle to avoid depenency
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            epoch_losses = []

            # Create mini-batches
            n_batches = int(X_train.shape[0] / batch_size)
            X_mb = np.array_split(X_train, n_batches)
            y_mb = np.array_split(y_train, n_batches)

            for Xi, yi in zip(X_mb, y_mb):
                # Do a forward pass using the current batch, labels are also passed and stored in a cache for the backward pass. 
                # This way the forward pass includes the loss calculation.

                prediction, loss = self._forward(Xi, yi)
                epoch_losses.append(loss)

                # Do a backward pass using the current batch
                self._backward()

                # Update the weights and biases of each layer
                for layer in [l for l in self.layers if isinstance(l, GradLayer)]:
                    layer.update(optimizer)

            # Logging
            self.training_loss.append(np.mean(epoch_losses))
            if self.metrics is not None and len(self.metrics) > 0:
                for metric in self.metrics:
                    if i % metric.every_n_epochs == 0:
                        metric.log_from_predictions(prediction, yi, i, dataset='train')

            if X_test is not None and y_test is not None:
                # Calculate the test loss
                test_prediction, test_loss = self._forward(X_test, y_test)
                self.test_loss.append(test_loss)

                if self.metrics is not None and len(self.metrics) > 0:
                    for metric in self.metrics:
                        if i % metric.every_n_epochs == 0:
                            metric.log_from_predictions(test_prediction, y_test, i, dataset='test')

    
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

    def get_params(self) -> List[typing.Dict[str, np.array]]:
        """
        Return a list of the parameters of each layer in the sequential model.
        """
        super().get_params()
        return [layer.get_params() for layer in self.layers if isinstance(layer, GradLayer)]

    # TODO: remove duplicate code between these two plotting functions
    def plot_training_progress(self) -> None:
        # Plot training error progression over time
        training_errors = np.asarray(self.training_loss)
        plt.plot(training_errors)
        plt.xlabel('Epochs')
        plt.ylabel('Training Error')
        plt.show()

    def plot_test_progress(self) -> None:
        if self.test_loss is None or len(self.test_loss) == 0:
            # Print a warning and return
            warnings.warn("No test loss data available. Please provide this data to the train method.", UserWarning)
            return
            
        # Plot test error progression over time
        test_errors = np.asarray(self.test_loss)
        plt.plot(test_errors)
        plt.xlabel('Epochs')
        plt.ylabel('Test Error')
        plt.show()

class MLP(SequentialModel):
    """
    Multi-layer perceptron model.
    """
    def __init__(self, input_size: int, output_size: int, hidden_size: List[int] = None, activations: List[ComputationalLayer] = [Sigmoid()], loss: Loss = MSE(), metrics: List[ScalarMetric] = None, random_seed: int = None) -> None:
        """
        Create an array of layers based on high-level inputs and pass it down to the base class Model.

        Args:
            input_size (int): The dimension of the MLP input.
            output_size (int): The dimension of the MLP output.
            hidden_size (n_hidden, optional): The size of each hidden layer. Defaults to None (no hidden layer).
            activation (ComputationalLayer, optional): The type of activation function used at the end of each linear layer. The length should be n_hidden+2 or 1 if the same activation is used everywhere. Defaults to [Sigmoid()].
            loss (Loss, optional): The loss function used at the end of the model. Defaults to MSE().
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Wrap in a list if hidden_size is not a list
        if not isinstance(hidden_size, list):
            hidden_size = [hidden_size]
        #...same for activations
        if not isinstance(activations, list):
            activations = [activations]

        n_hidden = len(hidden_size) if hidden_size is not None else 0
        assert(len(activations) == n_hidden+1 or len(activations) == 1)

        in_layer = Linear(input_size, hidden_size[0])
        out_layer = Linear(hidden_size[-1], output_size)
        layers = [in_layer, activations[0]]

        # If n_hidden = 1 this will be skipped because the hidden layer is implicitly created between the input and output layers
        for i in range(n_hidden-1):
            layers.append(Linear(hidden_size[i], hidden_size[i+1]))
            if len(activations) == 1:
                layers.append(activations[0])
            if not isinstance(activations[i+1], Identity): layers.append(activations[i+1])
        layers.append(out_layer)
        if not isinstance(activations[-1], Identity): layers.append(activations[-1])
        layers.append(loss)

        super().__init__(layers, metrics=metrics, random_seed=random_seed)
        
