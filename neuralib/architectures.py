import numpy as np
from neuralib.layers import ComputationalLayer, Loss
from abc import ABC,abstractmethod
from neuralib.layers.layers import GradLayer
from neuralib.optimizers import Optimizer, SGD
from typing import Union, List  

class Architecture(ABC):
    def __init__(self) -> None:
        self.layers = []
        pass

    @abstractmethod
    def train(self):
        assert(self.validate()), "Model is not valid"


    def predict(self, input, labels = None):
        assert(self.validate()), "Model is not valid"
        if labels is None:
            return self._forward(input)[0]
        return self._forward(input, labels)

    @abstractmethod
    def _forward(self, input):
        pass

    @abstractmethod
    def _backward(self, inputs, gradients):
        pass
    
    # TODO: do some more thorough checks, such as checking that the dimensions of adjacent layers match
    def validate(self) -> bool:
        '''
        Validate the architecture.
        '''
        print("Validating architecture...")
        print(self.layers)
        loss_layers = [layer for layer in self.layers[:-1] if isinstance(layer, Loss)]
        # Check that there is only one loss layer in self.layers and that it is at the end of the self.layers list
        if isinstance(self.layers[-1], Loss) and not loss_layers:
            return True
        return False


class Model(Architecture):
    '''
    General model architecture, can be customized by adding sequential layers
    '''
    training_loss: List[float]
    layers: List[ComputationalLayer]

    def __init__(self, layers: List[ComputationalLayer] = []) -> None:
        super().__init__()
        self.training_loss = []
        self.layers = layers

    def add(self, layer: ComputationalLayer) -> None:
        self.layers.append(layer)
  
    def train(self, X, y, batch_size: int, epochs: int, optimizer: Optimizer = SGD(lr=0.1)):
        super().train()
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

        
    
    def _forward(self, inputs, targets=None) -> Union[np.array, np.array]:
        output = inputs
        loss = None
        # Run a recursive loop through the layers of the model
        for layer in self.layers:
            # Check if layer is a subclass of Loss(). 
            # Assuming that there is only one loss layer and it's the last layer in the model
            if isinstance(layer, Loss):
                if targets is not None:
                    loss = layer.forward(output, targets)
                continue
            output = layer.forward(output)
         
        return output.squeeze(), loss

    def _backward(self) -> None:
        '''
        This should update the dweights and dbiases of each GradLayer in the model
        '''
        for layer in reversed(self.layers):
            if isinstance(layer, Loss):
                gradient = layer.backward()
                continue
            gradient = layer.backward(gradient)

    # def plot_progress(self):
    #     # Plot training error progression over time
    #     training_errors = np.asarray(self.training_loss)
    #     plt.plot(training_errors[:, 0], training_errors[:, 1])
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Training Error')

# TODO: implement parametrizable MLP model using Model class
# class MLP(Model):
