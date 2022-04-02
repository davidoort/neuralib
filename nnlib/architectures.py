from xmlrpc.client import Boolean
import numpy as np
from nnlib.layers import ComputationalLayer, Loss
from abc import ABC,abstractmethod
class Architecture(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def train(self):
        pass

    def predict(self, input):
        '''
        When predicting, we do not want to do a loss pass.
        '''
        return self._forward(input, loss_pass=False)

    @abstractmethod
    def _forward(self, input, loss_pass: Boolean = True):
        pass

    @abstractmethod
    def _backward(self, inputs, gradients):
        pass

class Model(Architecture):
    '''
    General model architecture, can be customized by adding sequential layers
    '''
    def __init__(self):
        self.training_loss = []
        self.layers = []
        self.output = None

    def add(self, layer: ComputationalLayer) -> None:
        self.layers.append(layer)
  
    def train(self):
        # TODO: Implement training
        self.training_loss.append()
    
    def _forward(self, inputs, loss_pass: Boolean = True):
        output = inputs
        # Run a recursive loop through the layers of the model
        for layer in self.layers:
            # Check if layer is a subclass of Loss() and omit the layer if loss_pass is False
            if isinstance(layer, Loss) and loss_pass == False:
                continue
            output = layer.forward(output)
        self.output = output.squeeze()
        return self.output

    def _backward(self, targets) -> None:
        '''
        This should update the dweights and dbiases of each GradLayer in the model
        '''
        # TODO: Implement backpropagation
        for layer in reversed(self.layers):
            gradient = layer.backward(targets, gradient)

    # def plot_progress(self):
    #     # Plot training error progression over time
    #     training_errors = np.asarray(self.training_loss)
    #     plt.plot(training_errors[:, 0], training_errors[:, 1])
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Training Error')

# class MLP(Model):
