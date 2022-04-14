from abc import ABC, abstractmethod
from matplotlib import pyplot as plt

import numpy as np

class ScalarMetric(ABC):
    '''
    Abstract class for scalar metrics (floats) that can be calculated and logged from a model during training, validation or testing.
    '''
    def __init__(self, every_n_epochs: int = 1) -> None:
        self.metric_history_train = []
        self.metric_history_test = []
        assert(every_n_epochs > 0 and isinstance(every_n_epochs, int))
        self.every_n_epochs = every_n_epochs

    def log(self, epoch: int, value, history: str = 'train') -> None:
        if history == 'train':
            self.metric_history_train.append((epoch, value))
        elif history == 'test':
            self.metric_history_test.append((epoch, value))
        else:
            raise ValueError('history must be either train or test')

    # def calculate_from_model(self, model, X: np.array, y: np.array) -> float:
    #     return self.calculate_from_predictions(model.predict(X), y)
    
    @abstractmethod
    def calculate_from_predictions(self, y_pred: np.array, y: np.array) -> float:
        raise NotImplementedError()

    # def log_from_model(self, model, X: np.array, y: np.array, epoch: int) -> None:
    #     self.log(epoch, self.calculate_from_model(model, X, y))

    def log_from_predictions(self, y_pred: np.array, y: np.array, epoch: int, history: str = 'train') -> None:
        self.log(epoch, self.calculate_from_predictions(y_pred, y), history)

    def visualize(self):
        raise NotImplementedError()

    def plot_progress(self, history: str = 'train') -> None:
        # Plot training error progression over time
        if history == 'train':
            history = self.metric_history_train
        elif history == 'test':
            history = self.metric_history_test
        else:
            raise ValueError('history must be either train or test')

        training_errors = np.asarray(history) 
        epochs = training_errors[:, 0]
        metric = training_errors[:, 1]
        plt.plot(epochs, metric)
        plt.xlabel('Epoch')
        plt.ylabel(self.__class__.__name__)
        plt.show()