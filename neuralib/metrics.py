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

    def log(self, epoch: int, value, dataset: str = 'train') -> None:
        if dataset == 'train':
            self.metric_history_train.append((epoch, value))
        elif dataset == 'test':
            self.metric_history_test.append((epoch, value))
        else:
            raise ValueError('dataset must be either train or test')

    # def calculate_from_model(self, model, X: np.array, y: np.array) -> float:
    #     return self.calculate_from_predictions(model.predict(X), y)
    
    @abstractmethod
    def calculate_from_predictions(self, y_pred: np.array, y: np.array) -> float:
        raise NotImplementedError()

    # def log_from_model(self, model, X: np.array, y: np.array, epoch: int) -> None:
    #     self.log(epoch, self.calculate_from_model(model, X, y))

    def log_from_predictions(self, y_pred: np.array, y: np.array, epoch: int, dataset: str = 'train') -> None:
        self.log(epoch, self.calculate_from_predictions(y_pred, y), dataset)

    def visualize(self):
        raise NotImplementedError()

    # TODO: Store history in a dictionary that is indexed by dataset and when dataset is not passed, plot all datasets
    def plot_progress(self, dataset: str = 'train') -> None:
        # Plot training error progression over time
        if dataset == 'train':
            history = self.metric_history_train
        elif dataset == 'test':
            history = self.metric_history_test
        else:
            raise ValueError('dataset must be either train or test or not specified (both)')

        training_errors = np.asarray(history) 
        epochs = training_errors[:, 0]
        metric = training_errors[:, 1]
        plt.plot(epochs, metric)
        plt.xlabel('Epoch')
        plt.ylabel(self.__class__.__name__ + ' on ' + dataset)
        plt.show()