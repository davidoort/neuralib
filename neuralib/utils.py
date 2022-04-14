import numpy as np
from numpy.random import randn

def xor_data(num_examples, noise=None):
    X = randn(num_examples, 2)

    if noise is None:
        X_ = X
    else:
        X_ = X + noise * randn(num_examples, 2)

    y = np.atleast_2d(np.logical_xor(X_[:, 0] > 0, X_[:, 1] > 0).astype(int)).T
    return X, y
