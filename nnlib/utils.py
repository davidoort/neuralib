import numpy as np

def initialize_weights(input_size, output_size):
    return np.random.uniform(size=(input_size, output_size))