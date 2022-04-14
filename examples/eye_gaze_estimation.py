import h5py
import os
from neuralib.architectures import MLP
from neuralib.layers.activations import ReLU
from neuralib.layers.layers import Identity
from neuralib.optimizers import VGD

from helpers.download_data import download
from helpers.eye_gaze_helpers import predict_and_visualize, predict_and_calculate_mean_error


if __name__ == '__main__':

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    file_path = os.path.join(dname, '../data/eye_gaze/eye_data.h5')

    # Download data
    download(url='https://github.com/jtj21/ComputationalInteraction18/blob/master/Otmar/data/eye_data.h5?raw=true',
             file_path=file_path)
    
    # Load in our data
    with h5py.File(file_path, 'r') as h5f:
        train_x = h5f['train/x_small'][:]
        train_y = h5f['train/y'][:]

        validation_x = h5f['validation/x_small'][:]
        validation_y = h5f['validation/y'][:]

        test_x = h5f['test/x_small'][:]
        test_y = h5f['test/y'][:]

    # A neural network should be trained until the training and test
    # errors plateau, that is, they do not improve any more.
    epochs = 201

    # Having more neurons in a network allows for more complex 
    # mappings to be learned between input data and expected outputs.
    # However, defining the function to be too complex can lead to 
    # overfitting, that is, any function can be learned to memorize
    # training data.
    n_hidden_units = 64

    # Lower batch sizes can cause noisy training error progression,
    # but sometimes lead to better generalization (less overfitting
    # to training data)
    batch_size = 16

    # A higher learning rate makes training faster, but can cause
    # overfitting
    learning_rate = 0.0005

    # TODO: implement L2 regularization
    # Increase to reduce over-fitting effects
    # l2_regularization_coefficient = 0.0001

    train_x = train_x.reshape(train_x.shape[0], -1)

    n_features = train_x.shape[1] # flattened grayscale image of eye
    n_outputs = train_y.shape[1] # Pitch and yaw in radians

    mlp = MLP(output_size=n_outputs,
              input_size=n_features,
              hidden_size=n_hidden_units,
              activations=[ReLU(), Identity()])
              
    mlp.train(train_x, train_y, epochs=epochs, batch_size=batch_size, optimizer=VGD(lr=learning_rate))

    print('Error: ', predict_and_calculate_mean_error(mlp, test_x, test_y))
    predict_and_visualize(mlp, test_x, test_y)