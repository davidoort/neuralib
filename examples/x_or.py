import numpy as np
import matplotlib.pyplot as plt
from neuralib import Model
from neuralib.layers import Linear, Sigmoid, MSE
from neuralib.optimizers import VGD

# from neuralib.utils import xor_data

def plot_simple(model):
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_hat = [np.round(model.predict(x)) for x in X]

    # Colors corresponding to class predictions y_hat.
    colors = ['green' if y_ == 1 else 'blue' for y_ in y_hat] 

    fig = plt.figure()
    fig.set_figwidth(6)
    fig.set_figheight(6)
    plt.scatter(X[:,0],X[:,1],s=200,c=colors)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def plot_grid(model):
    resolution = 20
    min_x, min_y = 0.0, 0.0
    max_x, max_y = 1.0, 1.0
    xv, yv = np.meshgrid(np.linspace(min_x, max_x, resolution), np.linspace(min_y, max_y, resolution))
    X_extended = np.concatenate([xv[..., np.newaxis], yv[..., np.newaxis]], axis=-1)
    X_extended = np.reshape(X_extended, [-1, 2])
    y_hat = [np.round(model.predict(x)) for x in X_extended]

    # Colors corresponding to class predictions y_hat.
    colors = ['green' if y_ == 1 else 'blue' for y_ in y_hat] 

    fig = plt.figure()
    fig.set_figwidth(6)
    fig.set_figheight(6)
    plt.scatter(X_extended[:,0],X_extended[:,1],s=200,c=colors)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


if __name__ == '__main__':
    # Create custom model
    input_dim = 2    # number of features (dimensionality of the data)
    hidden_dim = 4   # number of neurons in the hidden layer
    target_dim = 1    # label dimensionality 

    model = Model([Linear(input_size=input_dim, output_size=hidden_dim), 
                   Sigmoid(), 
                   Linear(input_size=hidden_dim, output_size=target_dim), 
                   MSE()])

    # Training the model
    # Generate training data
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([ [0],   [1],   [1],   [0]])

    model.train(X, y, batch_size=4, epochs=10000, optimizer=VGD(lr=0.1))
    y_pred, loss = model.predict(X, y)

    print("Training loss: ", loss)
    model.plot_progress()

    # Plotting
    plot_simple(model)
    plot_grid(model)