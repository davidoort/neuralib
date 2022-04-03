import numpy as np
import matplotlib.pyplot as plt
from neuralib import Model
from neuralib.layers import Linear, Sigmoid, MSE
from neuralib.optimizers import SGD

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
    model = Model()
    model.add(Linear(input_size=2, output_size=3))
    model.add(Sigmoid())
    model.add(Linear(input_size=3, output_size=1))
    model.add(MSE())

    # Training

    # Fix the random seed
    np.random.seed(0)

    # Generate training data
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([ [0],   [1],   [1],   [0]])

    # model.train(X, y, batch_size=4, num_epochs=10000, optimizer=SGD(lr=0.1))
    y_pred = model.predict(X)

    print("Training loss: ", MSE.mse(y_pred, y)[1])

    # Plotting
    plot_simple(model)
    plot_grid(model)