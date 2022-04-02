import numpy as np
import matplotlib.pyplot as plt
from nnlib import Model
from nnlib.layers import FullyConnected, Sigmoid, MSE

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
    model.add(FullyConnected(input_size=2, output_size=3))
    model.add(Sigmoid())
    model.add(FullyConnected(input_size=3, output_size=1))
    model.add(MSE())

    # model.compile(SGD(lr=1))

    # model.train(X, y, batch_size=n, num_epochs=180)
    # y_pred = model.predict(X)

    # acc = classification_accuracy(y, y_pred)

    # Plotting
    plot_simple(model)
    plot_grid(model)