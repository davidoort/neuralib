# Neuralib: a tiny library developed to demistify neural networks
![Neuralib](img/neuralib_landscape.png)
This small python module has been developed purely for learning purposes. It's an implementation of simple neural networks using numpy as only dependency.

> *"…everything became much clearer when I started writing code."* - Andrej Karpathy

## Supported features
Neuralib is designed to provide a suite of submodules that can be used separately or combined into custom neural net architectures. These submodules, and their classes are:

`neuralib.optimizers`: A submodule containing optimizers that define parameter update rules based on loss gradients.
- **Vanilla Gradient Descent (VGD)**: This optimizer simply updates parameters by multiplying a fixed learning rate `lr` by the gradient of the loss with respect to these parameters.

`neuralib.layers`: In neuralib, everything that performs a computation on inputs and returns an output is treated as a layer. The simplest form of a layer is a `ComputationalLayer`, which performs a fixed computation and does not contain parameters to be optimized. A `GradLayer` is built on top of a `ComputationalLayer` and additionally contains parameters to be optimized during training. Currently supported layers include:
- `neuralib.layers.losses`
    - **Mean-Squared Error (MSE)**
- `neuralib.layers.activations`
    - **Sigmoid**
- `neuralib.layers.layers`:
    - **Linear** (or Fully-Connected Layer)

`neuralib.architectures`: Neuralib architectures are an abstraction built on top of the above submodule components. They allow the user to quickly specify a model architecture, train and test it.
- **Model**: a generic sequential model that can be customized to any valid sequence of layers from `neuralib.layers`. The forward pass and backpropagation have been implemented.

In addition to these submodules, a test suite in `tests/` contains unit and integration tests to ensure that active development of this library does not cause regressions.

## Using neuralib
To update your conda environment so that neuralib dependencies are met, run:
`conda env update --file environment.yml --prune
`

To create a conda env from the environment.yml file, run:
`conda env create --file environment.yml --name neuralib_env`
### ...to learn about Neural Networks
Head over to the `examples/` folder and run one of the simple scripts. This should be your starting point to reverse engineer the inner workings of Neural Networks.
### ...as a Neural Network module
From the root of the repository, run `pip install .`
Now you should be able to open a Python console in your terminal and do `import neuralib`, which will also work in your scripts.

**If you find a bug or missing feature that you know how to implement, open a Pull Request and I'll be happy to get it merged! ❤️**

## Resources
The following resources have been very helpful in the development of this library:
- Machine Perception Course (ETH Zurich) Notebooks: https://ait.ethz.ch/teaching/courses/2022-SS-Machine-Perception/
- CS231n Course (Stanford) Notes: https://cs231n.github.io/
- Andrej Karpathy's blog: http://karpathy.github.io/neuralnets/ 