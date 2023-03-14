import numpy as np


def relu(z: np.ndarray):
    return np.maximum(0, z)


def relu_prime(z):
    z[z < 0] = 0
    return z


def sigmoid(z: np.ndarray):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(Z: np.ndarray):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=0, keepdims=True)


def softmax_dLdZ(output, target):
    return output - target
