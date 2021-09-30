import numpy as np


def positive_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def negative_sigmoid(x):
    exp = np.exp(x)
    return exp / (1 + exp)


def sigmoid_activation(x, epsilon=1e-15):
    posidx = x > 0
    negidx = ~posidx

    result = np.empty_like(x)
    result[posidx] = positive_sigmoid(x[posidx])
    result[negidx] = negative_sigmoid(x[negidx])

    np.clip(result, epsilon, 1 - epsilon)

    grad = result * (1 - result)

    return result, grad


def relu(x):
    result = np.zeros_like(x)
    posidx = x > 0
    result[posidx] = x[posidx]

    grad = np.zeros_like(x)
    grad[posidx] = 1.0

    return result, grad
