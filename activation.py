import numpy as np


def positive_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def negative_sigmoid(x):
    exp = np.exp(x)
    return exp / (1 + exp)


def clip_epsilon(x, epsilon):
    lessidx = x < epsilon
    greatidx = x > 1 - epsilon
    x[lessidx] = epsilon
    x[greatidx] = 1 - epsilon


def sigmoid_activation(x, epsilon=1e-15):
    posidx = x > 0
    negidx = ~posidx

    result = np.empty_like(x)
    result[posidx] = positive_sigmoid(x[posidx])
    result[negidx] = negative_sigmoid(x[negidx])
    clip_epsilon(result, epsilon)

    grad = result * (1 - result)

    return result, grad


def relu(x):
    result = np.zeros_like(x)
    posidx = x > 0
    result[posidx] = x[posidx]

    grad = np.zeros_like(x)
    grad[posidx] = 1.0

    return result,grad
