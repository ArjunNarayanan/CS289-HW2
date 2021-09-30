import numpy as np


def make_weight_matrix(rows, cols, loc=0.0, scale=0.01):
    return np.random.normal(loc, scale, (rows, cols))


def create_weight_matrix(layer_dims, seed=False):
    if seed:
        np.random.seed(42)

    numlayers = len(layer_dims)
    weights = [make_weight_matrix(layer_dims[i], layer_dims[i + 1]) for i in range(numlayers - 1)]

    return weights


def make_bias_vector(cols, loc=0.0, scale=0.01):
    return np.random.normal(loc, scale, (1, cols))


def create_bias_vectors(layer_dims, seed=False):
    if seed:
        np.random.seed(42)

    numlayers = len(layer_dims)
    biases = [make_bias_vector(layer_dims[i + 1]) for i in range(numlayers - 1)]

    return biases
