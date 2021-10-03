import numpy as np

import finite_difference
import forward_pass


def display_image(image):
    """
    Displays an image from the mnist dataset

    Make sure you have the matplotlib library installed

    If using Jupyter, you may need to add %matplotlib inline to the top
    of your notebook
    """
    import matplotlib.pyplot as plt
    plt.imshow(image, cmap="gray")


def get_mnist_threes_nines():
    """
    Creates MNIST train / test datasets
    """
    import mnist
    Y0 = 3
    Y1 = 9

    y_train = mnist.train_labels()
    y_test = mnist.test_labels()
    X_train = (mnist.train_images() / 255.0)
    X_test = (mnist.test_images() / 255.0)
    train_idxs = np.logical_or(y_train == Y0, y_train == Y1)
    test_idxs = np.logical_or(y_test == Y0, y_test == Y1)
    y_train = y_train[train_idxs].astype('int')
    y_test = y_test[test_idxs].astype('int')
    X_train = X_train[train_idxs]
    X_test = X_test[test_idxs]
    y_train = (y_train == Y1).astype('int')
    y_test = (y_test == Y1).astype('int')
    return (X_train, y_train), (X_test, y_test)


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


def logistic_loss(g, y):
    assert g.shape == y.shape
    loss = -(y * np.log(g) + (1 - y) * np.log(1 - g))
    grad = (g - y) / (g * (1 - g))
    return loss, grad


def make_weight_matrix(rows, cols, loc=0.0, scale=0.01):
    return np.random.normal(loc, scale, (rows, cols))


def create_weight_matrices(layer_dims, seed=False):
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


def layer_forward(X, W, b, activation_fn):
    S = X @ W + b
    out, cache = activation_fn(S)
    return out, cache


def forward_pass(X_batch, weight_matrices, biases, activations):
    layer_vals = [X_batch]
    layer_grads = []

    for idx, func in enumerate(activations):
        out, cache = layer_forward(layer_vals[-1], weight_matrices[idx], biases[idx], func)
        layer_vals.append(out)
        layer_grads.append(cache)

    return layer_vals, layer_grads


def compute_layer_delta(layer_grad, weight_matrix, next_delta):
    assert weight_matrix.shape[1] == next_delta.shape[1]
    assert layer_grad.shape[0] == next_delta.shape[0]
    assert layer_grad.shape[1] == weight_matrix.shape[0]

    delta = layer_grad * np.einsum('jk,nk->nj', weight_matrix, next_delta)
    return delta


def compute_deltas(layer_gradients, weight_matrices):
    assert len(layer_gradients) == len(weight_matrices)

    dL = layer_gradients[-1]
    deltas = [dL]

    for idx in range(len(weight_matrices) - 1, 0, -1):
        w = weight_matrices[idx]
        lg = layer_gradients[idx - 1]
        nd = deltas[0]

        d = compute_layer_delta(lg, w, nd)
        deltas.insert(0, d)

    return deltas


def compute_layer_loss_derivative_wrt_weights(dLdg, delta, nn_val):
    assert len(dLdg) == delta.shape[0] == nn_val.shape[0]

    return np.einsum('n,nj,ni->nij', dLdg, delta, nn_val)


def compute_loss_derivative_wrt_weights(dLdg, deltas, nn_vals):
    dLdw = []
    assert len(deltas) == len(nn_vals) - 1

    for (idx, d) in enumerate(deltas):
        x = nn_vals[idx]
        dLdw.append(compute_layer_loss_derivative_wrt_weights(dLdg, d, x))

    return dLdw


def compute_loss_derivative_wrt_parameters(dLdg, deltas, nn_vals):
    dLdw = []
    dLdb = []
    assert len(deltas) == len(nn_vals) - 1

    for (idx, d) in enumerate(deltas):
        x = nn_vals[idx]

        assert len(dLdg) == d.shape[0] == x.shape[0]

        layer_dLdb = np.einsum('n,nj->nj', dLdg, d)
        layer_dLdw = np.einsum('nj,ni->nij', layer_dLdb, x)

        dLdb.append(layer_dLdb)
        dLdw.append(layer_dLdw)

    return dLdw, dLdb



def loss_for_finite_difference(Xtrain, Ytrain, weight_matrices, biases, activations):
    nn_vals, nn_grads = forward_pass(Xtrain, weight_matrices, biases, activations)
    output = np.ravel(nn_vals[-1])
    loss, dL_dg = logistic_loss(output, Ytrain)
    return loss


# train, test = get_mnist_threes_nines()
# Xtrain, Ytrain = train
# Xtest, Ytest = test

Xtrain = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
Ytrain = np.array([0, 1])

num_train_samples, numpixels, _ = Xtrain.shape
input_layer_dim = numpixels * numpixels

# flatten the input data into a matrix
Xtrain = Xtrain.reshape(-1, input_layer_dim)

layer_dims = [input_layer_dim, 2, 1]
activations = [relu, sigmoid_activation]

weight_matrices = create_weight_matrices(layer_dims)
biases = create_bias_vectors(layer_dims)

nn_vals, nn_grads = forward_pass(Xtrain, weight_matrices, biases, activations)
output = np.ravel(nn_vals[-1])

loss, dL_dg = logistic_loss(output, Ytrain)
deltas = compute_deltas(nn_grads, weight_matrices)

dL_dw, dL_db = compute_loss_derivative_wrt_parameters(dL_dg, deltas, nn_vals)


