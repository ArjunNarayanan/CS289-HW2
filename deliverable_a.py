import neural_network as nn
import finite_difference as fd
import numpy as np
import pickle


def loss_for_finite_difference(Xtrain, Ytrain, weight_matrices, biases, activations):
    nn_vals, nn_grads = nn.forward_pass(Xtrain, weight_matrices, biases, activations)
    output = np.ravel(nn_vals[-1])
    loss, dL_dg = nn.logistic_loss(output, Ytrain)
    return loss


def my_nn_finite_difference_checker(Xbatch, Ybatch, weight_matrices, biases, activations):
    w0 = weight_matrices[0]
    w1 = weight_matrices[1]

    dLdw0 = np.zeros((2, 4, 2))

    for row in range(4):
        for col in range(2):
            dLdw0[:, row, col] = fd.finite_difference(
                lambda w: loss_for_finite_difference(Xbatch, Ybatch, [w, w1], biases, activations), w0,
                (row, col))

    dLdw1 = np.zeros((2, 2, 1))

    for row in range(2):
        for col in range(1):
            dLdw1[:, row, col] = fd.finite_difference(
                lambda w: loss_for_finite_difference(Xbatch, Ybatch, [w0, w], biases, activations), w1,
                (row, col))

    b0 = biases[0]
    b1 = biases[1]
    dLdb0 = np.zeros((2, 1, 2))

    for col in range(2):
        dLdb0[:, 0, col] = fd.finite_difference(
            lambda b: loss_for_finite_difference(Xbatch, Ybatch, weight_matrices, [b, b1], activations), b0, (0, col))

    dLdb1 = np.zeros((2, 1, 1))

    dLdb1[:, 0, 0] = fd.finite_difference(
        lambda b: loss_for_finite_difference(Xbatch, Ybatch, weight_matrices, [b0, b], activations), b1, (0, 0))

    grad_Ws = [dLdw0, dLdw1]
    grad_bs = [dLdb0, dLdb1]

    return grad_Ws, grad_bs


with open("test_batch_weights_biases.pkl", "rb") as file:
    (Xbatch, Ybatch, weight_matrices, biases) = pickle.load(file)

activations = [nn.relu, nn.sigmoid_activation]

grad_Ws, grad_bs = my_nn_finite_difference_checker(Xbatch, Ybatch, weight_matrices, biases, activations)

with np.printoptions(precision=2):
    print(grad_Ws[0])
    print()
    print(grad_Ws[1])
    print()
    print(grad_bs[0])
    print()
    print(grad_bs[1])
