import neural_network as nn
import numpy as np
import pickle

with open("test_batch_weights_biases.pkl", "rb") as file:
    (Xbatch, Ybatch, weight_matrices, biases) = pickle.load(file)

activations = [nn.relu, nn.sigmoid_activation]
nn_vals, nn_grads = nn.forward_pass(Xbatch, weight_matrices, biases, activations)
output = np.ravel(nn_vals[-1])

loss, dLdg = nn.logistic_loss(output, Ybatch)
deltas = nn.compute_deltas(nn_grads, weight_matrices)

grad_Ws, grad_bs = nn.compute_loss_derivative_wrt_parameters(dLdg, deltas, nn_vals)

with np.printoptions(precision=2):
    print(grad_Ws[0])
    print()
    print(grad_Ws[1])
    print()
    print(grad_bs[0])
    print()
    print(grad_bs[1])
