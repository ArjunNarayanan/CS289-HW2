import neural_network as nn
import numpy as np
import pickle

with open("test_batch_weights_biases.pkl", "rb") as file:
    (Xbatch, Ybatch, weight_matrices, biases) = pickle.load(file)

activations = [nn.relu, nn.sigmoid_activation]
output, _ = nn.forward_pass(Xbatch, weight_matrices, biases, activations)
op = np.ravel(output[-1])

loss, dLdg = nn.logistic_loss(op, Ybatch)
print(loss.mean())