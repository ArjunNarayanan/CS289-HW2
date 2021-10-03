import neural_network as nn
import numpy as np

s = np.asarray([1., 0., -1.])
result, grad = nn.sigmoid_activation(s)

s = np.asarray([-1000, 1000])
result, grad = nn.sigmoid_activation(s)