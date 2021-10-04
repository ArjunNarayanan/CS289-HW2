import neural_network as nn
import numpy as np
import matplotlib.pyplot as plt


def plot_train_test_loss(train_loss, test_loss, filename=""):
    fig, ax = plt.subplots()
    ax.plot(train_loss, label="train loss")
    ax.plot(test_loss, label="test loss")
    ax.grid()
    ax.set_xlabel("Mini-batch iterations")
    ax.set_ylabel("Loss")
    ax.legend()
    if len(filename) > 0:
        fig.savefig(filename)


def plot_train_test_accuracy(train_accuracy, test_accuracy, filename=""):
    fig, ax = plt.subplots()
    ax.plot(train_accuracy, label="train accuracy")
    ax.plot(test_accuracy, label="test accuracy")
    ax.grid()
    ax.set_xlabel("Mini-batch iterations")
    ax.set_ylabel("Accuracy %")
    ax.legend()
    if len(filename) > 0:
        fig.savefig(filename)


learning_rate = 10.0
batchsize = 100
numepochs = 5
train, test = nn.get_mnist_threes_nines()
Xtrain, Ytrain = train
Xtest, Ytest = test

num_train_samples, numpixels, _ = Xtrain.shape
input_layer_dim = numpixels * numpixels

Xtrain = Xtrain.reshape(-1, input_layer_dim)
Xtest = Xtest.reshape(-1, input_layer_dim)

layer_dims = [input_layer_dim, 200, 1]
activations = [nn.relu, nn.sigmoid_activation]

weight_matrices = nn.create_weight_matrices(layer_dims)
biases = nn.create_bias_vectors(layer_dims)

batch_train_losses, batch_test_losses, batch_train_accuracy, batch_test_accuracy = nn.run_epochs(Xtrain, Ytrain, Xtest,
                                                                                                 Ytest, weight_matrices,
                                                                                                 biases, activations,
                                                                                                 learning_rate,
                                                                                                 batchsize, numepochs)

combined_train_loss = np.concatenate(batch_train_losses)
combined_test_loss = np.concatenate(batch_test_losses)

combined_train_accuracy = np.concatenate(batch_train_accuracy)
combined_test_accuracy = np.concatenate(batch_test_accuracy)

# plot_train_test_loss(combined_train_loss, combined_test_loss, "train-test-loss.png")
# plot_train_test_accuracy(combined_train_accuracy, combined_test_accuracy, "train-test-accuracy.png")

# testvals, _ = nn.forward_pass(Xtest, weight_matrices, biases, activations)
# testop = np.ravel(testvals[-1])
# predictions = np.rint(testop)
# matches = predictions == Ytest
#
# failed_index = np.where(~matches)[0]


# def save_image(image, image_index):
#     foldername = "failed-images\\"
#     filename = foldername + "image-" + str(image_index) + ".png"
#     plt.imshow(image, cmap="gray")
#     plt.savefig(filename)
#
#
# Xtestimages, _ = test
# for idx in failed_index:
#     save_image(Xtestimages[idx], idx)
