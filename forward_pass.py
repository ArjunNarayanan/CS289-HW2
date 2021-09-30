def layer_forward(X, W, b, activation_fn):
    S = X @ W + b
    out, cache = activation_fn(S)
    return out, cache


def forward_pass(X_batch, weight_matrices, biases, activations):
    numdata, numinputs = X_batch.shape

    layer_vals = [X_batch]
    layer_grads = []

    for idx, func in enumerate(activations):
        out, cache = layer_forward(layer_vals[-1], weight_matrices[idx], biases[idx], func)
        layer_vals.append(out)
        layer_grads.append(cache)

    return layer_vals, layer_grads