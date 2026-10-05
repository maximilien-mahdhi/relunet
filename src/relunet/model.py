"""Two-layer ReLU network.

The network is f_w(x) = (1/m) * sum_j a_j * relu(b_j * x + c_j).
Its parameters are stored in a flat array `w` of length 3*m laid out as
(a_1, b_1, c_1, a_2, b_2, c_2, ...), which are the weights of the network.
"""

import numpy as np


def relu(x):
    """ReLU activation."""
    return np.maximum(0, x)


def relu_grad(x):
    """Subgradient of ReLU, taking the value 0 at x == 0."""
    return np.where(x > 0, 1, 0)


def init_weights(m, alpha_inner, alpha_outer):
    """Draw centered Gaussian weights with variance `alpha_outer` for the outer
    weights a_j and `alpha_inner` for the inner weights b_j and c_j."""
    w = np.zeros(3 * m)
    for i in range(m):
        w[3 * i] = np.random.randn() * np.sqrt(alpha_outer)
        w[3 * i + 1] = np.random.randn() * np.sqrt(alpha_inner)
        w[3 * i + 2] = np.random.randn() * np.sqrt(alpha_inner)
    return w


def forward(w, x):
    """Evaluate the network on all samples and neurons at once.

    Returns:
        predictions of shape (n,) and pre-activations Z of shape (n, m),
        which are reused by the backward pass.
    """
    m = len(w) // 3
    a = w[0::3]
    b = w[1::3]
    c = w[2::3]

    x_input = np.asarray(x).reshape(-1, 1)
    Z = x_input * b + c
    H = relu(Z)
    predictions = np.dot(H, a) / m
    return predictions, Z


def grad_forward(w, x, Z):
    """Gradient of f_w(x_i) with respect to w, for every sample.

    Returns:
        array of shape (n, 3*m), one gradient vector per row.
    """
    m = len(w) // 3
    n = len(x)

    a = w[0::3]
    x_input = np.asarray(x).reshape(-1, 1)

    H = relu(Z)
    H_der = relu_grad(Z)

    grad_a = H / m
    core_inner = a * H_der / m
    grad_b = core_inner * x_input
    grad_c = core_inner

    return np.stack([grad_a, grad_b, grad_c], axis=2).reshape(n, 3 * m)


def mse_loss(y_true, y_pred):
    """Mean squared error with a 1/2 factor"""
    return np.mean((y_true - y_pred) ** 2) / 2


def grad_loss(w, x, y, decay):
    """Gradient of mse_loss + regul.term : (decay / 2) * ||w||^2 with respect to w."""
    n = len(x)
    y_pred, Z = forward(w, x)
    error = y_pred - y
    grad_fw = grad_forward(w, x, Z)
    grad = np.sum(error[:, None] * grad_fw, axis=0) / n
    return grad + decay * w
