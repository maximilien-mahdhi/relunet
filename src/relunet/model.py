"""Two-layer ReLU network.

The network is

    f_w(x) = (1/m) Σ_{j=1}^{m} a_j · ReLU(b_j x + c_j).

Its parameters are stored in a flat array w of length 3m laid out as
(a_1, b_1, c_1, a_2, b_2, c_2, …), which are the weights of the network.
"""

import numpy as np
from numpy.typing import ArrayLike


def relu(x: ArrayLike) -> np.ndarray:
    """ReLU activation, ReLU(x) = max(0, x)."""
    return np.maximum(0, x)


def relu_grad(x: ArrayLike) -> np.ndarray:
    """Subgradient of ReLU: 1 if x > 0, 0 otherwise (value 0 taken at x = 0)."""
    return np.where(x > 0, 1, 0)


def init_weights(m: int, alpha_inner: float, alpha_outer: float) -> np.ndarray:
    """Draw centered Gaussian weights.

        a_j ~ N(0, α_outer),   b_j, c_j ~ N(0, α_inner),   j = 1, …, m

    Parameters
    ----------
    m:
        Number of hidden units.
    alpha_inner:
        Variance α_inner of the inner weights b_j and c_j.
    alpha_outer:
        Variance α_outer of the outer weights a_j.

    Returns
    -------
    numpy.ndarray
        Weights w of shape (3m,), laid out as (a_1, b_1, c_1, …).
    """
    # One row (a_j, b_j, c_j) per unit, drawn in the same order as the flat layout.
    std = np.sqrt([alpha_outer, alpha_inner, alpha_inner])
    return (np.random.randn(m, 3) * std).ravel()


def forward(w: np.ndarray, x: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the network on all samples and units at once.

    Parameters
    ----------
    w:
        Weights of shape (3m,).
    x:
        Inputs x_i, of shape (n,).

    Returns
    -------
    predictions : numpy.ndarray
        f_w(x_i), of shape (n,).
    Z : numpy.ndarray
        Pre-activations Z_ij = b_j x_i + c_j, of shape (n, m), reused by the backward pass.
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


def grad_forward(w: np.ndarray, x: ArrayLike, Z: np.ndarray) -> np.ndarray:
    """Gradient of f_w(x_i) with respect to w, for every sample.

        ∂f/∂a_j = ReLU(Z_ij) / m
        ∂f/∂b_j = a_j · 1[Z_ij > 0] · x_i / m
        ∂f/∂c_j = a_j · 1[Z_ij > 0] / m

    Parameters
    ----------
    w:
        Weights of shape (3m,).
    x:
        Inputs x_i, of shape (n,).
    Z:
        Pre-activations returned by :func:`forward`, of shape (n, m).

    Returns
    -------
    numpy.ndarray
        Array of shape (n, 3m), one gradient vector per row.
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


def mse_loss(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean squared error with a 1/2 factor.

    For n samples,

        L = (1/2n) Σ_i (y_i − ŷ_i)²
    """
    return np.mean((y_true - y_pred) ** 2) / 2


def grad_loss(w: np.ndarray, x: np.ndarray, y: np.ndarray, decay: float) -> np.ndarray:
    """Gradient of the regularized loss with respect to w.

        L_λ(w) = (1/2n) Σ_i (f_w(x_i) − y_i)² + (λ/2) ‖w‖²

        ∇L_λ(w) = (1/n) Σ_i (f_w(x_i) − y_i) ∇_w f_w(x_i) + λ w

    Parameters
    ----------
    w:
        Weights of shape (3m,).
    x, y:
        Inputs and targets, of shape (n,).
    decay:
        Weight decay λ ≥ 0.

    Returns
    -------
    numpy.ndarray
        Gradient of shape (3m,).
    """
    n = len(x)
    y_pred, Z = forward(w, x)
    error = y_pred - y
    grad_fw = grad_forward(w, x, Z)
    grad = np.sum(error[:, None] * grad_fw, axis=0) / n
    return grad + decay * w
