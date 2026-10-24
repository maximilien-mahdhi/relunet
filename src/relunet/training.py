"""Optimizers for the ReLU network."""

import logging
import time

import numpy as np

from relunet.model import forward, grad_loss, init_weights, mse_loss

log = logging.getLogger(__name__)

#: Result of a training run: final weights, train loss history, test loss history.
TrainResult = tuple[np.ndarray, list[float], list[float]]


def train_gd(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    m: int,
    alpha_inner: float,
    alpha_outer: float,
    learning_rate: float = 2.0,
    prec: float = 1e-2,
    max_iter: int = 1000,
    decay: float = 0.0,
    show_steps: bool = False,
) -> TrainResult:
    """Full-batch gradient descent.

        w ← w − η ∇L_λ(w)

    Stops after ``max_iter`` iterations or once the training loss is below ``prec``.
    Progress is logged at INFO level when ``show_steps`` is True, at DEBUG otherwise.

    Parameters
    ----------
    x_train, y_train, x_test, y_test:
        Training and test samples, of shape (n,).
    m:
        Number of hidden units.
    alpha_inner, alpha_outer:
        Initialization variances (see :func:`relunet.model.init_weights`).
    learning_rate:
        Step size η.
    prec:
        Target training loss.
    max_iter:
        Maximal number of iterations.
    decay:
        Weight decay λ.
    show_steps:
        Log progress at INFO level.

    Returns
    -------
    TrainResult
        Final weights, train loss history, test loss history.
    """
    level = logging.INFO if show_steps else logging.DEBUG

    w = init_weights(m, alpha_inner, alpha_outer)
    train_loss_history = []
    test_loss_history = []

    error = mse_loss(y_train, forward(w, x_train)[0])
    nb_iter = 0
    diverged = False

    log.log(level, "Starting gradient descent (m=%d, lr=%g)", m, learning_rate)
    start = time.time()

    while nb_iter < max_iter and error > prec:
        grad = grad_loss(w, x_train, y_train, decay)
        w -= learning_rate * grad

        error = mse_loss(y_train, forward(w, x_train)[0])
        train_loss_history.append(error)
        test_error = mse_loss(y_test, forward(w, x_test)[0])
        test_loss_history.append(test_error)

        if not diverged and not np.isfinite(error):
            log.warning("Training diverged at iteration %d (lr=%g)", nb_iter, learning_rate)
            diverged = True

        if nb_iter == 0 or (nb_iter + 1) % 100 == 0:
            log.log(
                level,
                "Iteration %d | train %.6f | test %.6f | grad norm %.4e",
                nb_iter,
                error,
                test_error,
                np.linalg.norm(grad),
            )
        nb_iter += 1

    elapsed = time.time() - start
    if train_loss_history:
        log.log(
            level,
            "Done in %.2fs, %d iterations | final train %.6f | final test %.6f",
            elapsed,
            nb_iter,
            train_loss_history[-1],
            test_loss_history[-1],
        )
    return w, train_loss_history, test_loss_history


def train_sgd(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    m: int,
    alpha_inner: float,
    alpha_outer: float,
    learning_rate: float = 2.0,
    prec: float = 1e-2,
    max_epochs: int = 1000,
    decay: float = 0.0,
    batch_size: int = 1,
    show_steps: bool = False,
) -> TrainResult:
    """Mini-batch stochastic gradient descent, with losses recorded once per epoch.

    At each epoch the samples are shuffled and split into batches B of size
    ``batch_size``; for each batch, w ← w − η ∇L_λ(w; B).
    Progress is logged at INFO level when ``show_steps`` is True, at DEBUG otherwise.

    Parameters
    ----------
    x_train, y_train, x_test, y_test:
        Training and test samples, of shape (n,).
    m:
        Number of hidden units.
    alpha_inner, alpha_outer:
        Initialization variances (see :func:`relunet.model.init_weights`).
    learning_rate:
        Step size η.
    prec:
        Target training loss.
    max_epochs:
        Maximal number of epochs.
    decay:
        Weight decay λ.
    batch_size:
        Number of samples per batch.
    show_steps:
        Log progress at INFO level.

    Returns
    -------
    TrainResult
        Final weights, train loss history, test loss history.
    """
    level = logging.INFO if show_steps else logging.DEBUG

    n_samples = len(x_train)
    n_batches = n_samples // batch_size

    w = init_weights(m, alpha_inner, alpha_outer)
    train_loss_history = []
    test_loss_history = []

    error = mse_loss(y_train, forward(w, x_train)[0])
    nb_epoch = 0
    diverged = False

    log.log(
        level,
        "Starting SGD (m=%d, lr=%g, batch size=%d)",
        m,
        learning_rate,
        batch_size,
    )
    start = time.time()

    while nb_epoch < max_epochs and error > prec:
        indices = np.random.permutation(n_samples)

        for i in range(n_batches):
            batch_indices = indices[i * batch_size : (i + 1) * batch_size]
            grad = grad_loss(w, x_train[batch_indices], y_train[batch_indices], decay)
            w -= learning_rate * grad

        error = mse_loss(y_train, forward(w, x_train)[0])
        test_error = mse_loss(y_test, forward(w, x_test)[0])
        train_loss_history.append(error)
        test_loss_history.append(test_error)

        if not diverged and not np.isfinite(error):
            log.warning("Training diverged at epoch %d (lr=%g)", nb_epoch, learning_rate)
            diverged = True

        if nb_epoch == 0 or (nb_epoch + 1) % 100 == 0:
            log.log(
                level,
                "Epoch %d | train %.6f | test %.6f",
                nb_epoch,
                error,
                test_error,
            )
        nb_epoch += 1

    elapsed = time.time() - start
    if train_loss_history:
        log.log(
            level,
            "Done in %.2fs, %d epochs | final train %.6f | final test %.6f",
            elapsed,
            nb_epoch,
            train_loss_history[-1],
            test_loss_history[-1],
        )
    return w, train_loss_history, test_loss_history


def train_outer_only(
    w_initial: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    learning_rate: float = 0.01,
    prec: float = 1e-6,
    max_iter: int = 1000,
    decay: float = 0.0,
) -> TrainResult:
    """Gradient descent on the outer weights a only; the inner weights b and c stay frozen.

        a ← a − η ∇_a L_λ(w)

    With b and c fixed, f_w is linear in a, so the problem is convex in a.

    Parameters
    ----------
    w_initial:
        Initial weights of shape (3m,); its inner weights are kept.
    x_train, y_train, x_test, y_test:
        Training and test samples, of shape (n,).
    learning_rate:
        Step size η.
    prec:
        Target training loss.
    max_iter:
        Maximal number of iterations.
    decay:
        Weight decay λ.

    Returns
    -------
    TrainResult
        Final weights, train loss history, test loss history.
    """
    w = w_initial.copy()

    train_loss_history = []
    test_loss_history = []

    error = mse_loss(y_train, forward(w, x_train)[0])
    nb_iter = 0

    while nb_iter < max_iter and error > prec:
        grad = grad_loss(w, x_train, y_train, decay)
        w[0::3] -= learning_rate * grad[0::3]

        error = mse_loss(y_train, forward(w, x_train)[0])
        train_loss_history.append(error)
        test_loss_history.append(mse_loss(y_test, forward(w, x_test)[0]))
        nb_iter += 1

    if train_loss_history:
        log.debug(
            "Outer-only training: %d iterations | final train %.6f | final test %.6f",
            nb_iter,
            train_loss_history[-1],
            test_loss_history[-1],
        )
    return w, train_loss_history, test_loss_history
