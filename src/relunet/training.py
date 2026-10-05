"""Optimizers for the ReLU network."""

import logging
import time

import numpy as np

from relunet.model import forward, grad_loss, init_weights, mse_loss

log = logging.getLogger(__name__)


def train_gd(
    x_train,
    y_train,
    x_test,
    y_test,
    m,
    alpha_inner,
    alpha_outer,
    learning_rate=2.0,
    prec=1e-2,
    max_iter=1000,
    decay=0.0,
    show_steps=False,
):
    """Full-batch gradient descent.

    Progress is logged at INFO level when `show_steps` is True, at DEBUG otherwise.

    Returns:
        final weights, train loss history, test loss history
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
    x_train,
    y_train,
    x_test,
    y_test,
    m,
    alpha_inner,
    alpha_outer,
    learning_rate=2.0,
    prec=1e-2,
    max_epochs=1000,
    decay=0.0,
    batch_size=1,
    show_steps=False,
):
    """Mini-batch stochastic gradient descent, with losses recorded once per epoch.

    Progress is logged at INFO level when `show_steps` is True, at DEBUG otherwise.

    Returns:
        final weights, train loss history, test loss history
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
    w_initial,
    x_train,
    y_train,
    x_test,
    y_test,
    learning_rate=0.01,
    prec=1e-6,
    max_iter=1000,
    decay=0.0,
):
    """Gradient descent on the outer weights a only; the inner weights b and c stay frozen.

    The problem is convex in a.

    Returns:
        final weights, train loss history, test loss history
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
