"""Experiments on the two-layer ReLU network.

Every experiment has the signature
    experiment(x_train, y_train, x_test, y_test, save_path=None, seed=55)
and resets the NumPy seed before each training run, so that runs that are
compared start from the same initial weights.
"""

import logging

import numpy as np

from relunet import plotting
from relunet.model import forward, init_weights, mse_loss
from relunet.training import train_gd, train_outer_only, train_sgd

log = logging.getLogger(__name__)


def show_data(x_train, y_train, x_test, y_test, save_path=None, seed=55):
    """Plot the train and test samples."""
    plotting.plot_data(x_train, y_train, x_test, y_test, save_path)


def hand_fit(x_train, y_train, x_test, y_test, save_path=None, seed=55):
    """Zero training loss reached by hand with 4 ReLU units."""
    order = np.lexsort((-y_train, x_train))
    x = x_train[order]
    y = y_train[order]

    for name, (i, j) in zip(("1st", "2nd", "3rd"), ((1, 2), (4, 5), (7, 8)), strict=True):
        slope = (y[j] - y[i]) / (x[j] - x[i])
        intercept = y[i] - slope * x[i]
        log.info("%s line: y = %.6g x + %.6g", name, slope, intercept)

    # Weights (a_j, b_j, c_j) of the 4 units, computed by hand.
    w = np.array([4, -2, 2, -4, 2, -2, 16, 1, 1 / 4, -24, 1, -1])
    log.info("Training loss of the hand-made fit: %.3e", mse_loss(y_train, forward(w, x_train)[0]))
    plotting.plot_hand_fit(x_train, y_train, w, save_path)
    return w


def single_run(
    x_train,
    y_train,
    x_test,
    y_test,
    save_path=None,
    seed=55,
    m=30,
    alpha_inner=4,
    alpha_outer=5,
    learning_rate=3,
    prec=1e-2,
    max_iter=1000,
):
    """Train once with gradient descent and plot the loss history and predictions."""
    np.random.seed(seed)
    w, train_losses, test_losses = train_gd(
        x_train,
        y_train,
        x_test,
        y_test,
        m,
        alpha_inner,
        alpha_outer,
        learning_rate=learning_rate,
        prec=prec,
        max_iter=max_iter,
        show_steps=True,
    )
    params = {"m": m, "prec": prec, "alpha_inner": alpha_inner, "alpha_outer": alpha_outer}
    plotting.plot_single_run(
        w, train_losses, test_losses, x_train, y_train, x_test, y_test, params, save_path
    )
    return w, train_losses, test_losses


def gd_vs_sgd(x_train, y_train, x_test, y_test, save_path=None, seed=55):
    """Compare gradient descent and SGD from identical initial weights."""
    m, lr, epochs, prec, batch_size = 10, 0.5, 1000, 1e-4, 1

    log.info("Training with gradient descent")
    np.random.seed(seed)
    gd = train_gd(
        x_train,
        y_train,
        x_test,
        y_test,
        m=m,
        alpha_inner=1.0,
        alpha_outer=1.0,
        learning_rate=lr,
        prec=prec,
        max_iter=epochs,
    )

    log.info("Training with stochastic gradient descent")
    np.random.seed(seed)
    sgd = train_sgd(
        x_train,
        y_train,
        x_test,
        y_test,
        m=m,
        alpha_inner=1.0,
        alpha_outer=1.0,
        learning_rate=lr,
        prec=prec,
        max_epochs=epochs,
        batch_size=batch_size,
    )

    log.info("Final test loss | GD: %.6f | SGD: %.6f", gd[2][-1], sgd[2][-1])
    plotting.plot_gd_vs_sgd(gd, sgd, x_train, y_train, x_test, y_test, m, lr, save_path)
    return gd, sgd


def initialization_scales(x_train, y_train, x_test, y_test, save_path=None, seed=55):
    """Effect of the initialization scale (alpha_inner = alpha_outer = scale)."""
    scales = [0.01, 0.1, 0.5, 1.0, 2.0]
    m, lr, epochs, prec = 15, 1, 500, 1e-3

    runs = {}
    w_inits = {}
    for scale in scales:
        log.info("Training with initialization scale %s", scale)
        np.random.seed(seed)
        w_inits[scale] = init_weights(m, scale, scale)
        np.random.seed(seed)
        runs[scale] = train_gd(
            x_train,
            y_train,
            x_test,
            y_test,
            m=m,
            alpha_inner=scale,
            alpha_outer=scale,
            learning_rate=lr,
            prec=prec,
            max_iter=epochs,
        )

    for scale, (_, train_losses, test_losses) in runs.items():
        log.info("Scale %s | train %.6f | test %.6f", scale, train_losses[-1], test_losses[-1])
    plotting.plot_init_scales(x_train, y_train, scales, runs, w_inits, m, lr, save_path)
    return runs


def network_width(x_train, y_train, x_test, y_test, save_path=None, seed=55):
    """Effect of the network width m."""
    widths = [2, 10, 20, 50, 100]
    lr, epochs, prec = 1, 500, 1e-6

    runs = {}
    for m in widths:
        log.info("Training with width m=%d", m)
        np.random.seed(seed)
        runs[m] = train_gd(
            x_train,
            y_train,
            x_test,
            y_test,
            m=m,
            alpha_inner=1.0,
            alpha_outer=1.0,
            learning_rate=lr,
            prec=prec,
            max_iter=epochs,
        )

    final_test_losses = {m: run[2][-1] for m, run in runs.items()}
    for m, final_loss in final_test_losses.items():
        log.info("Width %d | final test loss %.6f", m, final_loss)
    plotting.plot_widths(x_train, y_train, widths, runs, lr, save_path)
    return final_test_losses


def learning_rate_sweep(x_train, y_train, x_test, y_test, save_path=None, seed=55):
    """Effect of the learning rate on convergence speed and fit."""
    rates = [0.01, 0.1, 1.0, 5.0, 10.0]
    m, epochs, prec = 10, 1000, 1e-3
    threshold = 0.1

    runs = {}
    convergence_epochs = {}
    for lr in rates:
        log.info("Training with learning rate %s", lr)
        np.random.seed(seed)
        runs[lr] = train_gd(
            x_train,
            y_train,
            x_test,
            y_test,
            m=m,
            alpha_inner=1.0,
            alpha_outer=1.0,
            learning_rate=lr,
            prec=prec,
            max_iter=epochs,
        )

        train_losses = np.array(runs[lr][1])
        converged_epoch = int(np.argmax(train_losses < threshold))
        if converged_epoch == 0 and train_losses[0] >= threshold:
            convergence_epochs[lr] = epochs
        else:
            convergence_epochs[lr] = converged_epoch

    for lr, n_epochs in convergence_epochs.items():
        log.info("LR=%s | %d epochs to reach loss < %s", lr, n_epochs, threshold)
    plotting.plot_learning_rates(
        x_train, y_train, rates, runs, convergence_epochs, m, threshold, save_path
    )
    return convergence_epochs


def weight_decay_sweep(x_train, y_train, x_test, y_test, save_path=None, seed=55):
    """Effect of the weight decay strength lambda, trained with SGD."""
    lambdas = [0.0, 0.001, 0.01, 0.1]
    m, lr, epochs, prec = 10, 0.2, 500, 1e-6

    runs = {}
    for lam in lambdas:
        log.info("Training with weight decay λ=%s", lam)
        np.random.seed(seed)
        runs[lam] = train_sgd(
            x_train,
            y_train,
            x_test,
            y_test,
            m=m,
            alpha_inner=1.0,
            alpha_outer=1.0,
            learning_rate=lr,
            prec=prec,
            max_epochs=epochs,
            decay=lam,
        )

    for lam, (w_final, _, test_losses) in runs.items():
        log.info(
            "λ=%s | test loss %.6f | squared norm %.2f",
            lam,
            test_losses[-1],
            np.linalg.norm(w_final) ** 2,
        )
    plotting.plot_weight_decay(x_train, y_train, x_test, y_test, lambdas, runs, m, lr, save_path)
    return runs


def freeze_inner_layer(x_train, y_train, x_test, y_test, save_path=None, seed=55):
    """Train all weights versus training only the outer layer (convex problem)."""
    m, lr, epochs, prec = 5, 1.0, 500, 1e-6

    log.info("Case 1: training all weights (non-convex)")
    np.random.seed(seed)
    all_run = train_gd(
        x_train,
        y_train,
        x_test,
        y_test,
        m=m,
        alpha_inner=1.0,
        alpha_outer=1.0,
        learning_rate=lr,
        prec=prec,
        max_iter=epochs,
    )

    log.info("Case 2: training only the outer layer (convex)")
    np.random.seed(seed)
    w_initial = init_weights(m, alpha_inner=1.0, alpha_outer=1.0)
    frozen_run = train_outer_only(
        w_initial,
        x_train,
        y_train,
        x_test,
        y_test,
        learning_rate=lr,
        prec=prec,
        max_iter=epochs,
    )

    log.info("Final test loss | all weights: %.6f", all_run[2][-1])
    log.info("Final test loss | outer layer only: %.6f", frozen_run[2][-1])
    plotting.plot_freeze(x_train, y_train, all_run, frozen_run, w_initial, m, lr, save_path)
    return all_run, frozen_run


EXPERIMENTS = {
    "data": show_data,
    "hand-fit": hand_fit,
    "single-run": single_run,
    "gd-vs-sgd": gd_vs_sgd,
    "init-scales": initialization_scales,
    "width": network_width,
    "learning-rates": learning_rate_sweep,
    "weight-decay": weight_decay_sweep,
    "freeze-inner": freeze_inner_layer,
}
