"""Figures for the experiments.

Every function either displays the figure or, when `save_path` is given,
writes it to disk and closes it.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from relunet.model import forward

log = logging.getLogger(__name__)


def _finish(fig, save_path):
    fig.tight_layout()
    if save_path is None:
        plt.show()
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure to %s", save_path)


def _finite(values, max_value=1e12):
    """Replace non-finite or astronomically large values (diverged runs) by NaN so that
    they are not drawn; matplotlib cannot build log-scale ticks for them."""
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values) & (np.abs(values) < max_value), values, np.nan)


def _grid(x_train, margin=0.5, n=100):
    return np.linspace(x_train.min() - margin, x_train.max() + margin, n)


def _info_box(ax, text):
    ax.text(
        0.10,
        0.30,
        text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"),
    )


def plot_data(x_train, y_train, x_test, y_test, save_path=None):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(x_train, y_train, color="red", label="Train samples")
    ax.scatter(x_test, y_test, color="blue", label="Test samples")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    _finish(fig, save_path)


def plot_hand_fit(x_train, y_train, w, save_path=None):
    x_line = np.linspace(-2, 2, 500)
    y_line, _ = forward(w, x_line)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(x_train, y_train, color="red", label="Train samples")
    ax.plot(x_line, y_line, color="green", label="Model prediction")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Example of a perfect fit with 4 ReLUs")
    ax.legend()
    _finish(fig, save_path)


def plot_single_run(
    w, train_losses, test_losses, x_train, y_train, x_test, y_test, params, save_path=None
):
    """Loss history and final predictions on train and test data for one run."""
    m = params["m"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(train_losses, label="Train Loss", color="red")
    axes[0].plot(test_losses, label="Test Loss", color="blue", linestyle="--")
    axes[0].set_xlabel("Iteration / Epoch")
    axes[0].set_ylabel("Loss L(w)")
    axes[0].set_title(f"Loss History (m={m})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    x_vals = np.linspace(
        min(x_train.min(), x_test.min()) - 1, max(x_train.max(), x_test.max()) + 1, 200
    )
    y_vals, _ = forward(w, x_vals)

    common = (
        f"Iterations: {len(train_losses)}\n"
        f"training_precision: {params['prec']}\n"
        f"alpha_inner: {params['alpha_inner']}\n"
        f"alpha_outer: {params['alpha_outer']}"
    )
    panels = [
        (axes[1], x_train, y_train, "red", "Train", train_losses),
        (axes[2], x_test, y_test, "blue", "Test", test_losses),
    ]
    for ax, xs, ys, color, name, losses in panels:
        ax.plot(x_vals, y_vals, color="green", label="Prediction")
        ax.scatter(xs, ys, color=color, label=f"{name} samples", alpha=0.7)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        _info_box(ax, f"{name} error: {losses[-1]:.4f}\n{common}")
        ax.legend()
        ax.set_title(f"ReLU Network Prediction on {name} Data (m={m})")
    _finish(fig, save_path)


def plot_gd_vs_sgd(gd, sgd, x_train, y_train, x_test, y_test, m, lr, save_path=None):
    """`gd` and `sgd` are (weights, train losses, test losses) tuples."""
    w_gd, gd_train, gd_test = gd
    w_sgd, sgd_train, sgd_test = sgd

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Comparison between GD and SGD (Learning Rate = {lr},  M={m})", fontsize=14)

    axes[0].plot(gd_train, label="GD Train Loss", color="red", alpha=0.8, linestyle="-")
    axes[0].plot(gd_test, label="GD Test Loss", color="red", linestyle="--", alpha=0.8)
    axes[0].plot(sgd_train, label="SGD Train Loss", color="blue", alpha=0.8, linestyle="-")
    axes[0].plot(sgd_test, label="SGD Test Loss", color="blue", linestyle="--", alpha=0.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss L(w)")
    axes[0].set_ylim(0, 2)
    axes[0].set_title("GD vs SGD\n Comparison of Loss Histories")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    x_plot = _grid(x_train, n=100)
    axes[1].scatter(x_train, y_train, color="red", label="Train data", alpha=0.7)
    axes[1].scatter(x_test, y_test, color="blue", label="Test data", alpha=0.7)
    axes[1].plot(
        x_plot, forward(w_gd, x_plot)[0], label="GD prediction", color="orange", linewidth=2
    )
    axes[1].plot(
        x_plot, forward(w_sgd, x_plot)[0], label="SGD prediction", color="green", linewidth=2
    )
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("GD vs SGD\n Final Predictions Comparison")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    _finish(fig, save_path)


def plot_init_scales(x_train, y_train, scales, runs, w_inits, m, lr, save_path=None):
    """`runs` maps each scale to (weights, train losses, test losses)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        r"$\bf{Initialization\ Scale\ Comparison\ for\ GD}$" + "\n"
        f"(Learning Rate = {lr}, Width = {m})",
        fontsize=14,
        y=1.05,
    )
    x_plot = _grid(x_train, n=100)

    axes[1].scatter(x_train, y_train, color="red", label="Train data", alpha=0.7)
    for scale in scales:
        w_final, train_losses, test_losses = runs[scale]
        axes[0].plot(train_losses, label=f"Train (s={scale})", alpha=0.7, linestyle="-")
        axes[0].plot(test_losses, label=f"Test (s={scale})", alpha=0.5, linestyle=":")
        axes[1].plot(x_plot, forward(w_final, x_plot)[0], label=f"Final (s={scale})", linewidth=2)
        axes[2].plot(x_plot, forward(w_inits[scale], x_plot)[0], label=f"scale={scale}", alpha=0.7)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss L(w)")
    axes[0].set_title("Loss History vs Initialization Scale")
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Final Predictions vs Initialization Scale")
    axes[1].legend(loc="lower left", fontsize=8)
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_title("Initial Functions (Before Training)")
    axes[2].legend(loc="lower left", fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    _finish(fig, save_path)


def plot_widths(x_train, y_train, widths, runs, lr, save_path=None):
    """`runs` maps each width to (weights, train losses, test losses)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        r"$\bf{Network\ Width\ Comparison\ for\ GD}$" + "\n" + f"(Learning Rate = {lr})",
        fontsize=14,
        y=1.05,
    )
    x_plot = _grid(x_train, n=100)

    axes[1].scatter(x_train, y_train, color="red", label="Train data", alpha=0.7)
    for m in widths:
        w_final, train_losses, test_losses = runs[m]
        axes[0].plot(train_losses, label=f"Train (m={m})", alpha=0.7, linestyle="-")
        axes[0].plot(test_losses, label=f"Test (m={m})", alpha=0.5, linestyle=":")
        axes[1].plot(x_plot, forward(w_final, x_plot)[0], label=f"Final (m={m})", linewidth=2)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss L(w)")
    axes[0].set_title("Loss History vs Network Width")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Predictions vs Network Width")
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    final_test = [runs[m][2][-1] for m in widths]
    axes[2].bar(range(len(widths)), final_test, tick_label=widths)
    axes[2].set_xlabel("Network Width (m)")
    axes[2].set_ylabel("Final Test Loss L(w)")
    axes[2].set_title("Test Loss vs Network Width")
    axes[2].grid(axis="y", alpha=0.3)
    _finish(fig, save_path)


def plot_learning_rates(
    x_train, y_train, rates, runs, convergence_epochs, m, threshold, save_path=None
):
    """`runs` maps each learning rate to (weights, train losses, test losses)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        r"$\bf{Learning\ Rate\ Comparison\ for\ GD}$" + "\n" + f"(Network Width = {m})",
        fontsize=14,
        y=1.05,
    )
    x_plot = _grid(x_train, n=100)

    axes[1].scatter(x_train, y_train, color="red", label="Train data", alpha=0.7)
    for lr in rates:
        w_final, train_losses, test_losses = runs[lr]
        axes[0].semilogy(_finite(train_losses), label=f"Train (lr={lr})", alpha=0.7, linestyle="-")
        axes[0].semilogy(_finite(test_losses), label=f"Test (lr={lr})", alpha=0.5, linestyle=":")
        axes[1].plot(x_plot, forward(w_final, x_plot)[0], label=f"Final (lr={lr})", linewidth=2)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss L(w) (log scale)")
    axes[0].set_title("Loss History vs Learning Rate \n (for GD)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Final Predictions vs Learning Rate \n (for GD)")
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(
        range(len(rates)),
        [convergence_epochs[lr] for lr in rates],
        tick_label=[str(lr) for lr in rates],
    )
    axes[2].set_xlabel("Learning Rate (lr)")
    axes[2].set_ylabel(f"Epochs to Converge (loss < {threshold})")
    axes[2].set_title("Convergence Speed Analysis \n (for GD)")
    axes[2].grid(axis="y", alpha=0.3)
    _finish(fig, save_path)


def plot_weight_decay(x_train, y_train, x_test, y_test, lambdas, runs, m, lr, save_path=None):
    """`runs` maps each decay strength to (weights, train losses, test losses)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        r"$\bf{Weight\ Decay\ Comparison\ for\ SGD}$" + f"\n(Learning Rate = {lr}, Width = {m})",
        fontsize=14,
        y=1.05,
    )
    x_plot = _grid(x_train, n=100)

    axes[1].scatter(x_train, y_train, color="red", label="Train data", alpha=0.7)
    axes[1].scatter(x_test, y_test, color="blue", label="Test data", alpha=0.7)
    for lam in lambdas:
        w_final, train_losses, test_losses = runs[lam]
        axes[0].plot(train_losses, label=f"Train (λ={lam})", alpha=0.7, linestyle="-")
        axes[0].plot(test_losses, label=f"Test (λ={lam})", alpha=0.5, linestyle=":")
        axes[1].plot(x_plot, forward(w_final, x_plot)[0], label=f"Final (λ={lam})", linewidth=2)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss L(w)")
    axes[0].set_title("Loss History vs Weight Decay (λ)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Predictions vs Weight Decay (λ)")
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    norms = [np.linalg.norm(runs[lam][0]) ** 2 for lam in lambdas]
    axes[2].bar(range(len(lambdas)), norms, tick_label=[str(lam) for lam in lambdas])
    axes[2].set_xlabel("Weight Decay (λ)")
    axes[2].set_ylabel("Squared L2 Norm $\\Vert w \\Vert_2^2$")
    axes[2].set_title("Weight Norm vs Regularization")
    axes[2].grid(axis="y", alpha=0.3)
    _finish(fig, save_path)


def plot_freeze(x_train, y_train, all_run, frozen_run, w_initial, m, lr, save_path=None):
    """`all_run` and `frozen_run` are (weights, train losses, test losses) tuples."""
    w_all, all_train, all_test = all_run
    w_frozen, frozen_train, frozen_test = frozen_run

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        r"$\bf{Freezing\ Inner\ Layer\ Comparison\ for\ GD}$" + "\n"
        f"(Learning Rate = {lr}, Width = {m})",
        fontsize=14,
        y=1.05,
    )

    axes[0].plot(all_train, label="Train all (Train)", alpha=0.7, color="blue", linestyle="-")
    axes[0].plot(all_test, label="Train all (Test)", alpha=0.7, color="blue", linestyle=":")
    axes[0].plot(frozen_train, label="Frozen Inner (Train)", alpha=0.7, color="red", linestyle="-")
    axes[0].plot(frozen_test, label="Frozen Inner (Test)", alpha=0.7, color="red", linestyle=":")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss L(w)")
    axes[0].set_title("Loss: All Weights vs Outer Layer Only")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    x_plot = _grid(x_train, n=100)
    axes[1].scatter(x_train, y_train, color="red", label="Train data", alpha=0.7)
    axes[1].plot(
        x_plot, forward(w_all, x_plot)[0], label="Train all weights", color="blue", linewidth=2
    )
    axes[1].plot(
        x_plot,
        forward(w_frozen, x_plot)[0],
        label="Train only outer layer",
        color="orange",
        linewidth=2,
    )
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Final Predictions")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    initial_b = w_initial[1::3]
    initial_c = w_initial[2::3]
    for j in range(min(5, m)):
        activation = np.maximum(0, initial_b[j] * x_plot + initial_c[j])
        axes[2].plot(x_plot, activation, alpha=0.7, label=f"Neuron {j + 1}")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("Activation $\\max(0, b_j x + c_j)$")
    axes[2].set_title(f"Frozen Inner Layer Features (M={m})")
    axes[2].legend(fontsize=8, loc="upper left")
    axes[2].grid(True, alpha=0.3)
    _finish(fig, save_path)
