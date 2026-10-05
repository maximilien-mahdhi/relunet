import numpy as np
import pytest

from relunet.model import forward, grad_loss, init_weights, mse_loss
from relunet.training import train_gd, train_outer_only, train_sgd


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, 12)
    y = np.abs(x) - 0.5
    return x[:8], y[:8], x[8:], y[8:]


def test_init_is_reproducible():
    np.random.seed(0)
    first = init_weights(5, 1.0, 1.0)
    np.random.seed(0)
    second = init_weights(5, 1.0, 1.0)
    assert first.shape == (15,)
    assert np.array_equal(first, second)


def test_forward_matches_definition():
    w = np.array([2.0, 1.0, 0.5, -1.0, -2.0, 1.0])
    x = np.array([-1.0, 0.0, 1.5])
    pred, Z = forward(w, x)
    expected = [(2 * max(x_i + 0.5, 0) - max(-2 * x_i + 1, 0)) / 2 for x_i in x]
    assert Z.shape == (3, 2)
    assert np.allclose(pred, expected)


@pytest.mark.parametrize("decay", [0.0, 0.1])
def test_grad_loss_matches_finite_differences(data, decay):
    x, y, _, _ = data
    np.random.seed(1)
    w = init_weights(4, 1.0, 1.0)

    def penalized_loss(v):
        return mse_loss(y, forward(v, x)[0]) + 0.5 * decay * np.sum(v**2)

    eps = 1e-6
    numerical = np.zeros_like(w)
    for i in range(len(w)):
        step = np.zeros_like(w)
        step[i] = eps
        numerical[i] = (penalized_loss(w + step) - penalized_loss(w - step)) / (2 * eps)

    assert np.allclose(grad_loss(w, x, y, decay), numerical, atol=1e-5)


def test_gd_decreases_loss(data):
    x_train, y_train, x_test, y_test = data
    np.random.seed(2)
    _, train_losses, _ = train_gd(
        x_train,
        y_train,
        x_test,
        y_test,
        m=10,
        alpha_inner=1.0,
        alpha_outer=1.0,
        learning_rate=0.1,
        max_iter=200,
    )
    assert train_losses[-1] < train_losses[0]


def test_sgd_decreases_loss(data):
    x_train, y_train, x_test, y_test = data
    np.random.seed(2)
    _, train_losses, _ = train_sgd(
        x_train,
        y_train,
        x_test,
        y_test,
        m=10,
        alpha_inner=1.0,
        alpha_outer=1.0,
        learning_rate=0.05,
        max_epochs=100,
    )
    assert train_losses[-1] < train_losses[0]


def test_outer_only_keeps_inner_weights_frozen(data):
    x_train, y_train, x_test, y_test = data
    np.random.seed(3)
    w0 = init_weights(5, 1.0, 1.0)
    w, _, _ = train_outer_only(w0, x_train, y_train, x_test, y_test, learning_rate=0.1, max_iter=50)
    assert np.array_equal(w[1::3], w0[1::3])
    assert np.array_equal(w[2::3], w0[2::3])
    assert not np.array_equal(w[0::3], w0[0::3])
