"""Dataset loading."""

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_data(data_dir: Path | str = DEFAULT_DATA_DIR):
    """Load the 1D regression dataset.

    Each file holds an array of shape (2, n) with the inputs in the first row
    and the targets in the second.

    Returns:
        x_train, y_train, x_test, y_test
    """
    data_dir = Path(data_dir)
    x_train, y_train = np.load(data_dir / "train_data.npy")
    x_test, y_test = np.load(data_dir / "test_data.npy")
    log.info("Loaded %d train and %d test samples from %s", len(x_train), len(x_test), data_dir)
    return x_train, y_train, x_test, y_test
