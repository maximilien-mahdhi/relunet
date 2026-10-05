"""Run one or all experiments from the command line.

Examples:
    python scripts/run_experiment.py gd-vs-sgd
    python scripts/run_experiment.py all --save-dir results
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from relunet.data import DEFAULT_DATA_DIR, load_data
from relunet.experiments import EXPERIMENTS
from relunet.logger import setup_logging

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment", choices=[*EXPERIMENTS, "all"])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Save figures as PNG files in this directory instead of displaying them.",
    )
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(getattr(logging, args.log_level))

    x_train, y_train, x_test, y_test = load_data(args.data_dir)
    names = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]

    for name in names:
        log.info("=== %s ===", name)
        np.random.seed(args.seed)
        save_path = args.save_dir / f"{name}.png" if args.save_dir else None
        EXPERIMENTS[name](x_train, y_train, x_test, y_test, save_path=save_path, seed=args.seed)


if __name__ == "__main__":
    main()
