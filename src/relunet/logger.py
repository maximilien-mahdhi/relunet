"""Terminal logging configuration based on Rich."""

import logging

from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger to write formatted records to stderr.

    Should be called once, at program startup.
    """
    handler = RichHandler(
        console=_console,
        show_time=True,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        markup=False,
        log_time_format="[%H:%M:%S]",
    )
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[handler],
        force=True,
    )
