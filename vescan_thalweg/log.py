"""Pipeline logging: console + one file per patient (<results>/<patient>/pipeline.log).

Every module uses logging.getLogger(__name__) (all under "vescan_thalweg"); only level, format and
destinations are configured here. Each message shows its module without the "vescan_thalweg." prefix
(e.g. "stages.anatomy.reachability"), so it is clear which stage it comes from.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager

ROOT = "vescan_thalweg"
LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
FORMAT = "%(asctime)s %(levelname)-7s %(module_name)s: %(message)s"


class _ShortName(logging.Filter):
    def filter(self, record):
        record.module_name = record.name[len(ROOT) + 1:] if record.name.startswith(ROOT + ".") else record.name
        return True


def setup(level="INFO"):
    """Configure the "vescan_thalweg" logger (idempotent): level and console."""
    root = logging.getLogger(ROOT)
    root.setLevel(level)
    root.propagate = False
    if not any(getattr(h, "_vescan_thalweg_console", False) for h in root.handlers):
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(FORMAT, datefmt="%H:%M:%S"))
        h.addFilter(_ShortName())
        h._vescan_thalweg_console = True
        root.addHandler(h)
    return root


@contextmanager
def to_file(path):
    """Also copy the messages to `path` while the block is open (one file per patient)."""
    h = logging.FileHandler(path, mode="a", encoding="utf-8")
    h.setFormatter(logging.Formatter(FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    h.addFilter(_ShortName())
    root = logging.getLogger(ROOT)
    root.addHandler(h)
    try:
        yield
    finally:
        root.removeHandler(h)
        h.close()
