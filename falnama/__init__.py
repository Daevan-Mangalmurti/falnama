"""Falnama — prediction-market anomaly research pipeline.

The package is organized as one module per stage of the signal chain, plus three
foundation modules (config, io, schema). The most common entry points are
re-exported here so callers can write `from falnama import load_config, run`.

See README.md for the map, or open any module: each starts with a plain-language
header describing what it does, what it reads, and what it produces.
"""

from __future__ import annotations

from .config import ConfigError, Settings, load_config
from .io import RunContext

__all__ = ["ConfigError", "Settings", "load_config", "RunContext", "run"]

__version__ = "0.1.0"


def run(*args, **kwargs):
    """Convenience wrapper for falnama.pipeline.run (imported lazily to keep
    `import falnama` cheap and free of heavy dependencies)."""
    from .pipeline import run as _run

    return _run(*args, **kwargs)
