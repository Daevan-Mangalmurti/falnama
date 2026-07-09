"""Run Falnama without installing it: `python scripts/run.py [--check] [--stage ...]`.

A two-line shim so the pipeline is runnable straight from a clone. It puts the
repo root on sys.path and calls the real CLI in falnama/cli.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from falnama.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
