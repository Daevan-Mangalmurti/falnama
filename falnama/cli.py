"""Command-line interface for Falnama.

Thin by design: parse arguments, load + validate config, hand off to the
pipeline. All real work lives in the package so the same logic runs from the
CLI, `python -m falnama`, notebooks, tests, and CI without going through a shell.
"""

from __future__ import annotations

import argparse
import sys

from .config import ConfigError, load_config
from .pipeline import STAGE_ORDER


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="falnama", description="Run the Falnama research pipeline.")
    parser.add_argument("--config", default=None, help="Path to falnama.yaml (default: auto-detect).")
    parser.add_argument("--check", action="store_true",
                        help="Validate config and print a summary, then exit without running.")
    parser.add_argument("--stage", action="append", choices=STAGE_ORDER,
                        help="Run only this stage (repeatable). Default: all, in order.")
    args = parser.parse_args(argv)

    try:
        settings = load_config(args.config)
    except ConfigError as exc:
        print(f"Config error:\n{exc}", file=sys.stderr)
        return 2

    if args.check:
        _print_summary(settings)
        return 0

    from .pipeline import run
    ctx = run(settings, stages=args.stage)
    ok = not ctx.errors
    print(f"Run {ctx.run_id} {'complete' if ok else 'FINISHED WITH ERRORS'}. "
          f"See {settings.output_dir('run_logs')}/pipeline_health_{ctx.run_id}.json")
    return 0 if ok else 1


def _print_summary(settings) -> None:
    """Show the resolved, validated settings — a quick confidence check."""
    print("Falnama configuration OK.")
    print(f"  project root      : {settings.project_root}")
    print(f"  paper_trading_only: {settings.paper_trading_only}")
    print(f"  card_mode         : {settings.card_mode}")
    print(f"  data.source       : {settings.data_source}")
    print(f"  execution.mode    : {settings.execution_mode}")
    print(f"  backfill_mode     : {settings.backfill_mode}")
    print(f"  llm model         : {settings.llm.get('model')}")
    print(f"  stages            : {', '.join(STAGE_ORDER)}")


if __name__ == "__main__":
    raise SystemExit(main())
