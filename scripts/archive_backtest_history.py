"""Freeze the backtest cases' hourly price histories: `python scripts/archive_backtest_history.py`.

The backtest replays months-old markets at hourly resolution. Only Polymarket's
old CLOB /prices-history route still serves that (its v2 replacement keeps hourly
data for a few months only), so this copies each case's history into
data/backtest_archive/ while it still can. The backtest then reads the archive
first and needs the old route only for cases not archived yet.

Already-archived markets are left exactly as they are (the archive is a frozen
record); add --refresh to fetch them again. Read-only against Polymarket; no key.
Commit data/backtest_archive/ afterwards.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from falnama import backtest  # noqa: E402
from falnama.config import load_config  # noqa: E402


def main() -> int:
    settings = load_config()
    manifest = backtest.archive_histories(backtest.load_cases(settings=settings), settings,
                                          refresh="--refresh" in sys.argv[1:])
    for token, entry in manifest["tokens"].items():
        print(f"{entry['points']:5d} points  {entry['window_start_utc'][:10]} → {entry['window_end_utc'][:10]}  "
              f"{entry['market_name'][:60]}")
    for failure in manifest["failures"]:
        print(f"FAILED  {failure['event']} {failure['market_contains']!r}: {failure['error']}")
    print(f"\n{len(manifest['tokens'])} markets archived in {backtest.ARCHIVE_DIR}/; "
          f"{len(manifest['failures'])} failed. Commit the folder to keep them.")
    return 0 if not manifest["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
