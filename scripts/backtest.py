"""Run the Phase 1 detection backtest from a clone: `python scripts/backtest.py`.

Replays the curated resolved markets in config/backtest_markets.yaml through the
live detector and prints the results table + summary (detection recall on the
documented cases, false-positive rate on the controls). LLM-free and read-only —
it only reads Polymarket's public price history, so it needs no API key.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json  # noqa: E402

import pandas as pd  # noqa: E402

from falnama import backtest  # noqa: E402


def main() -> int:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 44)
    table, summary = backtest.run()
    cols = [c for c in ["label", "market_name", "resolved_outcome", "history_points",
                        "peak_anomaly_score", "detected_strong", "lead_hours", "error"]
            if c in table.columns]
    print(table[cols].to_string(index=False))
    print()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
