"""Compile the pipeline's output folders into tables for review.

WHAT:     Read-only helpers that load one run's artifacts and the cross-run
          history from outputs/, so an analyst can look at results without
          re-implementing the parsing or reading raw files in an editor.
CONSUMES: outputs/ (run manifests + health, relevant_markets, anomalies,
          recommendations, rejected_signals)
PRODUCES: pandas DataFrames / a small RunFiles bundle in memory (writes nothing)
REVIEWER: an analyst using notebooks/analyst_review.ipynb
ROLE:     the READ side of the audit trail. The pipeline writes durable files;
          this turns them back into tables. Keeping it here (not in the notebook)
          means the notebook stays thin and the parsing is reusable and testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import io
from .config import Settings, load_config


def run_history(settings: Settings | None = None) -> pd.DataFrame:
    """One row per completed run (parsed from the run manifests), oldest first.

    This is the at-a-glance trend of the whole system: how many markets were
    selected, scored, flagged strong, red-flagged, and recommended each run.
    """
    settings = settings or load_config()
    rows = []
    for path in sorted(settings.output_dir("run_logs").glob("run_manifest_*.json")):
        m = io.read_json(path, {})
        sel, an, rec = m.get("market_selector", {}), m.get("anomaly_detector", {}), m.get("recommender", {})
        rows.append({
            "run_id": m.get("run_id"),
            "run_time_utc": m.get("run_time_utc"),
            "data_source": m.get("data_source"),
            "card_mode": m.get("card_mode"),
            "selected": sel.get("selected_count"),
            "rejected_markets": sel.get("rejected_count"),
            "scored": an.get("markets_scored"),
            "strong": an.get("strong_count"),
            "red_flags": an.get("concentration_red_flags"),
            "recommended": rec.get("recommended_count"),
            "rejected_signals": rec.get("rejected_count"),
        })
    return pd.DataFrame(rows)


def latest_run_id(settings: Settings | None = None) -> str | None:
    """The most recent run_id present under outputs/run_logs/, or None."""
    settings = settings or load_config()
    manifests = sorted(settings.output_dir("run_logs").glob("run_manifest_*.json"))
    return io.read_json(manifests[-1], {}).get("run_id") if manifests else None


@dataclass
class RunFiles:
    """Everything one run produced, loaded into memory for review."""

    run_id: str
    manifest: dict
    health: dict
    relevant_markets: pd.DataFrame
    anomalies: pd.DataFrame        # the ranked anomalies
    strong: pd.DataFrame
    recommendations: pd.DataFrame
    rejected_signals: pd.DataFrame


def load_run(settings: Settings | None = None, run_id: str | None = None) -> RunFiles:
    """Load one run's artifacts (defaults to the latest run).

    Uses the run-specific timestamped files, so any past run can be reviewed, not
    only the latest. Missing/empty artifacts come back as empty frames.
    """
    settings = settings or load_config()
    run_id = run_id or latest_run_id(settings)
    if run_id is None:
        raise FileNotFoundError(
            "No runs found under outputs/run_logs/. Run the pipeline first "
            "(python scripts/run.py)."
        )
    logs = settings.output_dir("run_logs")

    def table(folder: str, basename: str) -> pd.DataFrame:
        return io.read_table(settings.output_dir(folder) / f"{basename}_{run_id}.csv")

    recommendations = pd.DataFrame(
        io.read_json(settings.output_dir("recommended_trades") / f"recommendations_{run_id}.json", [])
    )
    return RunFiles(
        run_id=run_id,
        manifest=io.read_json(logs / f"run_manifest_{run_id}.json", {}),
        health=io.read_json(logs / f"pipeline_health_{run_id}.json", {}),
        relevant_markets=table("relevant_markets", "relevant_markets"),
        anomalies=table("anomalies", "ranked_anomalies"),
        strong=table("anomalies", "strong_anomalies"),
        recommendations=recommendations,
        rejected_signals=table("rejected_signals", "rejected_signals"),
    )
