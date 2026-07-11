"""End-to-end: the orchestrator runs the whole chain on fixtures and leaves a
complete, consistent audit trail. Runs in a temp copy of config + fixtures, so
it writes nothing into the repo's real outputs/."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from falnama.config import load_config
from falnama.pipeline import run

REPO = Path(__file__).resolve().parent.parent


def _isolated(tmp_path: Path):
    """A self-contained project rooted at tmp_path: real config + fixtures copied
    in, so the whole pipeline reads and writes only under the temp directory."""
    (tmp_path / "config").mkdir()
    shutil.copy(REPO / "config" / "falnama.yaml", tmp_path / "config" / "falnama.yaml")
    (tmp_path / "data" / "fixtures").mkdir(parents=True)
    for csv in (REPO / "data" / "fixtures").glob("*.csv"):
        shutil.copy(csv, tmp_path / "data" / "fixtures" / csv.name)
    settings = load_config(tmp_path / "config" / "falnama.yaml")
    # Pin the fixture path so the end-to-end test stays deterministic and offline
    # no matter what the committed config sets — main may be flipped to live for
    # real runs, and CI must not depend on the network or on live market counts.
    settings.raw["data"]["source"] = "fixtures"
    settings.raw["card_mode"] = "mock"
    return settings


def _manifest(settings, run_id: str) -> dict:
    return json.loads((settings.output_dir("run_logs") / f"run_manifest_{run_id}.json").read_text())


def test_pipeline_runs_end_to_end(tmp_path):
    settings = _isolated(tmp_path)
    ctx = run(settings)

    # Health report: every stage ran and succeeded.
    health = json.loads((settings.output_dir("run_logs") / f"pipeline_health_{ctx.run_id}.json").read_text())
    assert not ctx.errors and health["success"] is True
    assert all(health["stages"].values())

    # Manifest ties the stages together with sensible counts.
    manifest = _manifest(settings, ctx.run_id)
    assert manifest["market_selector"]["selected_count"] == 6
    assert manifest["anomaly_detector"]["strong_count"] == 2
    assert manifest["anomaly_detector"]["concentration_red_flags"] == 2
    assert manifest["cards"]["cards_written"] == 6
    # Default (backfill off): strong signals are rejected as ex-post -> no trades.
    assert manifest["recommender"]["recommended_count"] == 0

    # The durable artifacts exist.
    assert (settings.output_dir("relevant_markets") / "relevant_markets_latest.csv").exists()
    assert list(settings.output_dir("index_cards").glob("*.json"))


def test_pipeline_backfill_produces_recommendations(tmp_path):
    settings = _isolated(tmp_path)
    settings.raw["backfill_mode"] = True  # relax anti-ex-post for a historical backtest
    ctx = run(settings)
    manifest = _manifest(settings, ctx.run_id)
    assert manifest["recommender"]["recommended_count"] >= 1
