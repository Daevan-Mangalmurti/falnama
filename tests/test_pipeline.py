"""End-to-end: the orchestrator runs the whole chain on fixtures and leaves a
complete, consistent audit trail. Runs in a temp copy of config + fixtures, so
it writes nothing into the repo's real outputs/."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

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
    _make_fixtures_live(tmp_path / "data" / "fixtures")
    settings = load_config(tmp_path / "config" / "falnama.yaml")
    # Pin the fixture path so the end-to-end test stays deterministic and offline
    # no matter what the committed config sets — main may be flipped to live for
    # real runs, and CI must not depend on the network or on live market counts.
    settings.raw["data"]["source"] = "fixtures"
    settings.raw["card_mode"] = "mock"
    settings.raw.setdefault("screener", {})["mode"] = "mock"
    return settings


def _make_fixtures_live(fixtures_dir: Path) -> None:
    """The committed fixtures are a frozen June-2026 snapshot, so their close_time
    is now in the past — which Stage 1's resolved-market filter would (correctly)
    drop. Shift every fixture date forward by a constant so the markets read as
    live as of the run (close just ahead of the latest price point), preserving
    the price-history-to-close spacing that the anomaly scores and the
    time-to-close bonus depend on. This keeps the end-to-end test hermetic and
    independent of the wall-clock date it runs on."""
    target_close = pd.Timestamp.now(tz="UTC").floor("h") + pd.Timedelta(days=1)
    for name in ("markets.csv", "price_history.csv"):
        path = fixtures_dir / name
        df = pd.read_csv(path)
        if "close_time" not in df.columns:
            continue
        offset = target_close - pd.to_datetime(df["close_time"], utc=True).max()
        df["close_time"] = (pd.to_datetime(df["close_time"], utc=True) + offset).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        if "timestamp" in df.columns:
            df["timestamp"] = (pd.to_datetime(df["timestamp"], utc=True) + offset).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        df.to_csv(path, index=False)


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
