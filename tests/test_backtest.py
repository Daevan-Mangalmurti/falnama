"""Phase 1 detection backtest: the replay harness detects a pre-resolution run-up,
ignores a flat control, splits the settlement tail off cleanly, and never lets one
unavailable market sink the run. All network access is mocked → offline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from falnama import backtest
from falnama.config import load_config

S = load_config()
# Pin the detector knobs these tests depend on, independent of committed config.
S.raw["anomaly"]["strong_threshold"] = 85
S.raw["anomaly"]["min_price_observations"] = 20


def _history(prices: list[float], start="2026-02-01T00:00:00Z") -> pd.DataFrame:
    times = pd.date_range(start, periods=len(prices), freq="1h", tz="UTC")
    return pd.DataFrame({"timestamp": times, "price": prices})


def _market(question="Test market?", outcome="YES") -> dict:
    return {"id": "1", "conditionId": "0xabc", "question": question,
            "clobTokenIds": '["tok_yes", "tok_no"]',
            "outcomePrices": '["1", "0"]' if outcome == "YES" else '["0", "1"]',
            "endDate": "2026-02-28T00:00:00Z"}


def _patch(monkeypatch, prices, outcome="YES"):
    """Mock the three network boundaries so run_case is fully offline."""
    monkeypatch.setattr(backtest, "_gamma_event", lambda slug, s: {"slug": slug, "markets": [_market(outcome=outcome)]})
    monkeypatch.setattr(backtest, "_clob_history", lambda tok, a, b, s: _history(prices))
    monkeypatch.setattr(backtest, "_resolution_anchor", lambda m, s: pd.Timestamp("2026-02-28T00:00:00Z"))


def test_pre_settlement_removes_the_terminal_settlement():
    # Flat, then a run-up, then parked at ~0.98 (the outcome going public).
    hist = _history([0.06] * 30 + list(np.linspace(0.06, 0.35, 10)) + [0.98] * 12)
    pre, onset = backtest._pre_settlement(hist, "YES")
    assert onset is not None
    assert (pre["price"] < backtest.SETTLED_HI).all()      # settlement tail cut
    assert len(pre) == 40                                   # only the run-up remains


def test_insider_style_runup_is_detected_strong(monkeypatch):
    # A sharp ~25-cent step before settlement should read strong, with the anomaly
    # triggering BEFORE the settlement (positive lead).
    _patch(monkeypatch, [0.10] * 40 + [0.35] * 12 + [0.98] * 10)
    out = backtest.run_case(backtest.BacktestCase(event="x", label="positive"), S)
    assert out.error == ""
    assert out.detected_strong is True
    assert out.peak_anomaly_score >= 85
    assert out.lead_hours is not None and out.lead_hours > 0


def test_flat_control_is_not_flagged(monkeypatch):
    # A market that barely moves then resolves must not read strong.
    rng = np.random.default_rng(0)
    drift = list(np.clip(0.30 + np.cumsum(rng.normal(0, 0.002, 60)), 0.05, 0.5)) + [0.02] * 10
    _patch(monkeypatch, drift, outcome="NO")
    out = backtest.run_case(backtest.BacktestCase(event="x", label="control"), S)
    assert out.detected_strong is False


def test_missing_history_is_recorded_not_raised(monkeypatch):
    monkeypatch.setattr(backtest, "_gamma_event", lambda slug, s: {"slug": slug, "markets": [_market()]})
    monkeypatch.setattr(backtest, "_resolution_anchor", lambda m, s: pd.Timestamp("2026-02-28T00:00:00Z"))
    monkeypatch.setattr(backtest, "_clob_history", lambda tok, a, b, s: backtest.pd.DataFrame(columns=["timestamp", "price"]))
    out = backtest.run_case(backtest.BacktestCase(event="old", label="positive"), S)
    assert out.detected_strong is None
    assert "no price history" in out.error


def test_one_bad_case_does_not_sink_the_run(monkeypatch):
    # A lookup failure on one case is captured on its row; the rest still score.
    def flaky(slug, s):
        if slug == "boom":
            raise LookupError("no such event")
        return {"slug": slug, "markets": [_market()]}
    monkeypatch.setattr(backtest, "_gamma_event", flaky)
    monkeypatch.setattr(backtest, "_clob_history", lambda tok, a, b, s: _history([0.10] * 40 + [0.35] * 12 + [0.98] * 10))
    monkeypatch.setattr(backtest, "_resolution_anchor", lambda m, s: pd.Timestamp("2026-02-28T00:00:00Z"))

    table, summary = backtest.run_backtest([
        backtest.BacktestCase(event="good", label="positive"),
        backtest.BacktestCase(event="boom", label="positive"),
    ], S)
    assert summary["cases"] == 2 and summary["unavailable"] == 1
    assert summary["positives_recall"]["flagged_strong"] == 1     # the good one still scored


def test_summary_separates_recall_from_false_positive(monkeypatch):
    # Two positives (one detected) and one control (not detected) → recall 0.5, FP 0.
    seqs = {
        "hit": [0.10] * 40 + [0.40] * 12 + [0.98] * 10,   # sharp step -> strong
        "miss": [0.30] * 55 + [0.31] * 15,                                       # quiet
        "ctrl": [0.50] * 55 + [0.49] * 15,                                       # quiet control
    }
    monkeypatch.setattr(backtest, "_gamma_event",
                        lambda slug, s: {"slug": slug, "markets": [_market(outcome="YES")]})
    monkeypatch.setattr(backtest, "_resolution_anchor", lambda m, s: pd.Timestamp("2026-02-28T00:00:00Z"))
    monkeypatch.setattr(backtest, "_clob_history", lambda tok, a, b, s, seqs=seqs: _history(seqs[tok]))

    def gamma(slug, s):
        return {"slug": slug, "markets": [{**_market(), "clobTokenIds": f'["{slug}"]'}]}
    monkeypatch.setattr(backtest, "_gamma_event", gamma)

    table, summary = backtest.run_backtest([
        backtest.BacktestCase(event="hit", label="positive"),
        backtest.BacktestCase(event="miss", label="positive"),
        backtest.BacktestCase(event="ctrl", label="control"),
    ], S)
    assert summary["positives_recall"] == {"scored": 2, "flagged_strong": 1, "rate": 0.5}
    assert summary["controls_false_positive"] == {"scored": 1, "flagged_strong": 0, "rate": 0.0}
