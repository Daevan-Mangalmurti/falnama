"""Phase 1 detection backtest: the replay harness detects a pre-resolution run-up,
ignores a flat control, splits the settlement tail off cleanly, and never lets one
unavailable market sink the run. All network access is mocked → offline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from falnama import backtest, polymarket, wallets
from falnama.config import load_config

S = load_config()
# Pin the detector knobs these tests depend on, independent of committed config.
S.raw["anomaly"]["strong_threshold"] = 85
S.raw["anomaly"]["min_price_observations"] = 20


@pytest.fixture(autouse=True)
def _no_wallet_network(monkeypatch):
    """Every case also runs the wallet sensor; stub its trade fetch so these price
    tests stay offline (an empty window reads as fingerprint tier 'none')."""
    monkeypatch.setattr(wallets, "_fetch_trades", lambda *a, **k: pd.DataFrame(
        columns=["wallet", "wallet_name", "side", "outcome", "price", "size", "timestamp"]))


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


def test_wallet_sensor_runs_beside_the_price_detector(monkeypatch):
    """The fingerprint looks back from the settlement onset, reports its tier on the
    same row, and still runs when the price history is too thin to score."""
    windows = []

    def trades(settings, market, start, end):
        windows.append((start, end))
        rows = [(f"0x{i}", f"w{i}", "BUY", "Yes", 0.10, 200000, "2026-02-01T20:00:00Z") for i in range(3)]
        return pd.DataFrame(rows, columns=["wallet", "wallet_name", "side", "outcome", "price", "size", "timestamp"])

    monkeypatch.setattr(wallets, "_fetch_trades", trades)
    monkeypatch.setattr(wallets, "_fetch_profile", lambda s, w: {
        "first_activity_utc": "2026-01-30T00:00:00Z", "markets_traded": 3})
    _patch(monkeypatch, [0.10] * 40 + [0.35] * 12 + [0.98] * 10)
    evidence: list = []
    out = backtest.run_case(backtest.BacktestCase(event="x", label="positive"), S, evidence)
    assert out.fingerprint_tier == "cluster" and out.fingerprint_matched_wallets == 3
    assert windows[0][1] == pd.Timestamp(out.settlement_time_utc)  # window ends at settlement onset
    assert len(evidence) == 1 and evidence[0]["label"].eq("positive").all()

    _patch(monkeypatch, [0.10] * 5)  # far too little price history to score...
    thin = backtest.run_case(backtest.BacktestCase(event="x", label="positive"), S)
    assert "too few" in thin.error and thin.fingerprint_tier == "cluster"  # ...wallets still judged


def test_resolution_anchor_is_the_last_trade_on_the_v2_feed(monkeypatch):
    sent = []

    def fake(settings, route, params):
        sent.append((route, params))
        return {"data": [{"timestamp": 1772236800}], "pagination": {"next_cursor": None}}

    monkeypatch.setattr(polymarket, "data_api_get", fake)
    anchor = backtest._resolution_anchor({"conditionId": "0xc", "endDate": "2030-01-01T00:00:00Z"}, S)
    assert anchor == pd.Timestamp(1772236800, unit="s", tz="UTC")
    assert sent == [("trades", {"condition": "0xc", "limit": 1})]


def test_resolution_anchor_falls_back_to_end_date(monkeypatch):
    def down(*a, **k):
        raise ConnectionError("down")
    monkeypatch.setattr(polymarket, "data_api_get", down)
    anchor = backtest._resolution_anchor({"conditionId": "0xc", "endDate": "2026-03-01T00:00:00Z"}, S)
    assert anchor == pd.Timestamp("2026-03-01T00:00:00Z")


# ---------------------------------------------------------------------------
# The price archive: frozen hourly histories, read before any live fetch
# ---------------------------------------------------------------------------
ANCHOR = pd.Timestamp("2026-02-28T00:00:00Z")


def _rooted(tmp_path):
    """Settings whose project root (and so whose archive) is a temp dir."""
    import dataclasses
    return dataclasses.replace(S, project_root=tmp_path)


def _archive_offline(monkeypatch, calls):
    """Mock the archive writer's network boundaries; record every CLOB window asked for."""
    monkeypatch.setattr(backtest, "_gamma_event", lambda slug, s: {"slug": slug, "markets": [_market()]})
    monkeypatch.setattr(backtest, "_resolution_anchor", lambda m, s: ANCHOR)

    def clob(tok, a, b, s):
        calls.append((a, b))
        hours = pd.date_range(pd.Timestamp(a, unit="s", tz="UTC").ceil("1h"),
                              pd.Timestamp(b, unit="s", tz="UTC"), freq="1h")
        return pd.DataFrame({"timestamp": hours, "price": 0.2})
    monkeypatch.setattr(backtest, "_clob_history", clob)


def test_archive_freezes_a_wide_band_in_short_pieces(tmp_path, monkeypatch):
    calls = []
    _archive_offline(monkeypatch, calls)
    manifest = backtest.archive_histories([backtest.BacktestCase(event="e", label="positive")], _rooted(tmp_path),
                                          now=pd.Timestamp("2026-11-25T00:00:00Z"))
    entry = manifest["tokens"]["tok_yes"]
    assert entry["window_start_utc"] == "2026-01-31T00:00:00Z" and entry["window_end_utc"] == "2026-03-02T00:00:00Z"
    assert all(b - a <= 7 * 86400 for a, b in calls)          # the route refuses long hourly windows
    assert entry["points"] == 30 * 24 + 1                      # the joins between pieces are not duplicated
    assert manifest["failures"] == []


def test_the_backtest_reads_the_archive_and_never_calls_clob(tmp_path, monkeypatch):
    settings = _rooted(tmp_path)
    _archive_offline(monkeypatch, [])
    backtest.archive_histories([backtest.BacktestCase(event="e", label="positive")], settings,
                               now=pd.Timestamp("2026-11-25T00:00:00Z"))
    monkeypatch.setattr(backtest, "_clob_history", lambda *a: (_ for _ in ()).throw(AssertionError("went live")))
    end = int(ANCHOR.timestamp())
    history, source = backtest._price_history("tok_yes", end - 14 * 86400, end, settings)
    assert source == "archive" and len(history) == 14 * 24 + 1
    assert history["timestamp"].min() >= ANCHOR - pd.Timedelta(days=14)


def test_a_window_the_archive_only_partly_covers_is_read_live(tmp_path, monkeypatch):
    settings = _rooted(tmp_path)
    _archive_offline(monkeypatch, [])
    backtest.archive_histories([backtest.BacktestCase(event="e", label="positive")], settings,
                               now=pd.Timestamp("2026-11-25T00:00:00Z"))
    late = int((ANCHOR + pd.Timedelta(days=10)).timestamp())  # runs past the archived band
    _, source = backtest._price_history("tok_yes", late - 14 * 86400, late, settings)
    assert source == "clob"


def test_archived_markets_are_frozen_unless_refreshed(tmp_path, monkeypatch):
    settings, calls = _rooted(tmp_path), []
    _archive_offline(monkeypatch, calls)
    case = [backtest.BacktestCase(event="e", label="positive")]
    backtest.archive_histories(case, settings, now=pd.Timestamp("2026-11-25T00:00:00Z"))
    first = len(calls)
    backtest.archive_histories(case, settings, now=pd.Timestamp("2026-11-26T00:00:00Z"))
    assert len(calls) == first                                  # nothing re-fetched
    manifest = backtest.archive_histories(case, settings, refresh=True, now=pd.Timestamp("2026-11-26T00:00:00Z"))
    assert len(calls) == 2 * first and manifest["tokens"]["tok_yes"]["archived_at_utc"].startswith("2026-11-26")
    archived, _ = backtest.load_archive(settings)
    assert len(archived) == manifest["tokens"]["tok_yes"]["points"]   # a refresh replaces, never duplicates


def test_a_case_that_fails_to_archive_is_recorded_not_fatal(tmp_path, monkeypatch):
    _archive_offline(monkeypatch, [])
    monkeypatch.setattr(backtest, "_gamma_event",
                        lambda slug, s: (_ for _ in ()).throw(LookupError("gone")) if slug == "bad"
                        else {"slug": slug, "markets": [_market()]})
    manifest = backtest.archive_histories([backtest.BacktestCase(event="bad", label="control"),
                                           backtest.BacktestCase(event="e", label="positive")], _rooted(tmp_path))
    assert list(manifest["tokens"]) == ["tok_yes"]
    assert manifest["failures"][0]["event"] == "bad" and "gone" in manifest["failures"][0]["error"]


def test_the_committed_archive_covers_every_curated_case():
    # Guards the frozen record itself: each case's scored window must be inside it.
    archived, manifest = backtest.load_archive(S)
    assert manifest["failures"] == []
    events = {(e["event"], e["market_contains"]) for e in manifest["tokens"].values()}
    assert {(c.event, c.market_contains) for c in backtest.load_cases(settings=S)} <= events
    for token, entry in manifest["tokens"].items():
        assert (archived["token_id"] == token).sum() == entry["points"]
        anchor = pd.Timestamp(entry["resolution_anchor_utc"])
        assert entry["window_start_ts"] <= int(anchor.timestamp()) - 14 * 86400 <= entry["window_end_ts"]
