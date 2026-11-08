"""The concentration overlay: an independent red flag that only ever escalates
a THICK market moved by a FEW wallets, and never penalizes missing data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from falnama.anomaly import concentration_overlay, score_market
from falnama.config import load_config
from falnama.polymarket import compute_concentration

S = load_config()
CFG = S.anomaly
# Pin the concentration thresholds so these overlay-logic tests stay valid even
# after production thresholds are recalibrated (a planned tuning task).
CFG["concentration"] = {
    "enabled": True, "thick_min_volume": 100000, "thick_min_liquidity": 20000,
    "extreme_top1_share": 0.30, "extreme_top3_share": 0.55, "extreme_gini": 0.80,
    "red_flag_bonus": 15,
}
BONUS = CFG["concentration"]["red_flag_bonus"]

THICK = {"volume": 200000, "liquidity": 40000}
THIN = {"volume": 10000, "liquidity": 2000}
EXTREME = {"available": True, "top1_share": 0.45, "top3_share": 0.70, "gini": 0.85}
DIFFUSE = {"available": True, "top1_share": 0.05, "top3_share": 0.12, "gini": 0.30}


def test_compute_concentration_math():
    trades = pd.DataFrame({"wallet": ["A", "A", "B", "C"], "size": [40, 30, 20, 10]})
    rec = compute_concentration(trades)
    assert rec["available"] and abs(rec["top1_share"] - 0.7) < 1e-9  # A did 70/100
    assert rec["wallet_count"] == 3 and rec["gini"] > 0


def test_thick_and_extreme_is_a_red_flag():
    overlay = concentration_overlay(THICK, EXTREME, CFG)
    assert overlay["concentration_red_flag"] is True
    assert overlay["concentration_tier"] == "red_flag"
    assert overlay["concentration_bonus"] == BONUS


def test_thin_and_extreme_is_not_flagged():
    overlay = concentration_overlay(THIN, EXTREME, CFG)
    assert overlay["concentration_red_flag"] is False
    assert overlay["concentration_tier"] == "concentrated_thin"
    assert overlay["concentration_bonus"] == 0.0


def test_thick_but_diffuse_is_not_flagged():
    overlay = concentration_overlay(THICK, DIFFUSE, CFG)
    assert overlay["concentration_red_flag"] is False
    assert overlay["concentration_tier"] == "diffuse"


def test_missing_data_never_penalizes():
    overlay = concentration_overlay(THICK, None, CFG)
    assert overlay["concentration_available"] is False
    assert overlay["concentration_tier"] == "unavailable"
    assert overlay["concentration_bonus"] == 0.0


def test_red_flag_promotes_composite_by_exactly_the_bonus():
    # A moderate jump so the base score is below 100 and the +bonus is visible.
    prices = list(0.15 + np.zeros(40)) + list(0.30 + np.zeros(15))
    times = pd.date_range("2026-06-01T00:00:00Z", periods=len(prices), freq="1h")
    series = pd.DataFrame({
        "market_id": "m", "market_name": "m", **THICK,
        "timestamp": [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in times],
        "price": prices, "close_time": None,
    })
    without = score_market(series, S)
    with_flag = score_market(series, S, EXTREME)
    assert with_flag["concentration_red_flag"] is True
    assert with_flag["anomaly_score"] == min(100.0, without["anomaly_score"] + BONUS)


def _live():
    live = load_config()
    live.raw["data"]["source"] = "live"
    return live


def test_live_fetch_sends_the_condition_id_not_the_market_id(monkeypatch):
    """Regression: the trades API answers Gamma's numeric id with an empty list,
    which silently kept this overlay dark on every live run until 2026-09. It
    must also use the v2 route and `condition` (v1 retired 2026-11-02)."""
    import requests

    from falnama import polymarket

    sent = []

    class Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"proxy_wallet": "0xa", "size": 10}, {"proxy_wallet": "0xb", "size": 30}],
                    "pagination": {"limit": 1000, "offset": 0, "has_more": False, "next_cursor": None}}

    monkeypatch.setattr(requests, "get", lambda url, params, **kw: sent.append((url, params)) or Resp())
    monkeypatch.setattr(polymarket.time, "sleep", lambda s: None)
    out = polymarket._fetch_live_concentration(_live(), ["540843", "999"], {"540843": "0xcond"})
    # only the market with a conditionId is asked about, on the v2 route
    assert sent == [("https://data-api.polymarket.com/v2/trades", {"condition": "0xcond", "limit": 1000})]
    assert out["540843"]["available"] and out["540843"]["top1_share"] == 0.75
    assert out["999"] == {"available": False, "reason": "no conditionId for this market"}


def test_a_dead_trades_api_reads_unavailable_and_warns(monkeypatch):
    """The 2026-11 outage: every call answered 410 Gone, every market read
    'unavailable' (correct: never a penalty) — but the run reported success with
    no warning. Now the dark sensor shows up in pipeline_health."""
    import requests

    from falnama import polymarket
    from falnama.io import RunContext

    class Gone:
        status_code = 410

        def raise_for_status(self):
            raise requests.HTTPError("410 Client Error: Gone for url: https://data-api.polymarket.com/trades?x")

    monkeypatch.setattr(requests, "get", lambda url, **kw: Gone())
    monkeypatch.setattr(polymarket.time, "sleep", lambda s: None)
    live = _live()
    ctx = RunContext.start(live)
    out = polymarket.fetch_trade_concentration(live, ["1", "2"], ctx, {"1": "0x1", "2": "0x2"})
    assert all(not r["available"] for r in out.values())
    assert concentration_overlay(THICK, out["1"], CFG)["concentration_bonus"] == 0.0  # fail open, never penalize
    assert len(ctx.warnings) == 1
    assert "wallet concentration available for only 0 of 2" in ctx.warnings[0]
    assert "410 Client Error: Gone" in ctx.warnings[0] and "for url" not in ctx.warnings[0]
