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
