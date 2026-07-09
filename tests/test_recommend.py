"""Stage 4: the recommender enforces anti-ex-post timing, caps size, and keeps
its no-trade bias."""

from __future__ import annotations

import pandas as pd

from falnama.config import load_config
from falnama.recommend import _eligible, evaluate_candidate

S = load_config()
RUN = {"run_id": "20260101T000000Z", "run_time_utc": "2026-01-01T00:00:00Z"}


def _card(created: str, *, eligible: bool = True, direction: str = "down") -> dict:
    return {
        "card_id": f"card-{created}", "card_hash": "a" * 64, "created_time_utc": created,
        "mock_mode": True, "market_name": "m", "source": {"market_id": "m"},
        "predictions": [{
            "asset": "Crude oil", "asset_class": "commodity", "expected_direction": direction,
            "expected_return_12h_bps": -900.0, "confidence": 0.6,
            "trade_eligibility": {"eligible": eligible, "reason": "x", "minimum_expected_move_bps": 700},
        }],
    }


def _anomaly(trigger: str, score: float = 100.0) -> dict:
    return {"market_name": "m", "market_id": "m", "anomaly_score": score,
            "anomaly_class": "strong", "anomaly_trigger_time_utc": trigger}


def test_pre_existing_card_yields_a_live_recommendation():
    decision = evaluate_candidate(_anomaly("2026-06-18T00:00:00Z"),
                                  [_card("2026-06-10T00:00:00Z")], S, **RUN)
    assert decision["kind"] == "recommended"
    assert decision["is_live_timing"] is True and decision["action"] == "sell"
    assert 0 < decision["notional_usd"] <= S.recommend["max_position_usd"]


def test_ex_post_card_is_rejected_by_default():
    # Card created AFTER the trigger, backfill off -> rejected.
    decision = evaluate_candidate(_anomaly("2026-06-18T00:00:00Z"),
                                  [_card("2026-06-20T00:00:00Z")], S, **RUN)
    assert decision["kind"] == "rejected" and "ex-post" in decision["reason"]


def test_backfill_allows_ex_post_but_marks_non_live():
    backfilled = load_config()
    backfilled.raw["backfill_mode"] = True
    decision = evaluate_candidate(_anomaly("2026-06-18T00:00:00Z"),
                                  [_card("2026-06-20T00:00:00Z")], backfilled, **RUN)
    assert decision["kind"] == "recommended" and decision["is_live_timing"] is False


def test_no_card_is_rejected():
    decision = evaluate_candidate(_anomaly("2026-06-18T00:00:00Z"), [], S, **RUN)
    assert decision["kind"] == "rejected" and "no scenario card" in decision["reason"]


def test_ineligible_card_is_rejected():
    decision = evaluate_candidate(_anomaly("2026-06-18T00:00:00Z"),
                                  [_card("2026-06-10T00:00:00Z", eligible=False)], S, **RUN)
    assert decision["kind"] == "rejected" and "ineligible" in decision["reason"]


def test_eligibility_keeps_tied_top_scores():
    anomalies = pd.DataFrame([
        {"market_id": "a", "anomaly_class": "strong", "anomaly_score": 100},
        {"market_id": "b", "anomaly_class": "strong", "anomaly_score": 100},
        {"market_id": "c", "anomaly_class": "medium", "anomaly_score": 72},
    ])
    kept = _eligible(anomalies, S)
    assert set(kept["market_id"]) == {"a", "b"}  # both strong ties kept; medium dropped
