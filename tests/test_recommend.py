"""Stage 4: the recommender enforces anti-ex-post timing, caps size, and keeps
its no-trade bias."""

from __future__ import annotations

import pandas as pd

from falnama.config import load_config
from falnama.recommend import _eligible, evaluate_candidate

S = load_config()
RUN = {"run_id": "20260101T000000Z", "run_time_utc": "2026-01-01T00:00:00Z"}


def _card(created: str, *, eligible: bool = True, direction: str = "down",
          mock_mode: bool | None = None) -> dict:
    # Cards default to the configured run's own mode, so these tests do not care
    # whether the committed config happens to be live or mock today.
    mock = (S.card_mode == "mock") if mock_mode is None else mock_mode
    return {
        "card_id": f"card-{created}", "card_hash": "a" * 64, "created_time_utc": created,
        "mock_mode": mock, "market_name": "m", "source": {"market_id": "m"},
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


# ---------------------------------------------------------------------------
# Mode match: a placeholder card can never justify a live recommendation
# ---------------------------------------------------------------------------
def _with_mode(mode: str, root=None):
    import dataclasses

    s = load_config()
    s.raw["card_mode"] = mode
    # Settings is frozen; a copy rooted at `root` sends every output to a temp dir.
    return dataclasses.replace(s, project_root=root) if root else s


def test_live_run_rejects_a_mock_card():
    # The live bug: Kim Jong Un was "recommended" four times on a mock card.
    decision = evaluate_candidate(_anomaly("2026-06-18T00:00:00Z"),
                                  [_card("2026-06-10T00:00:00Z", mock_mode=True)], _with_mode("live"), **RUN)
    assert decision["kind"] == "rejected" and "only mock cards exist" in decision["reason"]


def test_live_run_uses_the_live_card_beside_a_mock_one():
    cards = [_card("2026-06-09T00:00:00Z", mock_mode=False), _card("2026-06-10T00:00:00Z", mock_mode=True)]
    decision = evaluate_candidate(_anomaly("2026-06-18T00:00:00Z"), cards, _with_mode("live"), **RUN)
    assert decision["kind"] == "recommended" and decision["mock_mode"] is False
    assert decision["card_id"] == "card-2026-06-09T00:00:00Z"  # the newer card was mock, so skipped


def test_mock_run_still_uses_mock_cards():
    decision = evaluate_candidate(_anomaly("2026-06-18T00:00:00Z"),
                                  [_card("2026-06-10T00:00:00Z", mock_mode=True)], _with_mode("mock"), **RUN)
    assert decision["kind"] == "recommended" and decision["mock_mode"] is True


# ---------------------------------------------------------------------------
# Once per signal: the same move is not re-recommended on later runs
# ---------------------------------------------------------------------------
def _run_recommend(tmp_path, settings, run_id: str):
    from falnama.io import RunContext
    from falnama.recommend import recommend

    ctx = RunContext(settings=settings, run_id=run_id, run_time_utc="2026-06-19T00:00:00Z")
    anomalies = pd.DataFrame([_anomaly("2026-06-18T00:00:00Z")])
    return recommend(anomalies, [_card("2026-06-10T00:00:00Z", mock_mode=settings.card_mode == "mock")],
                     settings, ctx)


def test_a_signal_is_recommended_once_then_logged_as_a_repeat(tmp_path):
    live = _with_mode("live", tmp_path)
    first = _run_recommend(tmp_path, live, "20260619T000000Z")
    second = _run_recommend(tmp_path, live, "20260620T000000Z")
    assert len(first.recommended) == 1
    assert second.recommended == []
    assert second.rejected[0]["reason"] == "already recommended in run 20260619T000000Z (same move)"


def test_a_new_move_in_the_same_market_is_a_new_signal(tmp_path):
    from falnama.io import RunContext
    from falnama.recommend import recommend

    live = _with_mode("live", tmp_path)
    _run_recommend(tmp_path, live, "20260619T000000Z")
    ctx = RunContext(settings=live, run_id="20260625T000000Z", run_time_utc="2026-06-25T00:00:00Z")
    later_move = pd.DataFrame([_anomaly("2026-06-24T00:00:00Z")])
    result = recommend(later_move, [_card("2026-06-10T00:00:00Z", mock_mode=False)], live, ctx)
    assert len(result.recommended) == 1


def test_old_mock_recommendations_do_not_block_a_live_one(tmp_path):
    _run_recommend(tmp_path, _with_mode("mock", tmp_path), "20260619T000000Z")
    live = _run_recommend(tmp_path, _with_mode("live", tmp_path), "20260620T000000Z")
    assert len(live.recommended) == 1


def test_signal_key_survives_csv_round_trips():
    from falnama.io import signal_key

    # Live market ids come back from CSV as ints or floats; timestamps vary in form.
    assert signal_key(540843, "2026-10-29T18:00:32Z") == signal_key("540843", "2026-10-29 18:00:32+00:00")
    assert signal_key(540843.0, "2026-10-29T18:00:32Z") == "540843@2026-10-29T18:00:32Z"
    assert signal_key(None, "2026-10-29T18:00:32Z") is None and signal_key("1", "not a time") is None
