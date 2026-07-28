"""Stage 1: market classification and selection behave as intended.

These tests double as readable examples of what `classify_market` does.
"""

from __future__ import annotations

import pandas as pd

from falnama.config import load_config
from falnama.select import classify_market, select_markets

SETTINGS = load_config()
# Pin the volume floor to 0 so these topic-classification tests are independent of
# whatever liquidity floor production config currently sets.
SETTINGS.raw["selector"]["min_total_volume"] = 0


def _market(name: str, **extra) -> dict:
    return {"market_name": name, "question": name, "description": name,
            "event_title": name, "category": "", "tags": "", **extra}


def test_sanctions_plural_is_classified():
    # The classic bug this guards against: 'sanction' must match 'sanctions'.
    result = classify_market(_market("Will the EU impose new sanctions on Russia?"), SETTINGS)
    assert result["primary_topic"] == "sanctions"
    assert result["relevance_score"] >= 60
    assert result["country_or_region"] == "russia_ukraine"


def test_short_keyword_does_not_leak():
    # 'war' must NOT fire on 'warranty' — whole-word matching with optional suffix.
    result = classify_market(_market("Best extended warranty provider of 2026?"), SETTINGS)
    assert result["primary_topic"] != "military_conflict"


def test_boilerplate_according_is_not_diplomacy():
    # Real-data regression: "resolve according to" must NOT match 'accord'.
    # Nearly every Polymarket description contains this boilerplate.
    m = _market("Will Team X win the 2026 championship?",
                description="This market will resolve according to the official result.")
    assert classify_market(m, SETTINGS)["primary_topic"] != "diplomacy_treaty"


def test_hard_reject_topics():
    for noise in ["Will the Lakers win the NBA Finals?", "Will Bitcoin hit $150k?",
                  "Will this movie win an Oscar?"]:
        result = classify_market(_market(noise), SETTINGS)
        assert result["hard_rejected"] is True


def test_corporate_actions_are_in_scope():
    # Added when the screen's first axis became economic_salience: M&A and IPOs
    # move public assets and are insider-prone, so they must reach the screen.
    for name in ["Will OpenAI IPO by December 31 2026?",
                 "Will the DOJ block the Kroger-Albertsons merger?",
                 "Will Microsoft acquire Discord before 2027?"]:
        result = classify_market(_market(name), SETTINGS)
        assert result["primary_topic"] == "corporate_action"
        assert result["relevance_score"] >= 60          # survives Stage 1
        assert result["information_structure"] == "asymmetry_prone"


def test_leadership_departure_is_cabinet_government():
    # Real-data regression: "X out as President" / "steps down" were scored 'other'
    # (20) and silently dropped before reaching the screen.
    for name in ["Milei out as President of Argentina before 2027?",
                 "Mitch McConnell steps down from Senate before his term ends?"]:
        result = classify_market(_market(name), SETTINGS)
        assert result["primary_topic"] == "cabinet_government"
        assert result["relevance_score"] >= 60


def test_bitcoin_market_cap_is_still_hard_rejected():
    # 'market cap' is a corporate_action keyword, but a crypto market must still be
    # hard-rejected — the crypto keyword wins regardless of topic.
    result = classify_market(_market("Will Bitcoin's market cap exceed $3T in 2026?"), SETTINGS)
    assert result["hard_rejected"] is True


def test_selection_keeps_geopolitics_drops_noise():
    markets = pd.DataFrame([
        _market("Will the US strike Iran before August?", market_id="a", volume=1, liquidity=1),
        _market("Will the Lakers win the NBA Finals?", market_id="b", volume=1, liquidity=1),
    ])
    result = select_markets(markets, SETTINGS)
    kept = set(result.relevant["market_id"])
    assert "a" in kept and "b" not in kept
    assert result.diagnostics["selected_count"] == 1


def test_resolved_market_is_dropped_in_live_mode():
    # A market whose close_time has already passed is resolved; in live mode it
    # must not reach the expensive downstream stages (Opus cards, stale scoring).
    markets = pd.DataFrame([
        _market("Will the US strike Iran before August?", market_id="live",
                volume=1, liquidity=1, close_time="2026-12-31T00:00:00Z"),
        _market("Will the US strike Iran before August?", market_id="past",
                volume=1, liquidity=1, close_time="2026-06-30T00:00:00Z"),
    ])
    result = select_markets(markets, SETTINGS, now="2026-07-27T00:00:00Z")
    kept = set(result.relevant["market_id"])
    assert "live" in kept and "past" not in kept
    reason = result.rejected.set_index("market_id").loc["past", "rejection_reason"]
    assert "resolved" in reason


def test_missing_close_time_is_not_treated_as_resolved():
    # We can't tell when a market resolves without a close_time, so we keep it —
    # consistent with the pipeline's "missing data never penalizes" stance.
    markets = pd.DataFrame([_market("Will the US strike Iran before August?",
                                    market_id="noclose", volume=1, liquidity=1)])
    result = select_markets(markets, SETTINGS, now="2026-07-27T00:00:00Z")
    assert "noclose" in set(result.relevant["market_id"])


def test_backtest_mode_keeps_resolved_markets():
    # In historical-backtest mode (closed_only: true) past-close markets are the
    # entire point, so the resolved-market filter must switch off.
    from falnama.config import Settings
    bt = Settings(raw={**SETTINGS.raw, "selector": {**SETTINGS.raw["selector"], "closed_only": True}},
                  project_root=SETTINGS.project_root)
    markets = pd.DataFrame([_market("Will the US strike Iran before August?", market_id="past",
                                     volume=1, liquidity=1, close_time="2026-06-30T00:00:00Z")])
    result = select_markets(markets, bt, now="2026-07-27T00:00:00Z")
    assert "past" in set(result.relevant["market_id"])
