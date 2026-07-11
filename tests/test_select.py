"""Stage 1: market classification and selection behave as intended.

These tests double as readable examples of what `classify_market` does.
"""

from __future__ import annotations

import pandas as pd

from falnama.config import load_config
from falnama.select import classify_market, select_markets

SETTINGS = load_config()


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


def test_selection_keeps_geopolitics_drops_noise():
    markets = pd.DataFrame([
        _market("Will the US strike Iran before August?", market_id="a", volume=1, liquidity=1),
        _market("Will the Lakers win the NBA Finals?", market_id="b", volume=1, liquidity=1),
    ])
    result = select_markets(markets, SETTINGS)
    kept = set(result.relevant["market_id"])
    assert "a" in kept and "b" not in kept
    assert result.diagnostics["selected_count"] == 1
