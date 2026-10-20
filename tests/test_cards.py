"""Stage 3: mock cards are schema-valid, hashed, immutable, and deterministic."""

from __future__ import annotations

from falnama import cards
from falnama.config import load_config
from falnama.schema import validate

S = load_config()
S.raw["card_mode"] = "mock"  # these tests exercise the mock path regardless of committed config


def _context(name: str) -> dict:
    return {"market_id": "x1", "market_slug": "x1", "market_name": name, "market_url": None}


def test_mock_card_is_valid_and_self_hashed():
    card = cards.generate_card(_context("Will the US strike Iran?"), S)
    assert validate(card, "index_card") == []          # matches the schema exactly
    assert card["mock_mode"] is True and card["do_not_revise"] is True
    # The stored hash is the canonical hash of the card.
    assert card["card_hash"] == cards.canonical_hash(card)


def test_hash_detects_tampering():
    card = cards.generate_card(_context("Will the US strike Iran?"), S)
    tampered = {**card, "market_name": "Something else entirely"}
    assert cards.canonical_hash(tampered) != card["card_hash"]


def test_numeric_market_id_produces_valid_card():
    # Live Polymarket ids are numeric and arrive as ints after a CSV round-trip;
    # the card must still satisfy the schema (source.market_id is string|null).
    ctx = {"market_id": 540843, "market_slug": "will-x-happen",
           "market_name": "Will X happen?", "market_url": None}
    card = cards.generate_card(ctx, S)
    assert validate(card, "index_card") == []
    assert card["source"]["market_id"] == "540843"


def test_mock_mapping_is_deterministic():
    iran = cards.generate_card(_context("US strike on Iran / oil?"), S)["predictions"][0]
    china = cards.generate_card(_context("China blockade of Taiwan?"), S)["predictions"][0]
    assert iran["asset_class"] == "commodity" and iran["ticker"] == "USO"
    assert china["ticker"] == "FXI"


def test_write_is_immutable(tmp_path, monkeypatch):
    # Point the index_cards output at a temp dir so the test is self-contained.
    monkeypatch.setattr(type(S), "output_dir", lambda self, key: tmp_path)
    card = cards.generate_card(_context("Will the US strike Iran?"), S)
    path = cards.write_card(S, card)
    original = path.read_text()
    cards.write_card(S, {**card, "market_name": "TAMPERED"})  # same card_id
    assert path.read_text() == original                       # not overwritten


def test_live_path_builds_a_valid_card(monkeypatch):
    # Exercise the live-LLM path WITHOUT the network by mocking the one API
    # boundary (_call_scenario_llm). Verifies the structured output -> card
    # plumbing: normalization, live labeling, schema validity, and hashing.
    fake = cards.ScenarioAnalysis(
        predictions=[cards.LivePrediction(
            asset="Crude oil", ticker="uso", tradable_instrument_name="United States Oil Fund",
            asset_class="commodity", expected_direction="down", expected_return_12h_bps=-900.0,
            confidence=0.6, confidence_interval_low_bps=-1400.0, confidence_interval_high_bps=-300.0,
            reasoning="A strike would disrupt supply.",
            plan_30m="a", plan_1h="b", plan_2h="c", plan_6h="d", plan_12h="e")],
        evidence_used=["market metadata"], uncertainty_notes=["may already be public"],
        summary="Iran strike risk maps to crude oil.",
        key_assumptions=["supply disruption"], failure_modes=["the move reverses"])
    monkeypatch.setattr(cards, "_call_scenario_llm", lambda *a, **k: fake)

    live = load_config()
    live.raw["card_mode"] = "live"
    card = cards.generate_card(_context("Will the US strike Iran?"), live)

    assert validate(card, "index_card") == []
    assert card["mock_mode"] is False and card["card_version"] == "v1"
    assert card["predictions"][0]["ticker"] == "USO"  # normalized to upper-case
    assert card["llm_justification"]["failure_modes"] == ["the move reverses"]
    assert card["card_hash"] == cards.canonical_hash(card)


# ---------------------------------------------------------------------------
# Reuse: a still-valid card covers its market instead of a new one being written
# ---------------------------------------------------------------------------
NOW = "2026-10-20T12:00:00Z"


def _settings(**cards_cfg):
    s = load_config()
    s.raw["card_mode"] = "mock"
    s.raw["cards"] = {**s.raw.get("cards", {}), "generate_for": "all_markets",
                      "reuse_existing": True, "reuse_max_age_days": 7, **cards_cfg}
    return s


def _markets(*rows):
    import pandas as pd
    return pd.DataFrame([{"market_id": mid, "market_slug": mid, "market_name": name,
                          "market_url": None, "relevance_score": score} for mid, name, score in rows])


def _card_at(settings, market_id, name, created):
    ctx = {"market_id": market_id, "market_slug": market_id, "market_name": name, "market_url": None}
    return cards.generate_card(ctx, settings, created_time_utc=created)


def test_fresh_card_is_reused_not_regenerated():
    s = _settings()
    old = _card_at(s, "m1", "Will X happen?", "2026-10-18T12:00:00Z")
    plan = cards.plan_cards(_markets(("m1", "Will X happen?", 80)), [old], s, now=NOW)
    assert plan.to_generate == [] and [c["card_id"] for c in plan.reused] == [old["card_id"]]


def test_stale_renamed_or_future_cards_are_not_reused():
    s = _settings()
    stale = _card_at(s, "m1", "Will X happen?", "2026-10-12T12:00:00Z")    # 8 days old
    renamed = _card_at(s, "m2", "Old wording?", "2026-10-19T12:00:00Z")   # question changed
    future = _card_at(s, "m3", "Will Z happen?", "2026-10-21T12:00:00Z")  # after the run
    markets = _markets(("m1", "Will X happen?", 80), ("m2", "New wording?", 80), ("m3", "Will Z happen?", 80))
    plan = cards.plan_cards(markets, [stale, renamed, future], s, now=NOW)
    assert plan.reused == [] and len(plan.to_generate) == 3


def test_mock_card_never_stands_in_for_a_live_card():
    s = _settings()
    mock_card = _card_at(s, "m1", "Will X happen?", "2026-10-19T12:00:00Z")
    s.raw["card_mode"] = "live"
    plan = cards.plan_cards(_markets(("m1", "Will X happen?", 80)), [mock_card], s, now=NOW)
    assert plan.reused == [] and len(plan.to_generate) == 1


def test_tampered_card_is_regenerated_and_reported():
    s = _settings()
    card = _card_at(s, "m1", "Will X happen?", "2026-10-19T12:00:00Z")
    card["predictions"][0]["confidence"] = 0.99  # edited after the hash was stamped
    plan = cards.plan_cards(_markets(("m1", "Will X happen?", 80)), [card], s, now=NOW)
    assert plan.reused == [] and len(plan.to_generate) == 1
    assert plan.hash_mismatches == [card["card_id"]]


def test_top_n_caps_only_new_cards():
    # The top market is already covered, so the one new card goes to the
    # highest-relevance market WITHOUT a card — coverage grows at a fixed cost.
    s = _settings(generate_for="top_n", top_n=1)
    covered = _card_at(s, "a", "A?", "2026-10-19T12:00:00Z")
    markets = _markets(("a", "A?", 90), ("b", "B?", 80), ("c", "C?", 70))
    plan = cards.plan_cards(markets, [covered], s, now=NOW)
    assert [c["source"]["market_id"] for c in plan.reused] == ["a"]
    assert [r["market_id"] for r in plan.to_generate] == ["b"]
