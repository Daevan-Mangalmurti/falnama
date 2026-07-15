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
