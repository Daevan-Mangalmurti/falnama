"""Stage 1.5: the LLM relevance gate drops what keywords cannot, keeps every
verdict auditable, and fails OPEN when the model is unavailable."""

from __future__ import annotations

import pandas as pd
import pytest

from falnama import screen
from falnama.config import load_config

S = load_config()
# Pin the knobs these tests depend on, so they stay valid after the production
# thresholds are retuned (which is the whole point of shipping this stage).
S.raw["screener"] = {
    "enabled": True, "mode": "mock", "batch_size": 25,
    "min_geopolitical_relevance": 60, "min_information_asymmetry": 45,
}

# Three markets that stand for the three outcomes we care about: the one we want,
# the one that is not geopolitics, and the one nobody can know early.
MARKETS = pd.DataFrame([
    {"market_id": "1", "market_name": "Will the US impose new sanctions on Iran?", "primary_topic": "sanctions"},
    {"market_id": "2", "market_name": "Will Harvey Weinstein be sentenced to more than 30 years in prison?", "primary_topic": "elections"},
    {"market_id": "3", "market_name": "Will Gavin Newsom win the 2028 US Presidential Election?", "primary_topic": "elections"},
])


def _verdicts(result) -> dict[str, str]:
    return dict(zip(result.verdicts["market_id"], result.verdicts["screen_verdict"]))


def test_mock_screen_keeps_geopolitics_and_drops_the_two_failure_modes():
    result = screen.screen_markets(MARKETS, S)
    assert _verdicts(result) == {"1": "keep", "2": "drop", "3": "drop"}
    assert list(result.kept["market_id"]) == ["1"]


def test_the_two_floors_fail_for_different_reasons():
    # This is the case for two axes rather than one score: the sentencing market
    # is dropped as off-topic, the primary as unknowable-in-advance.
    result = screen.screen_markets(MARKETS, S)
    reasons = dict(zip(result.verdicts["market_id"], result.verdicts["screen_drop_reason"]))
    assert "geopolitical" in reasons["2"] and "asymmetry" not in reasons["2"]
    assert "asymmetry" in reasons["3"] and "geopolitical" not in reasons["3"]


def test_every_market_appears_in_the_verdict_table():
    # Dropped markets must stay auditable — the verdict file is the calibration
    # record, so it carries the rejects, not just the survivors.
    result = screen.screen_markets(MARKETS, S)
    assert len(result.verdicts) == len(MARKETS)
    assert result.verdicts["screen_rationale"].str.len().gt(0).all()


def test_live_path_maps_verdicts_back_by_index(monkeypatch):
    # Exercise the live path WITHOUT the network by mocking the one API boundary.
    live = load_config()
    live.raw["screener"] = {**S.raw["screener"], "mode": "live"}
    fake = screen.ScreenBatch(verdicts=[
        # Deliberately out of order: mapping is by echoed index, not arrival order.
        screen.MarketVerdict(index=2, geopolitical_relevance=70, information_asymmetry=10,
                             corrected_topic="elections", rationale="decided by voters"),
        screen.MarketVerdict(index=0, geopolitical_relevance=95, information_asymmetry=90,
                             corrected_topic="sanctions", rationale="a committee knows first"),
        screen.MarketVerdict(index=1, geopolitical_relevance=5, information_asymmetry=60,
                             corrected_topic="legal_judicial", rationale="not geopolitics"),
    ])
    monkeypatch.setattr(screen, "_call_screen_llm", lambda *a, **k: fake)

    result = screen.screen_markets(MARKETS, live)
    assert _verdicts(result) == {"1": "keep", "2": "drop", "3": "drop"}
    assert list(result.verdicts["screen_topic"]) == ["sanctions", "legal_judicial", "elections"]
    assert not result.diagnostics["error"]


def test_screen_fails_open_when_the_model_errors(monkeypatch):
    # A broken filter must never silently empty the universe: keep everything,
    # and say loudly in the diagnostics why nothing was judged.
    live = load_config()
    live.raw["screener"] = {**S.raw["screener"], "mode": "live"}

    def boom(*args, **kwargs):
        raise RuntimeError("no API key")

    monkeypatch.setattr(screen, "_call_screen_llm", boom)
    result = screen.screen_markets(MARKETS, live)

    assert len(result.kept) == len(MARKETS)          # nothing lost
    assert "no API key" in result.diagnostics["error"]
    assert result.verdicts["screen_rationale"].str.contains("unavailable").all()


def test_short_batch_from_the_model_also_fails_open(monkeypatch):
    # The model returning fewer verdicts than markets is a real failure mode; it
    # must not silently drop the markets it forgot to judge.
    live = load_config()
    live.raw["screener"] = {**S.raw["screener"], "mode": "live"}
    short = screen.ScreenBatch(verdicts=[
        screen.MarketVerdict(index=0, geopolitical_relevance=95, information_asymmetry=90,
                             corrected_topic="sanctions", rationale="only one verdict returned"),
    ])
    monkeypatch.setattr(screen, "_call_screen_llm", lambda *a, **k: short)

    result = screen.screen_markets(MARKETS, live)
    assert len(result.kept) == len(MARKETS)
    assert "1 usable verdicts" in result.diagnostics["error"]


def test_disabled_screen_is_a_pass_through(tmp_path, monkeypatch):
    from falnama.io import RunContext

    off = load_config()
    off.raw["screener"] = {**S.raw["screener"], "enabled": False}
    monkeypatch.setattr(type(off), "output_dir", lambda self, key: tmp_path)
    ctx = RunContext.start(off)

    result = screen.run(ctx, MARKETS)
    assert len(result.kept) == len(MARKETS)
    assert result.diagnostics == {"enabled": False}


def test_universe_path_switches_only_when_screening_ran(tmp_path, monkeypatch):
    settings = load_config()
    monkeypatch.setattr(type(settings), "output_dir", lambda self, key: tmp_path)

    settings.raw["screener"] = {**S.raw["screener"], "enabled": False}
    assert screen.universe_path(settings).name == "relevant_markets_latest.csv"

    # Enabled but no screened file yet (e.g. the stage was skipped) → fall back
    # rather than pointing downstream stages at a file that does not exist.
    settings.raw["screener"] = {**S.raw["screener"], "enabled": True}
    assert screen.universe_path(settings).name == "relevant_markets_latest.csv"

    (tmp_path / "screened_markets_latest.csv").write_text("market_id\n1\n", encoding="utf-8")
    assert screen.universe_path(settings).name == "screened_markets_latest.csv"


def test_empty_universe_is_handled():
    result = screen.screen_markets(pd.DataFrame(), S)
    assert result.kept.empty and result.diagnostics["screened"] == 0


@pytest.mark.parametrize("value,expected", [(150, 100.0), (-5, 0.0), ("abc", 100.0), (None, 100.0)])
def test_scores_are_clamped_and_unreadable_ones_are_permissive(value, expected):
    assert screen._clamp(value) == expected
