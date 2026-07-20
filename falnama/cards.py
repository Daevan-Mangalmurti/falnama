"""Stage 3 — generate immutable "scenario index cards" (the LLM seam).

WHAT:     For a selected market, produce a card mapping the prediction-market
          question to its plausible asset implications (asset, direction,
          expected move, confidence, time plan, evidence, failure modes).
CONSUMES: selected markets (Stage 1) + settings.cards + the LLM seam
PRODUCES: immutable, schema-validated, content-hashed JSON cards under
          outputs/index_cards/, plus a card-generation record in the manifest
REVIEWER: a human checking that a recommendation rests on a sound, pre-existing
          mapping rather than an after-the-fact story
ROLE:     Falnama's defense against EX-POST RATIONALIZATION. A card is written
          BEFORE any anomaly is acted on, is never edited in place, and is
          hashed so tampering is detectable. The recommender (Stage 4) refuses
          any card created after an anomaly's trigger time (unless backfill_mode).

Two generation paths, chosen by `card_mode`:
  * mock → a deterministic placeholder card (no network, no key). Lets anyone
           run and test the full pipeline for free.
  * live → a real card from Claude (the `anthropic` SDK), using STRUCTURED
           OUTPUT: the model fills a pydantic schema, so its analysis is valid
           against index_card_schema.json by construction.

The split of responsibility is deliberate: the LLM produces the ANALYSIS (which
assets move, why, and how it could be wrong); the code produces the IDENTITY and
INTEGRITY (card id, timestamp, immutability flags, canonical hash). Humans and
the LLM own meaning; code owns the audit guarantees.

To change HOW the model reasons, edit `SCENARIO_SYSTEM_PROMPT` and
`_build_user_prompt` below — that is the analytical heart of this stage.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from . import io
from .config import Settings
from .io import RunContext
from .schema import validate_or_raise

# The five time-plan checkpoints every prediction must fill (schema-enforced).
_TIME_PLAN_KEYS = ["30m", "1h", "2h", "6h", "12h"]

# Structural statement, identical on every card: it does not depend on the market,
# so the code authors it rather than the LLM.
_WHY_NOT_EX_POST = (
    "The card is timestamped, content-hashed, and never overwritten. Execution "
    "must reject any card created after an anomaly's trigger time."
)


class CardGenerationError(RuntimeError):
    """Raised when the live LLM card generator cannot produce a usable card."""


def canonical_hash(card: dict[str, Any]) -> str:
    """Return the SHA-256 hash that fingerprints an index card.

    The hash is computed over the card with `card_hash` blanked, using a
    canonical (sorted-key, compact) JSON encoding so the same card always hashes
    the same way. Every validator recomputes this the identical way to detect any
    later edit. This function is the contract — keep it stable.
    """
    payload = json.loads(json.dumps(card, default=str))
    payload["card_hash"] = ""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# The structured shape the LLM fills (valid-by-construction against the schema)
# ---------------------------------------------------------------------------
_ASSET_CLASSES = Literal["equity", "etf", "equity_index", "fx_proxy", "commodity", "other"]
_DIRECTIONS = Literal["up", "down", "mixed", "unclear"]


class LivePrediction(BaseModel):
    """One asset implication of a market event, as the model returns it."""

    asset: str = Field(description="Plain-language asset, e.g. 'Crude oil' or 'China equity risk'.")
    ticker: str | None = Field(default=None, description="Symbol of a liquid instrument, e.g. 'USO'. Null if none.")
    tradable_instrument_name: str | None = Field(default=None, description="The concrete instrument, e.g. 'United States Oil Fund'.")
    asset_class: _ASSET_CLASSES
    expected_direction: _DIRECTIONS
    expected_return_12h_bps: float = Field(description="Expected return over the window in basis points (100 = 1%); sign matches direction.")
    confidence: float = Field(description="Your calibrated probability the mapping is correct, between 0 and 1.")
    confidence_interval_low_bps: float = Field(description="Low end of the plausible return range, in bps.")
    confidence_interval_high_bps: float = Field(description="High end of the plausible return range, in bps.")
    reasoning: str = Field(description="Why this market move implies this asset move.")
    plan_30m: str = Field(description="What to re-check 30 minutes after the signal.")
    plan_1h: str = Field(description="What to re-check at 1 hour.")
    plan_2h: str = Field(description="What to re-check at 2 hours.")
    plan_6h: str = Field(description="What to re-check at 6 hours.")
    plan_12h: str = Field(description="What to re-check at 12 hours (close of the research window).")


class ScenarioAnalysis(BaseModel):
    """The full analytical payload the model produces for one market."""

    predictions: list[LivePrediction] = Field(description="One or two well-supported asset implications.")
    evidence_used: list[str] = Field(description="The inputs your reasoning rests on.")
    uncertainty_notes: list[str] = Field(description="Caveats and what would change your view.")
    summary: str = Field(description="One-paragraph summary of the scenario mapping.")
    key_assumptions: list[str] = Field(description="Assumptions the thesis depends on.")
    failure_modes: list[str] = Field(description="Concrete ways this thesis could be wrong (reversal, already public, weak proxy).")


# ---------------------------------------------------------------------------
# The prompt — the analytical heart of this stage. Edit here to change reasoning.
# ---------------------------------------------------------------------------
SCENARIO_SYSTEM_PROMPT = """\
You are the scenario-analysis engine for Falnama, a paper-only research pipeline \
that studies whether geopolitical prediction markets leak information before it \
is public. Your job: map ONE prediction-market question to its plausible \
downstream PUBLIC-MARKET (asset) implications.

This is a PRE-COMMITMENT written BEFORE any market anomaly is acted on. Your card \
will be timestamped, content-hashed, and never edited. So reason forward — "if \
this market's implied probability moved sharply, what liquid assets would move, \
and why?" — not backward from a headline. Do not invent a convenient story.

For each implication, choose a LIQUID, broadly-traded proxy (an ETF, commodity, \
index, or FX proxy is usually better than a single name). Give the expected \
direction and 12-hour magnitude in basis points, a calibrated confidence in \
[0,1], a plausible interval, concrete reasoning, and a checkpoint plan.

Also record the evidence you used, your key assumptions, and — most important — \
the FAILURE MODES: how this thesis could be wrong (the move reverses, it was \
already public, the proxy is weakly exposed).

Be skeptical and calibrated. The market is a noisy sensor, not truth. If the \
mapping is weak, say so: use 'unclear' direction and low confidence rather than \
manufacturing a signal. Give one or two predictions, not a laundry list. This is \
research, not investment advice.\
"""


def _build_user_prompt(market_context: dict[str, Any], settings: Settings) -> str:
    """Assemble the per-market user message from the market's fields + guardrails."""
    cfg = settings.cards
    lines = ["Map this prediction market to its public-market implications.", ""]
    for label, key in [("Market question", "market_name"), ("Description", "description"),
                       ("Category", "category"), ("Primary topic", "primary_topic"),
                       ("Region", "country_or_region"), ("URL", "market_url")]:
        value = market_context.get(key)
        if value not in (None, "", "nan"):
            lines.append(f"{label}: {value}")
    lines += [
        "",
        f"Prediction window: {cfg.get('prediction_window', '12h')}.",
        f"A prediction is only 'tradable' if the expected move is at least "
        f"{cfg.get('minimum_expected_move_bps', 700)} basis points.",
        f"Allowed asset classes: {', '.join(cfg.get('allowed_asset_classes', []))}.",
        "Confidence must be between 0 and 1. Give one or two predictions.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generating one card
# ---------------------------------------------------------------------------
def generate_card(market_context: dict[str, Any], settings: Settings,
                  created_time_utc: str | None = None) -> dict[str, Any]:
    """Generate one immutable index card for a market, routing on card_mode.

    Returns a fully-formed, schema-valid, hashed card dict (not yet written).
    """
    created = created_time_utc or io.utc_now()
    mock = settings.card_mode == "mock"
    analysis = _mock_card(market_context, settings) if mock else _live_card(market_context, settings)
    card = _build_card(market_context, analysis, settings, created_time_utc=created, mock_mode=mock)
    validate_or_raise(card, "index_card")  # never write a card that fails its contract
    return card


def _build_card(market_context: dict[str, Any], analysis: dict[str, Any], settings: Settings,
                *, created_time_utc: str, mock_mode: bool) -> dict[str, Any]:
    """Assemble an analysis payload into a complete card: identity, immutability
    flags, provenance, then stamp the canonical hash. Shared by mock and live so
    cards are structurally identical however the analysis was produced."""
    source = {
        "market_id": io.clean_id(market_context.get("market_id")),
        "market_slug": io.clean_id(market_context.get("market_slug")),
        "market_name": io.clean_id(market_context.get("market_name")) or "Unknown market",
        "market_url": io.clean_id(market_context.get("market_url")),
        "source_market_file": io.clean_id(market_context.get("source_market_file")),
    }
    natural_key = source["market_id"] or source["market_slug"] or _safe_slug(source["market_name"])
    justification = analysis.get("llm_justification") or {}
    card = {
        "card_id": f"{_safe_slug(natural_key)}_{io.compact_stamp(created_time_utc)}",
        "card_version": "v1-mock" if mock_mode else "v1",
        "card_hash": "",
        "do_not_revise": True,
        "created_time_utc": created_time_utc,
        "mock_mode": mock_mode,
        "source": source,
        "market_name": source["market_name"],
        "prediction_window": str(settings.cards.get("prediction_window", "12h")),
        "predictions": [_normalize_prediction(p, settings) for p in analysis["predictions"]],
        "evidence_used": [str(x) for x in (analysis.get("evidence_used") or _default_evidence())],
        "uncertainty_notes": [str(x) for x in (analysis.get("uncertainty_notes") or _default_uncertainty(mock_mode))],
        "llm_justification": {
            "summary": str(justification.get("summary") or _default_summary(mock_mode)),
            "key_assumptions": [str(x) for x in (justification.get("key_assumptions") or _default_assumptions())],
            "failure_modes": [str(x) for x in (justification.get("failure_modes") or _default_failures())],
            "why_not_ex_post": _WHY_NOT_EX_POST,
        },
    }
    card["card_hash"] = canonical_hash(card)
    return card


def _normalize_prediction(raw: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """Coerce a raw prediction into the exact shape index_card_schema.json
    requires: valid enums, a 0-1 confidence, a two-number interval, a complete
    time plan, and a computed trade-eligibility. A safety net over both paths."""
    cfg = settings.cards
    min_move = float(cfg.get("minimum_expected_move_bps", 700))
    allowed_classes = set(cfg.get("allowed_asset_classes", ["etf", "other"]))

    asset = str(raw.get("asset") or "Unspecified asset")
    ticker = raw.get("ticker")
    ticker = str(ticker).upper() if ticker else None
    instrument = raw.get("tradable_instrument_name") or None

    asset_class = str(raw.get("asset_class") or "other").lower()
    if asset_class not in allowed_classes:
        asset_class = "other"

    direction = str(raw.get("expected_direction") or "unclear").lower()
    direction = {"long": "up", "buy": "up", "short": "down", "sell": "down"}.get(direction, direction)
    if direction not in {"up", "down", "mixed", "unclear"}:
        direction = "unclear"

    expected = _as_float(raw.get("expected_return_12h_bps"), 0.0)
    confidence = max(0.0, min(1.0, _as_float(raw.get("confidence"), 0.0)))

    ci = raw.get("confidence_interval_bps")
    if not isinstance(ci, list) or len(ci) != 2:
        ci = [expected, expected]
    ci = [_as_float(ci[0], expected), _as_float(ci[1], expected)]

    eligibility = raw.get("trade_eligibility") if isinstance(raw.get("trade_eligibility"), dict) else {}
    eligible = bool(eligibility.get("eligible", abs(expected) >= min_move and direction in {"up", "down"}))
    reason = str(eligibility.get("reason") or
                 ("Meets minimum expected move." if eligible else "Below minimum expected move or unclear direction."))

    time_plan = raw.get("time_plan") if isinstance(raw.get("time_plan"), dict) else {}
    time_plan = {k: str(time_plan.get(k) or f"Reassess the thesis at T+{k}.") for k in _TIME_PLAN_KEYS}

    return {
        "asset": asset, "ticker": ticker, "tradable_instrument_name": instrument,
        "asset_class": asset_class, "expected_direction": direction,
        "expected_return_12h_bps": expected, "confidence": confidence,
        "confidence_interval_bps": ci,
        "trade_eligibility": {"eligible": eligible, "reason": reason, "minimum_expected_move_bps": min_move},
        "time_plan": time_plan,
        "reasoning": str(raw.get("reasoning") or "No reasoning supplied."),
    }


def _mock_card(market_context: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """Deterministic placeholder analysis, clearly labeled as mock. Maps the
    market to a broad, liquid proxy by simple keyword rules — a stand-in for the
    live LLM that keeps the pipeline runnable with no key."""
    name = str(market_context.get("market_name", "")).lower()
    if any(t in name for t in ("iran", "oil", "opec", "israel", "middle east", "strait")):
        asset, ticker, instrument, cls = "Crude oil", "USO", "United States Oil Fund", "commodity"
    elif any(t in name for t in ("china", "taiwan")):
        asset, ticker, instrument, cls = "China equity risk proxy", "FXI", "iShares China Large-Cap ETF", "etf"
    elif any(t in name for t in ("russia", "ukraine", "europe", "france", "nato")):
        asset, ticker, instrument, cls = "Europe equity risk proxy", "VGK", "Vanguard FTSE Europe ETF", "etf"
    else:
        asset, ticker, instrument, cls = "Emerging-market risk proxy", "EEM", "iShares MSCI Emerging Markets ETF", "etf"

    min_move = float(settings.cards.get("minimum_expected_move_bps", 700))
    move = -max(min_move + 150.0, 850.0)  # a downside risk move that clears the threshold
    return {"predictions": [{
        "asset": asset, "ticker": ticker, "tradable_instrument_name": instrument, "asset_class": cls,
        "expected_direction": "down", "expected_return_12h_bps": move, "confidence": 0.62,
        "confidence_interval_bps": [move - 550, move + 600],
        "reasoning": ("MOCK: maps a geopolitical prediction-market signal to a broad liquid proxy. "
                      "Replace with live LLM reasoning by setting card_mode: live."),
    }]}


def _live_card(market_context: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """Ask Claude to produce the scenario analysis via structured output, then
    convert it to the shared analysis dict for `_build_card`."""
    system = SCENARIO_SYSTEM_PROMPT
    user = _build_user_prompt(market_context, settings)
    analysis = _call_scenario_llm(system, user, settings)
    return _analysis_to_dict(analysis)


def _call_scenario_llm(system: str, user: str, settings: Settings) -> ScenarioAnalysis:
    """The single network boundary. Isolated so tests can mock it and keep CI
    offline. Uses messages.parse() so the response is a validated pydantic object."""
    import anthropic  # imported lazily so the mock path needs no SDK / key

    client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY from the environment
    model = str(settings.llm.get("model", "claude-opus-4-8"))
    timeout = int(settings.llm.get("timeout_seconds", 60))
    try:
        response = client.with_options(timeout=timeout).messages.parse(
            model=model,
            max_tokens=8000,
            thinking={"type": "adaptive"},  # let the model reason before committing
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=ScenarioAnalysis,
        )
    except anthropic.AnthropicError as exc:  # auth, rate limit, network, etc.
        raise CardGenerationError(
            f"Scenario LLM call failed ({type(exc).__name__}): {exc}. "
            "Check that ANTHROPIC_API_KEY is set and the anthropic SDK is current."
        ) from exc
    if response.parsed_output is None:
        raise CardGenerationError("Scenario LLM returned no parseable structured output.")
    return response.parsed_output


def _analysis_to_dict(analysis: ScenarioAnalysis) -> dict[str, Any]:
    """Convert the parsed pydantic analysis into the raw dict `_build_card` consumes."""
    predictions = [{
        "asset": p.asset, "ticker": p.ticker, "tradable_instrument_name": p.tradable_instrument_name,
        "asset_class": p.asset_class, "expected_direction": p.expected_direction,
        "expected_return_12h_bps": p.expected_return_12h_bps, "confidence": p.confidence,
        "confidence_interval_bps": [p.confidence_interval_low_bps, p.confidence_interval_high_bps],
        "reasoning": p.reasoning,
        "time_plan": {"30m": p.plan_30m, "1h": p.plan_1h, "2h": p.plan_2h, "6h": p.plan_6h, "12h": p.plan_12h},
    } for p in analysis.predictions]
    return {
        "predictions": predictions,
        "evidence_used": list(analysis.evidence_used),
        "uncertainty_notes": list(analysis.uncertainty_notes),
        "llm_justification": {
            "summary": analysis.summary,
            "key_assumptions": list(analysis.key_assumptions),
            "failure_modes": list(analysis.failure_modes),
        },
    }


# ---------------------------------------------------------------------------
# Default card-level content (used by the mock path, or as a live-path fallback)
# ---------------------------------------------------------------------------
def _default_evidence() -> list[str]:
    return ["prediction-market question and metadata", "market-to-asset sensitivity mapping",
            "public information to be checked before any action"]


def _default_uncertainty(mock_mode: bool) -> list[str]:
    return ["MOCK card — not an investment recommendation."] if mock_mode else \
           ["LLM output; requires downstream news-lag and public-information checks."]


def _default_summary(mock_mode: bool) -> str:
    return ("Mock scenario card generated for pipeline testing." if mock_mode
            else "LLM-generated pre-trade market-to-asset scenario card.")


def _default_assumptions() -> list[str]:
    return ["The prediction-market signal contains information not fully in public assets.",
            "The chosen instrument is a liquid enough proxy for the event risk."]


def _default_failures() -> list[str]:
    return ["The prediction-market move reverses.", "Public information already explained the move.",
            "The proxy asset is weakly exposed to the event."]


# ---------------------------------------------------------------------------
# Writing cards immutably
# ---------------------------------------------------------------------------
def write_card(settings: Settings, card: dict[str, Any]) -> Path:
    """Write a card to outputs/index_cards/<card_id>.json. Immutable: if the file
    already exists it is NOT overwritten (regeneration produces a new card_id, so
    this only guards against an exact-identity rewrite)."""
    path = settings.output_dir("index_cards") / f"{card['card_id']}.json"
    if not path.exists():
        path.write_text(json.dumps(card, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def load_cards(settings: Settings) -> list[dict[str, Any]]:
    """Read every card currently on disk (across all runs). Stage 4 uses these to
    find a pre-existing card for an anomaly."""
    cards = []
    for path in sorted(settings.output_dir("index_cards").glob("*.json")):
        try:
            cards.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return cards


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run(ctx: RunContext, markets: pd.DataFrame | None = None) -> list[Path]:
    """Generate and write immutable cards for the configured subset of markets,
    then record the batch in the manifest. Returns the written card paths."""
    from . import screen

    settings = ctx.settings
    # The screened universe when the LLM relevance gate ran, else Stage 1's. Cards
    # are the most expensive artifact per market, so this is where the screen pays.
    universe = screen.universe_path(settings)
    if markets is None:
        markets = io.read_table(universe)

    subset = _card_markets(markets, settings)
    paths: list[Path] = []
    for record in subset.to_dict(orient="records"):
        record.setdefault("source_market_file", universe.name)
        card = generate_card(record, settings, created_time_utc=ctx.run_time_utc)
        paths.append(write_card(settings, card))

    io.update_manifest(ctx, "cards", {
        "card_mode": settings.card_mode,
        "cards_written": len(paths),
        "files": [str(p.relative_to(settings.project_root)) for p in paths],
    })
    return paths


def _card_markets(markets: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Pick which selected markets get a card, per cards.generate_for."""
    cfg = settings.cards
    mode = str(cfg.get("generate_for", "all_markets"))
    if markets.empty or mode == "all_markets":
        return markets
    if mode == "top_n":
        return markets.sort_values("relevance_score", ascending=False).head(int(cfg.get("top_n", 20)))
    if mode == "manual_ids":
        wanted = {str(m) for m in cfg.get("manual_ids", [])}
        return markets[markets["market_id"].astype(str).isin(wanted)]
    return markets


def _safe_slug(value: Any, fallback: str = "market") -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value or fallback).strip().lower()).strip("-")
    return text[:100] or fallback


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default
