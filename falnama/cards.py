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
  * mock → a deterministic placeholder card (no network, no key). Lets anyone run
           and test the full pipeline for free. IMPLEMENTED here.
  * live → a real card from the LLM (Claude via the `anthropic` SDK).

=== NEXT MILESTONE: realize `_live_card` ===
The live path is the immediate follow-up task. The plan:
  1. Build a prompt from `market_context` (question, metadata, selector
     classification) plus the guardrails (paper-only, no ex-post, allowed asset
     classes, prediction window, minimum expected move).
  2. Call the model in settings.llm["model"] and have it return a `predictions`
     array. To make the output valid BY CONSTRUCTION, define the prediction
     shape as a pydantic model and pass it as a tool / structured-output schema,
     so the model must fill exactly the fields index_card_schema.json requires.
  3. Return that list of raw predictions. Everything downstream — normalization,
     `_build_card`, hashing, the immutable write — is SHARED with the mock path,
     so realizing the milestone is essentially steps 1-2 plus the API call.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from . import io
from .config import Settings
from .io import RunContext
from .schema import validate_or_raise

# The five time-plan checkpoints every prediction must fill (schema-enforced).
_TIME_PLAN_KEYS = ["30m", "1h", "2h", "6h", "12h"]


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
# Generating one card
# ---------------------------------------------------------------------------
def generate_card(market_context: dict[str, Any], settings: Settings,
                  created_time_utc: str | None = None) -> dict[str, Any]:
    """Generate one immutable index card for a market, routing on card_mode.

    Returns a fully-formed, schema-valid, hashed card dict (not yet written).
    """
    created = created_time_utc or io.utc_now()
    mock = settings.card_mode == "mock"
    predictions = _mock_card(market_context, settings) if mock else _live_card(market_context, settings)
    card = _build_card(market_context, predictions, settings, created_time_utc=created, mock_mode=mock)
    validate_or_raise(card, "index_card")  # never write a card that fails its contract
    return card


def _build_card(market_context: dict[str, Any], predictions: list[dict[str, Any]],
                settings: Settings, *, created_time_utc: str, mock_mode: bool,
                justification: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble raw predictions into a complete card: identity, immutability
    flags, provenance, evidence/uncertainty defaults, then stamp the canonical
    hash. Shared by both the mock and live paths so cards are structurally
    identical however they were produced."""
    source = {
        "market_id": market_context.get("market_id") or None,
        "market_slug": market_context.get("market_slug") or None,
        "market_name": str(market_context.get("market_name") or "Unknown market"),
        "market_url": market_context.get("market_url") or None,
        "source_market_file": market_context.get("source_market_file"),
    }
    natural_key = source["market_id"] or source["market_slug"] or _safe_slug(source["market_name"])
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
        "predictions": [_normalize_prediction(p, settings) for p in predictions],
        "evidence_used": [
            "prediction-market question and metadata",
            "market-to-asset sensitivity mapping",
            "public information to be checked before any action",
        ],
        "uncertainty_notes": [
            "MOCK card — not an investment recommendation." if mock_mode
            else "LLM output; requires downstream news-lag and public-information checks."
        ],
        "llm_justification": justification or {
            "summary": ("Mock scenario card generated for pipeline testing." if mock_mode
                        else "LLM-generated pre-trade market-to-asset scenario card."),
            "key_assumptions": [
                "The prediction-market signal contains information not fully in public assets.",
                "The chosen instrument is a liquid enough proxy for the event risk.",
            ],
            "failure_modes": [
                "The prediction-market move reverses.",
                "Public information already explained the move.",
                "The proxy asset is weakly exposed to the event.",
            ],
            "why_not_ex_post": ("The card is timestamped, content-hashed, and never overwritten. "
                                "Execution must reject cards created after a trigger time."),
        },
    }
    card["card_hash"] = canonical_hash(card)
    return card


def _normalize_prediction(raw: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """Coerce a raw prediction into the exact shape index_card_schema.json
    requires: valid enums, a 0-1 confidence, a two-number interval, a complete
    time plan, and a computed trade-eligibility. Shared by mock and live paths so
    both are valid by construction."""
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


def _mock_card(market_context: dict[str, Any], settings: Settings) -> list[dict[str, Any]]:
    """Deterministic placeholder prediction, clearly labeled as mock. Maps the
    market to a broad, liquid proxy by simple keyword rules — a stand-in for the
    live LLM's reasoning that keeps the pipeline runnable with no key."""
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
    return [{
        "asset": asset, "ticker": ticker, "tradable_instrument_name": instrument, "asset_class": cls,
        "expected_direction": "down", "expected_return_12h_bps": move, "confidence": 0.62,
        "confidence_interval_bps": [move - 550, move + 600],
        "reasoning": ("MOCK: maps a geopolitical prediction-market signal to a broad liquid proxy. "
                      "Replace with live LLM reasoning (see the module's NEXT MILESTONE note)."),
    }]


def _live_card(market_context: dict[str, Any], settings: Settings) -> list[dict[str, Any]]:
    """Call the LLM to produce predictions. See the NEXT MILESTONE note above.
    Intentionally not yet implemented — this is the immediate follow-up task."""
    raise NotImplementedError(
        "Live LLM card generation is the next milestone. Set card_mode: mock to run today."
    )


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
    settings = ctx.settings
    if markets is None:
        markets = pd.read_csv(settings.output_dir("relevant_markets") / "relevant_markets_latest.csv")

    subset = _card_markets(markets, settings)
    paths: list[Path] = []
    for record in subset.to_dict(orient="records"):
        record.setdefault("source_market_file", "relevant_markets_latest.csv")
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
