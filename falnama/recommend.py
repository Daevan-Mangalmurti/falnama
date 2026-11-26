"""Stage 4 — turn eligible anomalies + index cards into PAPER recommendations.

WHAT:     Matches strong anomalies to their pre-existing index cards, applies
          eligibility and timing rules, sizes paper positions, and writes the
          recommendations (and every rejection) as auditable artifacts.
CONSUMES: strong anomalies (Stage 2) + index cards (Stage 3) + settings.recommend
PRODUCES: outputs/recommended_trades/ (an .xlsx workbook + JSON), and
          outputs/rejected_signals/ (every rejected candidate, with the reason)
REVIEWER: a human deciding whether a paper recommendation is worth acting on
ROLE:     the decision gate, with a deliberate NO-TRADE BIAS. The desired failure
          mode is rejecting too many weak signals, not emitting fragile ones.
          Producing zero recommendations and only rejections is a valid outcome.

Critical timing rule (anti-ex-post): a candidate needs a card that existed
BEFORE the anomaly's trigger time — you may not invent a justifying mapping after
seeing the move. If the only cards for a market were created after the trigger,
the candidate is rejected as ex-post, UNLESS backfill_mode is set (in which case
the recommendation is produced but flagged non-live).

Two more rules keep the paper record honest:
  * Mode match: a live run only accepts live cards; a mock placeholder card can
    never justify a live recommendation.
  * Once per signal: a move (market + trigger time) is recommended at most once.
    The anomaly stage keeps re-reporting a move for as long as it sits in the
    price-history window; later sightings go to the rejection log as repeats.
    The top-half eligibility cut is taken BEFORE repeats are removed, so an
    older strong move still sets the bar a new one must clear.

Position size blends the card's confidence, expected magnitude, and the anomaly
strength — each capped so no single factor dominates — then scales to
recommend.max_position_usd.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import cards as cards_module
from . import io
from .config import Settings
from .io import RunContext
from .schema import validate_or_raise


@dataclass
class RecommendationResult:
    """Outcome of Stage 4: the paper recommendations, the rejected candidates
    (with reasons), and the path to the human-readable workbook."""

    recommended: list[dict]
    rejected: list[dict]
    workbook_path: Path | None


# ---------------------------------------------------------------------------
# Evaluating one candidate (pure, unit-testable)
# ---------------------------------------------------------------------------
def evaluate_candidate(anomaly: dict, cards_for_market: list[dict], settings: Settings,
                       *, run_id: str, run_time_utc: str) -> dict:
    """Decide one strong anomaly. Returns either a recommendation dict (with
    kind='recommended') or a rejection dict (kind='rejected', plus a reason).

    Pure: given the same inputs it always returns the same decision, which is
    what makes the pipeline auditable.
    """
    market_name = io.clean_id(anomaly.get("market_name")) or "unknown market"
    market_id = io.clean_id(anomaly.get("market_id"))  # numeric on live data; keep it a string
    trigger = pd.to_datetime(anomaly.get("anomaly_trigger_time_utc"), utc=True, errors="coerce")

    reject = {"kind": "rejected", "market_name": market_name, "market_id": market_id,
              "anomaly_score": anomaly.get("anomaly_score"),
              "anomaly_trigger_time_utc": anomaly.get("anomaly_trigger_time_utc")}

    # Only cards of the run's own mode count. A mock card is a placeholder: in a
    # live run it must never justify a recommendation (and a live card is not
    # what a mock run is exercising). Same rule as card reuse in cards.py.
    mock_run = settings.card_mode == "mock"
    same_mode = [c for c in cards_for_market if bool(c.get("mock_mode")) == mock_run]
    card, is_live = _pick_card(same_mode, trigger, settings.backfill_mode)
    if card is None:
        if not cards_for_market:
            reason = "no scenario card for this market"
        elif not same_mode:
            reason = (f"only {'live' if mock_run else 'mock'} cards exist; "
                      f"a {settings.card_mode} run needs a {settings.card_mode} card")
        else:
            reason = "only ex-post cards exist (created after the trigger)"
        return {**reject, "reason": reason}

    prediction = card["predictions"][0]  # the primary asset implication
    if not prediction["trade_eligibility"]["eligible"]:
        return {**reject, "card_id": card["card_id"],
                "reason": f"card marks the asset ineligible: {prediction['trade_eligibility']['reason']}"}

    position_score, notional = _size_position(prediction, float(anomaly.get("anomaly_score", 0)), settings)
    direction = prediction["expected_direction"]
    action = {"up": "buy", "down": "sell"}.get(direction, "no_trade")
    if action == "no_trade":
        return {**reject, "card_id": card["card_id"], "reason": f"unclear direction ({direction})"}

    return {
        "kind": "recommended",
        "run_id": run_id, "run_time_utc": run_time_utc,
        "market_name": market_name, "market_id": market_id,
        "card_id": card["card_id"], "card_hash": card["card_hash"],
        "anomaly_trigger_time_utc": anomaly.get("anomaly_trigger_time_utc"),
        "asset": prediction["asset"], "asset_class": prediction["asset_class"],
        "expected_direction": direction, "expected_return_bps": prediction["expected_return_12h_bps"],
        "confidence": prediction["confidence"], "anomaly_score": float(anomaly.get("anomaly_score", 0)),
        "position_score": position_score, "notional_usd": notional, "action": action,
        "is_paper": True, "mock_mode": bool(card.get("mock_mode", False)),
        # Not part of the schema, but useful context carried into the workbook:
        "is_live_timing": is_live,
        "concentration_red_flag": bool(anomaly.get("concentration_red_flag", False)),
    }


def _pick_card(cards_for_market: list[dict], trigger, backfill: bool) -> tuple[dict | None, bool]:
    """Choose the card that justifies acting on an anomaly at `trigger`.

    Prefers the most recent card created at or before the trigger (a genuine
    pre-existing hypothesis → live timing). If none qualifies, returns the latest
    card only when backfill_mode is on (→ non-live), else nothing.
    """
    if not cards_for_market:
        return None, False
    dated = sorted(cards_for_market, key=lambda c: c.get("created_time_utc", ""))
    pre_existing = [c for c in dated
                    if pd.to_datetime(c.get("created_time_utc"), utc=True, errors="coerce") <= trigger]
    if pre_existing:
        return pre_existing[-1], True
    if backfill:
        return dated[-1], False
    return None, False


def _size_position(prediction: dict, anomaly_score: float, settings: Settings) -> tuple[float, float]:
    """Blend confidence, expected magnitude, and anomaly strength (each capped)
    into a 0-1 position score, then scale to the max paper position. Returns
    (position_score, notional_usd)."""
    cfg = settings.recommend
    weights = cfg.get("score_weights", {})
    caps = cfg.get("score_caps", {})

    conf = min(prediction["confidence"], float(caps.get("confidence", 1.0))) / float(caps.get("confidence", 1.0))
    mag = min(abs(prediction["expected_return_12h_bps"]), float(caps.get("expected_magnitude_bps", 2000))) / float(caps.get("expected_magnitude_bps", 2000))
    strength = min(anomaly_score, float(caps.get("anomaly_strength", 100))) / float(caps.get("anomaly_strength", 100))

    w = {"confidence": float(weights.get("confidence", 1.0)),
         "expected_magnitude": float(weights.get("expected_magnitude", 1.0)),
         "anomaly_strength": float(weights.get("anomaly_strength", 1.0))}
    total_w = sum(w.values()) or 1.0
    score = (w["confidence"] * conf + w["expected_magnitude"] * mag + w["anomaly_strength"] * strength) / total_w

    notional = round(score * float(cfg.get("max_position_usd", 10000)), 2)
    return round(score, 4), notional


# ---------------------------------------------------------------------------
# Evaluating the whole batch
# ---------------------------------------------------------------------------
def recommend(anomalies: pd.DataFrame, cards: list[dict], settings: Settings,
              ctx: RunContext, prior: dict[str, str] | None = None) -> RecommendationResult:
    """Evaluate every eligible anomaly against its cards; split into recommended
    and rejected; write the workbook and rejection log.

    `prior` maps signal keys (io.signal_key) to the run that already recommended
    them; it is read from earlier runs' recommendation files when not supplied.
    A signal is recommended at most ONCE: the anomaly stage re-reports the same
    move every run for up to two weeks, and re-recommending it each time would
    stack duplicate paper positions on one piece of evidence.
    """
    eligible = _eligible(anomalies, settings)
    by_market = _cards_by_market(cards)
    prior = already_recommended(settings) if prior is None else prior

    recommended, rejected = [], []
    for anomaly in eligible.to_dict(orient="records"):
        key = io.signal_key(anomaly.get("market_id"), anomaly.get("anomaly_trigger_time_utc"))
        if key in prior:
            # Recorded, not silently dropped: the rejection log shows the repeat.
            rejected.append({"kind": "rejected", "market_name": io.clean_id(anomaly.get("market_name")),
                             "market_id": io.clean_id(anomaly.get("market_id")),
                             "anomaly_score": anomaly.get("anomaly_score"),
                             "anomaly_trigger_time_utc": anomaly.get("anomaly_trigger_time_utc"),
                             "reason": f"already recommended in run {prior[key]} (same move)"})
            continue
        market_cards = by_market.get(str(anomaly.get("market_id")), [])
        decision = evaluate_candidate(anomaly, market_cards, settings,
                                      run_id=ctx.run_id, run_time_utc=ctx.run_time_utc)
        (recommended if decision["kind"] == "recommended" else rejected).append(decision)

    # Every recommendation must satisfy the recommendation contract before we
    # present it — a schema failure here is a bug, not a data issue.
    for rec in recommended:
        validate_or_raise({k: v for k, v in rec.items()
                           if k not in {"kind", "is_live_timing", "concentration_red_flag"}},
                          "recommended_trade")

    workbook = _write_workbook(recommended, rejected, cards, settings, ctx)
    return RecommendationResult(recommended=recommended, rejected=rejected, workbook_path=workbook)


def already_recommended(settings: Settings) -> dict[str, str]:
    """Signal key -> the first run that recommended it, from earlier runs'
    recommendation files. Only recommendations of the current mode count, so the
    placeholder recommendations an old mock run made cannot block a live one."""
    mock_run = settings.card_mode == "mock"
    seen: dict[str, str] = {}
    for run_id, recs in io.run_stamped_json(settings.output_dir("recommended_trades"), "recommendations"):
        for rec in recs if isinstance(recs, list) else []:
            if bool(rec.get("mock_mode")) != mock_run:
                continue
            key = io.signal_key(rec.get("market_id"), rec.get("anomaly_trigger_time_utc"))
            if key:
                seen.setdefault(key, run_id)
    return seen


def _eligible(anomalies: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Keep only the strongest anomalies: the configured class, and within it the
    top fraction by score (ties at the cutoff are all kept). This is the
    no-trade bias made concrete."""
    if anomalies.empty:
        return anomalies
    cfg = settings.recommend
    target_class = cfg.get("eligible_anomaly_class", "strong")
    strong = anomalies[anomalies["anomaly_class"] == target_class].sort_values("anomaly_score", ascending=False)
    if strong.empty:
        return strong
    cutoff = float(cfg.get("strong_rank_percentile_cutoff", 0.5))
    keep = max(1, math.ceil(len(strong) * cutoff))
    threshold = strong["anomaly_score"].iloc[keep - 1]
    return strong[strong["anomaly_score"] >= threshold]


def _cards_by_market(cards: list[dict]) -> dict[str, list[dict]]:
    """Group cards by their source market id, so each anomaly can find its cards."""
    grouped: dict[str, list[dict]] = {}
    for card in cards:
        market_id = str((card.get("source") or {}).get("market_id"))
        grouped.setdefault(market_id, []).append(card)
    return grouped


# ---------------------------------------------------------------------------
# The human-readable workbook + machine-readable outputs
# ---------------------------------------------------------------------------
def _write_workbook(recommended: list[dict], rejected: list[dict], cards: list[dict],
                    settings: Settings, ctx: RunContext) -> Path | None:
    """Write the recommendation workbook (.xlsx) plus JSON recommendations and a
    CSV of rejected signals. The workbook is for humans; the JSON/CSV are for
    machines and audits."""
    out_dir = settings.output_dir("recommended_trades")
    stamp = ctx.run_id

    # Machine-readable + the rejection log.
    io.write_json(out_dir / f"recommendations_{stamp}.json", recommended)
    io.write_table(pd.DataFrame(rejected), settings.output_dir("rejected_signals"),
                   "rejected_signals", stamp, also_latest=True)

    workbook_path = out_dir / f"paper_recommendations_{stamp}.xlsx"
    card_links = [{"card_id": c["card_id"], "market_name": c["market_name"],
                   "created_time_utc": c["created_time_utc"], "mock_mode": c.get("mock_mode"),
                   "card_hash": c["card_hash"]} for c in cards]
    config_snapshot = [{"setting": k, "value": v} for k, v in {
        "card_mode": settings.card_mode, "backfill_mode": settings.backfill_mode,
        "eligible_anomaly_class": settings.recommend.get("eligible_anomaly_class"),
        "max_position_usd": settings.recommend.get("max_position_usd"),
        "paper_trading_only": settings.paper_trading_only,
    }.items()]

    # A prominent banner if this run is not fit for analysis.
    banners = []
    if settings.card_mode == "mock":
        banners.append("MOCK DATA — NOT FOR ANALYSIS (cards are placeholders).")
    if settings.backfill_mode:
        banners.append("BACKFILL / NON-LIVE — timing checks relaxed; not a live signal.")
    notice = [{"notice": b} for b in banners] or [{"notice": "Live-mode run."}]

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        pd.DataFrame(notice).to_excel(writer, sheet_name="README", index=False)
        _frame(recommended, "no recommendations this run").to_excel(writer, sheet_name="Recommendations", index=False)
        _frame(rejected, "no rejected signals").to_excel(writer, sheet_name="RejectedSignals", index=False)
        _frame(card_links, "no cards").to_excel(writer, sheet_name="CardLinks", index=False)
        pd.DataFrame(config_snapshot).to_excel(writer, sheet_name="ConfigSnapshot", index=False)

    io.update_manifest(ctx, "recommender", {
        "recommended_count": len(recommended),
        "rejected_count": len(rejected),
        "repeat_signals_skipped": sum(str(r.get("reason", "")).startswith("already recommended")
                                      for r in rejected),
        "workbook": str(workbook_path.relative_to(settings.project_root)),
        "backfill_mode": settings.backfill_mode,
    })
    return workbook_path


def _frame(rows: list[dict], empty_note: str) -> pd.DataFrame:
    """A DataFrame from rows, or a one-cell note so empty sheets read clearly."""
    return pd.DataFrame(rows) if rows else pd.DataFrame([{"note": empty_note}])


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run(ctx: RunContext, anomalies: pd.DataFrame | None = None,
        cards: list[dict] | None = None) -> RecommendationResult:
    """Produce paper recommendations from this run's strong anomalies and cards,
    write the workbook + rejection log, and update the manifest."""
    settings = ctx.settings
    if anomalies is None:
        anomalies = io.read_table(settings.output_dir("anomalies") / "strong_anomalies_latest.csv")
    if cards is None:
        cards = cards_module.load_cards(settings)
    return recommend(anomalies, cards, settings, ctx)
