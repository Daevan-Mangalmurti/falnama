"""Stage 2 — score market behavior for anomalies (interpretable, not a black box).

WHAT:     Turns each market's price history into a transparent composite
          "anomalousness" score (0-100) and a severity class.
CONSUMES: price history from polymarket.fetch_price_history + settings.anomaly
PRODUCES: outputs/anomalies/ — ranked anomalies, the strong subset, and
          concentration diagnostics (or a recorded reason they're unavailable)
REVIEWER: a human deciding which signals deserve investigation
ROLE:     the detector. It answers only "did the market move unusually?" — NOT
          "was it already public?" (that is Stage 5, news-lag).

The composite is a weighted blend of four visible sub-scores, plus a small
time-to-close bonus. Each sub-score is kept on the output row so a reviewer can
see *why* a market ranked where it did:

    magnitude    how big was the largest move over any rolling window
    speed        how fast (the biggest move over the shortest window)
    persistence  did the move stick, or did it snap back
    unusualness  how large the move was relative to THIS market's own volatility
    (+ time_to_close bonus: moves just before resolution are more suspicious)

We score ONE anomaly per market — its single most anomalous window — which
naturally de-duplicates repeated bursts (the cooldown idea) without extra state.

The calibration references below (what counts as a "big"/"fast" move, etc.) are
documented module constants rather than config: they define the score's meaning.
The severity thresholds that act on the score ARE config (settings.anomaly).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Settings
from .io import RunContext

# ---- calibration references (the score's "meaning") -------------------------
# Prices are probabilities in [0, 1], so moves are in the same units.
MOVE_REFERENCE = 0.40       # an absolute move of 0.40 over a window → magnitude 100
SPEED_REFERENCE = 0.20      # a 0.20 move within the SHORTEST window → speed 100
Z_REFERENCE = 6.0           # a 6-sigma step (vs the market's own noise) → unusualness 100
TTC_WINDOW_DAYS = 7.0       # moves within 7 days of resolution earn the time-to-close bonus
TTC_MAX_BONUS = 10.0        # ...worth at most +10 on the composite
PERSISTENCE_WINDOW = "6h"   # how long after the move we check whether it held

# How the four core sub-scores combine (weights sum to 1.0). Documented here so
# the composite is fully explainable from this one place.
WEIGHTS = {"magnitude": 0.35, "speed": 0.25, "persistence": 0.20, "unusualness": 0.20}


@dataclass
class AnomalyResult:
    """Outcome of Stage 2: all ranked anomalies, the strong subset that Stage 4
    may act on, and concentration diagnostics."""

    ranked: pd.DataFrame
    strong: pd.DataFrame
    concentration: pd.DataFrame


# ---------------------------------------------------------------------------
# Scoring one market (pure, unit-testable)
# ---------------------------------------------------------------------------
def score_market(price_history: pd.DataFrame, settings: Settings,
                 concentration: dict | None = None) -> dict | None:
    """Return the sub-scores + composite for one market's price history, or None
    if there is too little history to judge.

    The input is the rows for a single market (columns: timestamp, price, and
    optionally close_time / volume / liquidity). `concentration` is an optional
    wallet-concentration record for the red-flag overlay (None for the many
    markets without such data — that case never lowers the score). Output is a
    flat dict, ready to become a row in the ranked-anomalies table.
    """
    cfg = settings.anomaly
    min_obs = int(cfg.get("min_price_observations", 20))

    series = (price_history.dropna(subset=["price"])
              .assign(timestamp=lambda d: pd.to_datetime(d["timestamp"], utc=True))
              .sort_values("timestamp")
              .set_index("timestamp")["price"].astype(float))
    if len(series) < min_obs:
        return None

    windows = [pd.Timedelta(w) for w in cfg.get("rolling_windows", ["1h", "6h", "24h"])]

    # --- magnitude: the largest absolute move over ANY window, and when it happened
    max_move, trigger_time, pre_price, peak_price = _largest_move(series, windows)
    magnitude = _scale(abs(max_move), MOVE_REFERENCE)

    # --- speed: the largest move over the SHORTEST window (a fast move is worse)
    shortest = min(windows)
    fast_move, _, _, _ = _largest_move(series, [shortest])
    speed = _scale(abs(fast_move), SPEED_REFERENCE)

    # --- persistence: did the move hold over the following window, or revert?
    persistence = _persistence(series, trigger_time, pre_price, peak_price)

    # --- unusualness: the biggest single-step return vs this market's own noise
    unusualness = _unusualness(series)

    # --- time-to-close bonus: moves near resolution are more suspicious
    ttc_bonus = _time_to_close_bonus(price_history, trigger_time)

    # --- concentration overlay: an independent red flag, never a core sub-score
    first = price_history.iloc[0]
    overlay = concentration_overlay(first, concentration, cfg)

    core = sum(WEIGHTS[name] * value for name, value in {
        "magnitude": magnitude, "speed": speed,
        "persistence": persistence, "unusualness": unusualness,
    }.items())
    composite = float(min(100.0, round(core + ttc_bonus + overlay["concentration_bonus"], 1)))

    return {
        "market_id": first.get("market_id"),
        "market_name": first.get("market_name"),
        "anomaly_trigger_time_utc": trigger_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "price_before": round(float(pre_price), 4),
        "price_after": round(float(peak_price), 4),
        "max_abs_move": round(float(abs(max_move)), 4),
        "score_magnitude": magnitude,
        "score_speed": speed,
        "score_persistence": persistence,
        "score_unusualness": unusualness,
        "time_to_close_bonus": round(ttc_bonus, 1),
        **overlay,  # concentration_available / _tier / _red_flag / _bonus + metrics
        "anomaly_score": composite,
        "observations": int(len(series)),
    }


def _largest_move(series: pd.Series, windows: list[pd.Timedelta]):
    """Find the largest absolute (price now − price one window ago) across all
    windows. Returns (signed_move, trigger_time, price_before, price_after)."""
    best = (0.0, series.index[-1], float(series.iloc[0]), float(series.iloc[-1]))
    for window in windows:
        earlier = series.asof(series.index - window)  # price `window` ago, at each timestamp
        moves = series.values - earlier.values
        if np.all(np.isnan(moves)):
            continue
        i = int(np.nanargmax(np.abs(moves)))
        if abs(moves[i]) > abs(best[0]):
            best = (float(moves[i]), series.index[i], float(earlier.values[i]), float(series.values[i]))
    return best


def _persistence(series: pd.Series, trigger_time, pre_price: float, peak_price: float) -> float:
    """Fraction of the move still in place over the window AFTER the trigger.
    ~100 means the move held; ~0 means it fully reverted."""
    move = peak_price - pre_price
    if abs(move) < 1e-9:
        return 0.0
    after = series[(series.index > trigger_time) & (series.index <= trigger_time + pd.Timedelta(PERSISTENCE_WINDOW))]
    if after.empty:
        return 100.0  # no reversal observed within the window
    held = (after.mean() - pre_price) / move
    return float(round(100.0 * min(1.0, max(0.0, held)), 1))


def _unusualness(series: pd.Series) -> float:
    """Largest single-step return expressed in standard deviations of this
    market's returns, scaled to 0-100. Flags moves that are big *for this market*."""
    returns = series.diff().dropna()
    std = float(returns.std())
    if std < 1e-9 or returns.empty:
        return 0.0
    z = float(returns.abs().max() / std)
    return _scale(z, Z_REFERENCE)


def _time_to_close_bonus(price_history: pd.DataFrame, trigger_time) -> float:
    """Up to +TTC_MAX_BONUS when the move lands within TTC_WINDOW_DAYS of
    resolution. Zero (no bonus) when the close time is unknown."""
    close_raw = price_history.iloc[0].get("close_time")
    close_time = pd.to_datetime(close_raw, utc=True, errors="coerce")
    if pd.isna(close_time):
        return 0.0
    days_to_close = (close_time - trigger_time).total_seconds() / 86400.0
    if days_to_close < 0:
        return 0.0
    nearness = max(0.0, (TTC_WINDOW_DAYS - days_to_close) / TTC_WINDOW_DAYS)
    return float(TTC_MAX_BONUS * nearness)


def _scale(value: float, reference: float) -> float:
    """Map a raw magnitude onto 0-100, saturating at `reference`."""
    return float(round(100.0 * min(1.0, value / reference), 1)) if reference > 0 else 0.0


def _num(value) -> float:
    """Coerce to float; missing/garbage becomes NaN (which fails every >= test,
    so absent volume/liquidity is treated as 'not thick', never as thick)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Concentration overlay (an independent red flag — see the module docstring)
# ---------------------------------------------------------------------------
def concentration_overlay(market_row, concentration: dict | None, cfg: dict) -> dict:
    """Interpret a wallet-concentration record as a red flag for one market.

    This NEVER lowers a score. When wallet data is missing — the common case —
    the tier is 'unavailable' and the bonus is 0, so the composite is unchanged.
    The red flag fires only for a THICK market (broad participation is the norm)
    showing EXTREME concentration (a few wallets dominate): a liquid market moved
    by a small group, which is the insider-leak fingerprint Falnama hunts for. A
    thin-but-concentrated market is unremarkable and gets no flag.

    Returns the overlay columns that ride along on the anomaly row.
    """
    c = cfg.get("concentration", {})
    overlay = {
        "concentration_available": False, "concentration_tier": "unavailable",
        "concentration_red_flag": False, "concentration_bonus": 0.0,
        "top1_wallet_share": None, "top3_wallet_share": None, "wallet_gini": None,
        "concentration_reason": "",
    }
    if not c.get("enabled", True):
        overlay["concentration_reason"] = "concentration overlay disabled in config"
        return overlay
    if not concentration or not concentration.get("available", False):
        overlay["concentration_reason"] = (concentration or {}).get("reason", "no wallet-level trade data for this market")
        return overlay

    top1 = float(concentration.get("top1_share") or 0.0)
    top3 = float(concentration.get("top3_share") or 0.0)
    gini = float(concentration.get("gini") or 0.0)
    volume, liquidity = _num(market_row.get("volume")), _num(market_row.get("liquidity"))

    thick = (volume >= float(c.get("thick_min_volume", float("inf")))) or \
            (liquidity >= float(c.get("thick_min_liquidity", float("inf"))))
    extreme = (top1 >= float(c.get("extreme_top1_share", 1.1)) or
               top3 >= float(c.get("extreme_top3_share", 1.1)) or
               gini >= float(c.get("extreme_gini", 1.1)))

    if extreme and thick:
        tier, flag, bonus = "red_flag", True, float(c.get("red_flag_bonus", 15))
        reason = f"THICK market moved by a few wallets (top1={top1:.0%}, top3={top3:.0%}, gini={gini:.2f})"
    elif extreme:
        tier, flag, bonus = "concentrated_thin", False, 0.0
        reason = "concentrated, but market is thin where concentration is unremarkable"
    else:
        tier, flag, bonus = "diffuse", False, 0.0
        reason = "wallet participation is diffuse"

    overlay.update({
        "concentration_available": True, "concentration_tier": tier,
        "concentration_red_flag": flag, "concentration_bonus": bonus,
        "top1_wallet_share": round(top1, 4), "top3_wallet_share": round(top3, 4),
        "wallet_gini": round(gini, 4), "concentration_reason": reason,
    })
    return overlay


# ---------------------------------------------------------------------------
# Scoring the whole universe
# ---------------------------------------------------------------------------
def detect_anomalies(price_history: pd.DataFrame, settings: Settings,
                     concentration_by_market: dict[str, dict] | None = None) -> AnomalyResult:
    """Score every market, rank by composite score, and classify severity.

    `concentration_by_market` maps market_id -> a wallet-concentration record; it
    is absent for the many markets without such data, and feeds the red-flag
    overlay for the few that have it.
    """
    cfg = settings.anomaly
    concentration_by_market = concentration_by_market or {}
    scored = []
    for market_id, group in price_history.groupby("market_id", sort=False):
        row = score_market(group, settings, concentration_by_market.get(str(market_id)))
        if row is not None:
            scored.append(row)

    ranked = pd.DataFrame(scored)
    if not ranked.empty:
        ranked = ranked.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
        ranked["anomaly_class"] = ranked["anomaly_score"].map(lambda s: _classify(s, cfg))
        ranked = ranked.head(int(cfg.get("top_n", 100)))

    strong = ranked[ranked["anomaly_class"] == "strong"].copy() if not ranked.empty else ranked
    return AnomalyResult(ranked=ranked, strong=strong, concentration=_concentration(ranked))


def _classify(score: float, cfg: dict) -> str:
    """Map a composite score to a severity class using the configured thresholds."""
    if score >= float(cfg.get("strong_threshold", 85)):
        return "strong"
    if score >= float(cfg.get("medium_threshold", 70)):
        return "medium"
    if score >= float(cfg.get("weak_threshold", 50)):
        return "weak"
    return "subthreshold"


def _concentration(ranked: pd.DataFrame) -> pd.DataFrame:
    """Slice the concentration overlay columns into a standalone diagnostics table.

    Wallet-concentration data covers only a minority of markets, so each row is
    explicit about availability and tier — downstream code never confuses MISSING
    data ('unavailable') with LOW concentration ('diffuse')."""
    cols = ["market_id", "market_name", "concentration_available", "concentration_tier",
            "concentration_red_flag", "top1_wallet_share", "top3_wallet_share",
            "wallet_gini", "concentration_reason"]
    if ranked.empty:
        return pd.DataFrame(columns=cols)
    return ranked[[c for c in cols if c in ranked.columns]].copy()


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run(ctx: RunContext, price_history: pd.DataFrame | None = None) -> AnomalyResult:
    """Detect anomalies, write artifacts, and record what happened in the manifest.

    If `price_history` is None, it is fetched (via the data layer) for the markets
    selected in this run's latest relevant-markets file.
    """
    from . import io, polymarket

    concentration = None
    if price_history is None:
        selected = pd.read_csv(ctx.settings.output_dir("relevant_markets") / "relevant_markets_latest.csv")
        market_ids = [str(m) for m in selected["market_id"].dropna().tolist()]
        price_history = polymarket.fetch_price_history(ctx.settings, market_ids, ctx)
        # Best-effort wallet concentration; returns records only for covered markets.
        concentration = polymarket.fetch_trade_concentration(ctx.settings, market_ids, ctx)

    result = detect_anomalies(price_history, ctx.settings, concentration)

    out_dir = ctx.settings.output_dir("anomalies")
    ranked_path = io.write_table(result.ranked, out_dir, "ranked_anomalies", ctx.run_id)
    io.write_table(result.strong, out_dir, "strong_anomalies", ctx.run_id, also_latest=True)
    io.write_table(result.concentration, out_dir, "concentration_diagnostics", ctx.run_id, also_latest=False)

    red_flags = int(result.ranked["concentration_red_flag"].sum()) if not result.ranked.empty else 0
    io.update_manifest(ctx, "anomaly_detector", {
        "markets_scored": int(len(result.ranked)),
        "strong_count": int(len(result.strong)),
        "concentration_red_flags": red_flags,
        "ranked_file": str(ranked_path.relative_to(ctx.settings.project_root)),
    })
    return result
