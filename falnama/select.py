"""Stage 1 — build a small, high-quality geopolitical market universe.

WHAT:     Scores every candidate market for geopolitical relevance, drops
          structurally irrelevant ones (sports, crypto, entertainment...), and
          balances the result so no single event or region dominates.
CONSUMES: candidate markets from polymarket.fetch_markets + settings.selector
PRODUCES: outputs/relevant_markets/ (selected universe + per-market accept/reject
          log) and a selection-diagnostics summary under outputs/diagnostics/
REVIEWER: anyone asking "why is this market in (or out of) the analysis?"
ROLE:     the funnel. The goal is NOT to maximize market count but to make every
          inclusion and exclusion auditable — a clean universe makes every later
          stage cheaper and more trustworthy.

How the relevance score is built (all transparent, all preserved on the row):

    score = topic_base_score
          + bonuses   (small decision-maker, information asymmetry, cross-asset)
          - penalties (public polling, hard reject)      →  clamped to 0..100

The TOPIC is decided by keyword patterns below; the SCORES for each topic are
config knobs (settings.selector.topic_base_scores). That split is deliberate:
the taxonomy (what counts as "sanctions") is code; how much we value it is a dial.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Settings
from .io import RunContext

# Which keywords map a market's text onto a topic. Order matters: the first topic
# that matches becomes the primary topic, so the most information-sensitive
# categories are listed first. These are the analytic taxonomy, kept in code and
# commented; the *value* of each topic lives in the config.
TOPIC_KEYWORDS: list[tuple[str, list[str]]] = [
    ("military_conflict", ["war", "invasion", "missile", "strike", "airstrike", "military", "ceasefire", "nato", "troops", "attack", "conflict", "hostage"]),
    ("sanctions", ["sanction", "embargo", "asset freeze", "export ban"]),
    ("technology_controls", ["export control", "chip ban", "semiconductor", "huawei", "asml", "tech restriction"]),
    ("trade_policy", ["tariff", "trade war", "import ban", "trade deal", "duties", "quota"]),
    ("cabinet_government", ["cabinet", "coalition", "prime minister", "no confidence", "government collapse", "coup", "resign", "impeach"]),
    ("diplomacy_treaty", ["treaty", "summit", "negotiation", "peace talks", "accord", "diplomat"]),
    ("elections", ["election", "presidential", "parliament", "senate", "nominee", "primary", "referendum"]),
    ("central_bank_macro", ["central bank", "federal reserve", "interest rate", "rate cut", "inflation", "cpi", "ecb"]),
    ("energy", ["oil", "gas", "opec", "pipeline", "lng", "uranium", "energy"]),
    ("civil_unrest", ["protest", "riot", "civil unrest", "state of emergency", "martial law"]),
    ("regulatory_policy", ["regulation", "regulator", "sec", "ftc", "antitrust", "license"]),
    ("legal_judicial", ["court", "supreme court", "ruling", "verdict", "indictment", "trial"]),
    # Structurally low-relevance categories — matched so they can be scored near zero.
    ("weather", ["hurricane", "temperature", "rainfall", "snowfall", "weather"]),
    ("sports", ["nba", "nfl", "mlb", "nhl", "ufc", "soccer", "tennis", "esports", "premier league"]),
    ("crypto", ["bitcoin", "ethereum", "crypto", "solana", "memecoin", "token price"]),
    ("entertainment", ["oscar", "grammy", "emmy", "movie", "box office", "celebrity", "taylor swift"]),
]

# Topics whose outcomes often hinge on a few decision-makers (insider-prone) and
# whose information tends to be held unevenly — they earn the thesis bonuses.
_ASYMMETRIC_TOPICS = {
    "military_conflict", "sanctions", "trade_policy", "technology_controls",
    "cabinet_government", "diplomacy_treaty", "legal_judicial",
}
# Topics with clear knock-on effects for public financial assets.
_CROSS_ASSET_TOPICS = {"trade_policy", "sanctions", "energy", "central_bank_macro", "military_conflict"}

# Rough country/region inference, purely for diversity balancing and diagnostics.
_REGION_KEYWORDS = {
    "united_states": ["united states", " u.s.", "america", "trump", "biden", "congress"],
    "china_taiwan": ["china", "chinese", "taiwan", "beijing"],
    "russia_ukraine": ["russia", "ukraine", "putin", "zelensky", "moscow", "kyiv"],
    "middle_east": ["iran", "israel", "gaza", "hamas", "hezbollah", "syria", "lebanon"],
    "europe": ["europe", "european", "france", "germany", "britain", "uk ", "nato"],
    "korea": ["north korea", "south korea", "korea", "pyongyang"],
    "latin_america": ["mexico", "brazil", "argentina", "venezuela"],
}


@dataclass
class SelectionResult:
    """Outcome of Stage 1: the kept markets, the rejected ones (with reasons),
    and a small diagnostics summary (topic/region/score distributions)."""

    relevant: pd.DataFrame
    rejected: pd.DataFrame
    diagnostics: dict


# ---------------------------------------------------------------------------
# Per-market classification (pure, unit-testable)
# ---------------------------------------------------------------------------
def _text_blob(market: dict) -> str:
    """Concatenate a market's text fields into one lowercase string to search."""
    parts = [market.get(k) for k in ("market_name", "question", "description", "event_title", "category", "tags")]
    return " ".join(str(p) for p in parts if p).lower()


def _matches_any(blob: str, terms: list[str]) -> bool:
    """Whole-word match, tolerant of common English suffixes: 'sanction' matches
    'sanctions'/'sanctioned', while short words like 'war' still never fire on
    'warranty' (the suffix is optional and the word boundary is still required)."""
    return any(re.search(rf"(?<!\w){re.escape(t)}(?:s|es|ed|ing)?(?!\w)", blob) for t in terms)


def _infer_region(blob: str) -> str:
    for region, terms in _REGION_KEYWORDS.items():
        if any(t in blob for t in terms):
            return region
    return "unknown"


def classify_market(market: dict, settings: Settings) -> dict:
    """Return the relevance classification for one market: score, primary topic,
    region, event family, information structure, and the reasons behind the score.

    Kept pure (no I/O) so it is easy to test and reason about in isolation.
    """
    cfg = settings.selector
    base_scores = cfg.get("topic_base_scores", {})
    bonuses = cfg.get("bonuses", {})
    penalties = cfg.get("penalties", {})
    blob = _text_blob(market)

    # 1. Primary topic = first keyword group that matches (else "other").
    primary_topic = "other"
    reasons: list[str] = []
    for topic, terms in TOPIC_KEYWORDS:
        if _matches_any(blob, terms):
            primary_topic = topic
            reasons.append(f"matched {topic} keywords")
            break

    # 2. Start from the topic's base score.
    score = float(base_scores.get(primary_topic, base_scores.get("other", 20)))

    # 3. Thesis bonuses.
    if primary_topic in _ASYMMETRIC_TOPICS:
        score += float(bonuses.get("small_decision_maker", 0)) + float(bonuses.get("information_asymmetry", 0))
        reasons.append("small decision-maker + information-asymmetry bonus")
    if primary_topic in _CROSS_ASSET_TOPICS:
        score += float(bonuses.get("cross_asset", 0))
        reasons.append("cross-asset bonus")

    # 4. Penalties. Poll-driven markets reflect public info, not private leaks.
    if _matches_any(blob, ["poll", "polling", "approval rating", "popular vote"]):
        score += float(penalties.get("public_polling", 0))
        reasons.append("public-polling penalty")
    hard_rejected = _is_hard_reject(blob, primary_topic, cfg)
    if hard_rejected:
        score += float(penalties.get("hard_reject", 0))
        reasons.append("hard-reject penalty")

    score = float(max(0, min(100, round(score, 1))))
    return {
        "relevance_score": score,
        "primary_topic": primary_topic,
        "country_or_region": _infer_region(blob),
        "event_family": str(market.get("event_slug") or market.get("event_title") or f"{primary_topic}:{_infer_region(blob)}"),
        "information_structure": "asymmetry_prone" if primary_topic in _ASYMMETRIC_TOPICS else "public_or_macro_driven",
        "hard_rejected": hard_rejected,
        "classification_reason": "; ".join(reasons) if reasons else "no relevance rule matched",
    }


def _is_hard_reject(blob: str, primary_topic: str, selector_cfg: dict) -> bool:
    """A market is a hard reject if it hits a hard-reject keyword or its topic is
    on the hard-reject topic list (sports/crypto/entertainment/weather)."""
    if _matches_any(blob, [str(k).lower() for k in selector_cfg.get("hard_reject_keywords", [])]):
        return True
    return primary_topic in set(selector_cfg.get("hard_reject_topics", []))


# ---------------------------------------------------------------------------
# Universe selection (classify → filter → diversity-balance)
# ---------------------------------------------------------------------------
def select_markets(markets: pd.DataFrame, settings: Settings) -> SelectionResult:
    """Classify every candidate, apply the relevance/quality filters, then
    diversity-balance the survivors. Returns kept + rejected + diagnostics."""
    cfg = settings.selector
    min_score = float(cfg.get("min_relevance_score", 60))
    min_volume = float(cfg.get("min_total_volume", 0))
    min_liquidity = float(cfg.get("min_liquidity", 0))

    kept: list[dict] = []
    rejected: list[dict] = []
    for record in markets.to_dict(orient="records"):
        classification = classify_market(record, settings)
        row = {**record, **classification}
        reason = _rejection_reason(row, classification, min_score, min_volume, min_liquidity)
        if reason:
            rejected.append({**row, "rejection_reason": reason})
        else:
            kept.append(row)

    relevant = pd.DataFrame(kept)
    relevant, diversity_removed = _diversity_balance(relevant, settings)
    rejected_df = pd.concat([pd.DataFrame(rejected), diversity_removed], ignore_index=True) if diversity_removed is not None and not diversity_removed.empty else pd.DataFrame(rejected)
    return SelectionResult(relevant=relevant, rejected=rejected_df, diagnostics=_diagnostics(relevant, rejected_df))


def _rejection_reason(row: dict, classification: dict, min_score: float,
                      min_volume: float, min_liquidity: float) -> str:
    """Return the first reason this market fails selection, or '' if it passes."""
    if classification["hard_rejected"]:
        return "hard reject (keyword or topic)"
    if classification["relevance_score"] < min_score:
        return f"relevance score {classification['relevance_score']} < {min_score}"
    volume = _as_float(row.get("volume"))
    if min_volume > 0 and not np.isnan(volume) and volume < min_volume:
        return f"volume {volume:g} < {min_volume:g}"
    liquidity = _as_float(row.get("liquidity"))
    if min_liquidity > 0 and not np.isnan(liquidity) and 0 < liquidity < min_liquidity:
        return f"liquidity {liquidity:g} < {min_liquidity:g}"
    return ""


def _diversity_balance(relevant: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Cap how many markets any one region / event family / topic contributes,
    while always preserving the strongest few per cluster. Keeps the universe
    from being flooded by a single busy event."""
    cfg = settings.selector.get("diversity", {})
    if relevant.empty or not cfg.get("enabled", True):
        return relevant.reset_index(drop=True), None

    ranked = relevant.sort_values("relevance_score", ascending=False).reset_index(drop=True)
    preserve_n = int(cfg.get("preserve_top_n_per_cluster", 5))
    caps = {
        "country_or_region": int(cfg.get("max_per_country_or_region", 15)),
        "event_family": int(cfg.get("max_per_event_family", 10)),
        "primary_topic": int(cfg.get("max_per_primary_topic", 50)),
    }
    # Always-keep the top-N per cluster so a strong signal is never balanced away.
    protected: set[int] = set()
    for column in caps:
        for _, group in ranked.groupby(column, dropna=False):
            protected.update(group.head(preserve_n).index)

    kept_idx, removed_rows, counts = [], [], {c: {} for c in caps}
    for idx, row in ranked.iterrows():
        capped_by = ""
        if idx not in protected:
            for column, cap in caps.items():
                value = str(row.get(column, "unknown"))
                if counts[column].get(value, 0) >= cap:
                    capped_by = f"diversity cap: {column}={value}"
                    break
        if capped_by:
            removed_rows.append({**row.to_dict(), "rejection_reason": capped_by})
            continue
        kept_idx.append(idx)
        for column in caps:
            value = str(row.get(column, "unknown"))
            counts[column][value] = counts[column].get(value, 0) + 1

    kept = ranked.loc[kept_idx].reset_index(drop=True)
    removed = pd.DataFrame(removed_rows)
    return kept, removed


def _diagnostics(relevant: pd.DataFrame, rejected: pd.DataFrame) -> dict:
    """Small summary of the selection, saved alongside the run for quick review."""
    def counts(df: pd.DataFrame, col: str) -> dict:
        return {} if df.empty or col not in df else df[col].astype(str).value_counts().to_dict()

    return {
        "selected_count": int(len(relevant)),
        "rejected_count": int(len(rejected)),
        "by_primary_topic": counts(relevant, "primary_topic"),
        "by_country_or_region": counts(relevant, "country_or_region"),
        "rejection_reasons": counts(rejected, "rejection_reason"),
    }


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run(ctx: RunContext, markets: pd.DataFrame | None = None) -> SelectionResult:
    """Select markets, write artifacts, and record what happened in the manifest.

    If `markets` is None, the candidate universe is fetched via the data layer
    (fixtures or live, per config).
    """
    from . import io, polymarket

    if markets is None:
        markets = polymarket.fetch_markets(ctx.settings, ctx)

    result = select_markets(markets, ctx.settings)

    out_dir = ctx.settings.output_dir("relevant_markets")
    selected_path = io.write_table(result.relevant, out_dir, "relevant_markets", ctx.run_id)
    io.write_table(result.rejected, out_dir, "rejected_markets", ctx.run_id, also_latest=False)
    io.write_json(ctx.settings.output_dir("diagnostics") / f"selection_{ctx.run_id}.json", result.diagnostics)

    io.update_manifest(ctx, "market_selector", {
        "input_markets": int(len(markets)),
        "selected_count": int(len(result.relevant)),
        "rejected_count": int(len(result.rejected)),
        "selected_file": str(selected_path.relative_to(ctx.settings.project_root)),
    })
    return result
