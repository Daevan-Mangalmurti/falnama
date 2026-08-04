"""Phase 1 detection backtest — would the detector have caught known anomalies?

WHAT:     Replays a curated list of ALREADY-RESOLVED Polymarket markets through
          the live anomaly detector and asks, for each: did a strong anomaly form
          in the run-up BEFORE the outcome became public, and how many hours of
          lead would we have had?
CONSUMES: config/backtest_markets.yaml (the curated cases) + Polymarket's public
          read-only APIs (Gamma for metadata, CLOB for price history)
PRODUCES: outputs/backtest/detection_<run_id>.csv (one row per case) + a summary
          (recall on the documented insider cases, false-positive rate on controls)
REVIEWER: anyone asking "would Falnama actually have flagged the Iran-strike moves
          the world later decided were insider trading?"
ROLE:     the honest, LLM-free half of the backtest. It touches only price data and
          the deterministic detector, so there is no hindsight contamination — the
          card/LLM step (which an out-of-cutoff model cannot un-know) belongs to the
          separate, explicitly-illustrative financial backtest.

Method, and its honest limits:
  * For each resolved market we fetch the ~14-day window ending at resolution (the
    CLOB history endpoint caps a request near 14 days at hourly resolution, which
    is exactly our lookback).
  * We EXCLUDE the terminal settlement — once the price parks in the resolved zone
    (>= SETTLED_HI or <= SETTLED_LO) it is just the outcome becoming public, not a
    tradable lead — and score only the PRE-settlement run-up with the same
    `anomaly.score_market` the live pipeline uses.
  * "Detected" = the peak composite in that pre-settlement window reaches the
    strong threshold. "Lead" = settlement onset minus the anomaly's trigger time.
  * We cannot pin the exact public-news timestamp from price alone (that is what
    the news-lag module adds); here the price settlement is the news proxy, so the
    lead is a lower bound on the informational head start, reported as such.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from . import anomaly, io
from .config import Settings, load_config
from .io import RunContext

# A binary market has "settled" once its price parks near 0 or 1 — that tail is the
# outcome going public, not a lead, so we cut it before scoring the run-up.
SETTLED_HI = 0.90
SETTLED_LO = 0.10
# The CLOB history request's hard span cap at hourly resolution (see module note).
_MAX_WINDOW_DAYS = 14
_GAMMA = "https://gamma-api.polymarket.com"
_CLOB = "https://clob.polymarket.com"


@dataclass
class BacktestCase:
    """One curated market to replay: an event slug, which market inside it, and a
    label. `market_contains` disambiguates when an event holds many dated markets."""

    event: str
    label: str                      # "positive" (documented case) | "control"
    market_contains: str = ""
    note: str = ""


@dataclass
class BacktestOutcome:
    """The detector's verdict on one replayed market — a row in the results table."""

    event: str
    label: str
    note: str
    market_name: str | None = None
    resolved_outcome: str | None = None       # "YES" | "NO" | None
    history_points: int = 0
    peak_anomaly_score: float | None = None
    detected_strong: bool | None = None
    lead_hours: float | None = None           # settlement onset − anomaly trigger
    trigger_time_utc: str | None = None
    settlement_time_utc: str | None = None
    error: str = ""


# ---------------------------------------------------------------------------
# Data access (Polymarket public read-only APIs) — isolated so tests can mock it
# ---------------------------------------------------------------------------
def _gamma_event(slug: str, settings: Settings) -> dict:
    """Fetch one event (with its nested markets) by slug."""
    import requests

    headers = {"User-Agent": str(settings.data.get("user_agent", "falnama-research/0.1 (read-only)"))}
    resp = requests.get(f"{_GAMMA}/events", params={"slug": slug}, headers=headers,
                        timeout=int(settings.data.get("request_timeout_seconds", 30)))
    resp.raise_for_status()
    payload = resp.json()
    events = payload if isinstance(payload, list) else payload.get("data", [])
    if not events:
        raise LookupError(f"no event found for slug {slug!r}")
    return events[0]


def _clob_history(token_id: str, start_ts: int, end_ts: int, settings: Settings) -> pd.DataFrame:
    """Fetch hourly price history for one CLOB token over [start_ts, end_ts]."""
    import requests

    resp = requests.get(f"{_CLOB}/prices-history",
                        params={"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": 60},
                        timeout=int(settings.data.get("request_timeout_seconds", 30)))
    resp.raise_for_status()
    points = resp.json().get("history", [])
    if not points:
        return pd.DataFrame(columns=["timestamp", "price"])
    return pd.DataFrame({
        "timestamp": [pd.Timestamp(p["t"], unit="s", tz="UTC") for p in points],
        "price": [float(p["p"]) for p in points],
    })


# ---------------------------------------------------------------------------
# Selecting the market and its pre-resolution window
# ---------------------------------------------------------------------------
def _pick_market(event: dict, contains: str) -> dict:
    """Choose the market inside an event: the first whose question contains
    `contains` (case-insensitive), or the only/first market if `contains` is blank."""
    markets = event.get("markets") or []
    if not markets:
        raise LookupError("event has no markets")
    if not contains:
        return markets[0]
    needle = contains.lower()
    for market in markets:
        if needle in str(market.get("question", "")).lower():
            return market
    raise LookupError(f"no market matching {contains!r} in event {event.get('slug')!r}")


def _resolution_anchor(market: dict, settings: Settings) -> pd.Timestamp:
    """Best estimate of when the market resolved, used to anchor the 14-day fetch.

    Gamma's `endDate` is unreliable for dated sub-markets, so we prefer the most
    recent trade timestamp (the market stops trading at resolution) and fall back
    to endDate only if the trades feed is unavailable."""
    import requests

    condition = market.get("conditionId")
    if condition:
        try:
            resp = requests.get(f"{settings.data.get('data_api_base_url', 'https://data-api.polymarket.com')}/trades",
                                params={"market": condition, "limit": 1},
                                timeout=int(settings.data.get("request_timeout_seconds", 30)))
            trades = resp.json() if resp.ok else []
            if isinstance(trades, list) and trades and trades[0].get("timestamp"):
                return pd.Timestamp(int(trades[0]["timestamp"]), unit="s", tz="UTC")
        except Exception:
            pass
    return pd.to_datetime(market.get("endDate"), utc=True, errors="coerce")


def _outcome(market: dict) -> str | None:
    """Resolve YES/NO from the settled outcome prices, if present."""
    raw = market.get("outcomePrices")
    prices = json.loads(raw) if isinstance(raw, str) else raw
    if not prices:
        return None
    return "YES" if float(prices[0]) >= 0.5 else "NO"


def _pre_settlement(history: pd.DataFrame, outcome: str | None) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Split the run-up from the terminal settlement. Returns (pre_settlement rows,
    settlement_onset_time). The settlement is the final stretch parked in the
    resolved zone; everything before it is the tradable run-up we score."""
    if history.empty:
        return history, None
    series = history.sort_values("timestamp").reset_index(drop=True)
    price = series["price"]
    in_zone = price >= SETTLED_HI if outcome == "YES" else price <= SETTLED_LO
    # Find the start of the LAST contiguous settled stretch that runs to the end.
    onset_idx = None
    for i in range(len(series) - 1, -1, -1):
        if in_zone.iloc[i]:
            onset_idx = i
        else:
            break
    if onset_idx is None or onset_idx == 0:
        return series, (series["timestamp"].iloc[onset_idx] if onset_idx == 0 else None)
    return series.iloc[:onset_idx].reset_index(drop=True), series["timestamp"].iloc[onset_idx]


# ---------------------------------------------------------------------------
# Running one case and the whole set
# ---------------------------------------------------------------------------
def run_case(case: BacktestCase, settings: Settings) -> BacktestOutcome:
    """Replay one resolved market and return the detector's verdict on it. Every
    failure mode (missing event, no history, pruned data) is captured on the row
    rather than raised, so one bad case never sinks the whole backtest."""
    out = BacktestOutcome(event=case.event, label=case.label, note=case.note)
    try:
        event = _gamma_event(case.event, settings)
        market = _pick_market(event, case.market_contains)
        out.market_name = str(market.get("question") or event.get("title"))
        out.resolved_outcome = _outcome(market)

        token_ids = market.get("clobTokenIds")
        token_ids = json.loads(token_ids) if isinstance(token_ids, str) else token_ids
        anchor = _resolution_anchor(market, settings)
        if not token_ids or pd.isna(anchor):
            out.error = "no CLOB token or resolution time"
            return out

        end_ts = int(anchor.timestamp())
        history = _clob_history(str(token_ids[0]), end_ts - _MAX_WINDOW_DAYS * 86400, end_ts, settings)
        out.history_points = int(len(history))
        if history.empty:
            out.error = "no price history (likely pruned — market too old)"
            return out

        pre, settlement = _pre_settlement(history, out.resolved_outcome)
        if settlement is not None:
            out.settlement_time_utc = settlement.strftime("%Y-%m-%dT%H:%M:%SZ")
        if len(pre) < int(settings.anomaly.get("min_price_observations", 20)):
            out.error = f"too few pre-settlement points ({len(pre)}) to score"
            return out

        scored = _score(pre, market, settings)
        if scored is None:
            out.error = "detector returned no score"
            return out
        out.peak_anomaly_score = float(scored["anomaly_score"])
        out.detected_strong = bool(scored["anomaly_score"] >= float(settings.anomaly.get("strong_threshold", 85)))
        out.trigger_time_utc = scored["anomaly_trigger_time_utc"]
        if settlement is not None and scored["anomaly_trigger_time_utc"]:
            trigger = pd.to_datetime(scored["anomaly_trigger_time_utc"], utc=True)
            out.lead_hours = round((settlement - trigger).total_seconds() / 3600.0, 1)
    except Exception as exc:  # network, parse, lookup — recorded, never fatal
        out.error = f"{type(exc).__name__}: {exc}"
    return out


def _score(pre: pd.DataFrame, market: dict, settings: Settings) -> dict | None:
    """Score the pre-settlement window with the live detector. We shape the frame
    to what `anomaly.score_market` expects (a single market's price rows)."""
    frame = pre.assign(
        market_id=io.clean_id(market.get("id")) or market.get("conditionId"),
        market_name=str(market.get("question") or ""),
        timestamp=pre["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        close_time=None,
    )
    return anomaly.score_market(frame, settings)


def run_backtest(cases: list[BacktestCase], settings: Settings) -> tuple[pd.DataFrame, dict]:
    """Replay every case and summarize: detection recall on the documented
    positives, and the false-positive rate on the controls."""
    rows = [asdict(run_case(c, settings)) for c in cases]
    table = pd.DataFrame(rows)

    def rate(label: str) -> dict:
        sub = table[(table["label"] == label) & table["detected_strong"].notna()]
        flagged = int(sub["detected_strong"].sum()) if not sub.empty else 0
        return {"scored": int(len(sub)), "flagged_strong": flagged,
                "rate": round(flagged / len(sub), 3) if len(sub) else None}

    summary = {
        "cases": int(len(table)),
        "usable": int(table["detected_strong"].notna().sum()),
        "unavailable": int(table["error"].astype(bool).sum()),
        "positives_recall": rate("positive"),
        "controls_false_positive": rate("control"),
    }
    return table, summary


# ---------------------------------------------------------------------------
# Loading the curated set + entry point
# ---------------------------------------------------------------------------
def load_cases(path: str | Path | None = None, settings: Settings | None = None) -> list[BacktestCase]:
    """Read config/backtest_markets.yaml into a list of cases."""
    import yaml

    settings = settings or load_config()
    path = Path(path) if path else settings.project_root / "config" / "backtest_markets.yaml"
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return [BacktestCase(**{k: v for k, v in item.items() if k in BacktestCase.__annotations__})
            for item in raw.get("cases", [])]


def run(settings: Settings | None = None, cases_path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Run the detection backtest end to end and write the results + summary."""
    settings = settings or load_config()
    ctx = RunContext.start(settings)
    cases = load_cases(cases_path, settings)
    table, summary = run_backtest(cases, settings)

    out_dir = settings.output_dir("backtest") if "backtest" in settings.section("outputs") else \
        (settings.project_root / "outputs" / "backtest")
    out_dir.mkdir(parents=True, exist_ok=True)
    io.write_table(table, out_dir, "detection", ctx.run_id, also_latest=True)
    io.write_json(out_dir / f"summary_{ctx.run_id}.json", summary)
    return table, summary
