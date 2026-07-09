"""Read-only access to Polymarket data (with a deterministic fixtures fallback).

WHAT:     Fetches markets and price history from Polymarket's public APIs, and —
          when data.source is 'fixtures' — reads committed sample data instead.
CONSUMES: Polymarket Gamma/CLOB APIs (live) or data/fixtures/*.csv (fixtures)
PRODUCES: pandas DataFrames; on a live run also freezes the raw pull under
          data/snapshots/<run_id>/ so the run is reproducible from frozen inputs.
REVIEWER: anyone debugging data quality or reproducing a past run
ROLE:     the boundary between Falnama and the outside world. It is strictly
          read-only: Falnama observes markets, it never interacts with them.

Two paths, chosen by settings.data_source:
  * fixtures → load small, committed sample tables. No network, fully
    deterministic. This is the default, and what tests and CI use.
  * live → page the public APIs. Exercised only when you opt in with
    data.source: live; the fixtures path is the one covered by tests.

Both paths return the same normalized columns, so the rest of the pipeline does
not know or care which source produced the data.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .io import RunContext

# The normalized columns every market row carries, whatever the source.
MARKET_COLUMNS = [
    "market_id", "market_slug", "market_name", "question", "description",
    "category", "tags", "event_slug", "event_title", "volume", "liquidity",
    "closed", "close_time", "clob_token_ids", "market_url",
]


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def fetch_markets(settings: Settings, ctx: RunContext | None = None) -> pd.DataFrame:
    """Return the candidate market universe as a normalized DataFrame."""
    if settings.data_source == "fixtures":
        return load_fixture(settings, "markets")
    df = _fetch_gamma_markets(settings)
    if ctx is not None and settings.data.get("snapshot_raw", True):
        snapshot(settings, ctx.run_id, "markets", df)
    return df


def fetch_price_history(settings: Settings, market_ids: list[str],
                        ctx: RunContext | None = None) -> pd.DataFrame:
    """Return long-format price history (one row per market per timestamp) for the
    given market IDs. Columns: market_id, market_name, timestamp, price,
    volume, liquidity, close_time."""
    if settings.data_source == "fixtures":
        history = load_fixture(settings, "price_history")
        return history[history["market_id"].astype(str).isin({str(m) for m in market_ids})].copy()
    df = _fetch_clob_history(settings, market_ids)
    if ctx is not None and settings.data.get("snapshot_raw", True):
        snapshot(settings, ctx.run_id, "price_history", df)
    return df


def fetch_trade_concentration(settings: Settings, market_ids: list[str],
                              ctx: RunContext | None = None) -> dict[str, dict]:
    """Return {market_id: concentration_record} for the markets we have wallet
    data for. Coverage is PARTIAL by nature — a market with no record simply
    isn't in the returned dict, and Stage 2 treats that as 'unavailable' (never a
    penalty). Records carry available=True plus top-k shares, Gini, and HHI.
    """
    if settings.data_source == "fixtures":
        path = settings.fixtures_dir / "concentration.csv"
        if not path.exists():
            return {}
        wanted = {str(m) for m in market_ids}
        df = pd.read_csv(path)
        fields = ("top1_share", "top3_share", "top5_share", "gini", "hhi", "wallet_count", "trade_count")
        return {
            str(r["market_id"]): {"available": True, **{k: r.get(k) for k in fields}}
            for r in df.to_dict(orient="records") if str(r["market_id"]) in wanted
        }
    return _fetch_live_concentration(settings, market_ids)


def load_fixture(settings: Settings, name: str) -> pd.DataFrame:
    """Load a committed sample table from data/fixtures/<name>.csv."""
    path = settings.fixtures_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Fixture '{name}' not found at {path}. Fixtures ship with the repo; "
            "if you meant to fetch live data, set data.source: live in the config."
        )
    return pd.read_csv(path)


def snapshot(settings: Settings, run_id: str, name: str, df: pd.DataFrame) -> Path:
    """Freeze a raw input under data/snapshots/<run_id>/<name>.csv so the run can
    be reproduced later from exactly the data it saw."""
    snap_dir = settings.snapshots_dir / run_id
    snap_dir.mkdir(parents=True, exist_ok=True)
    path = snap_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Live path — Polymarket public APIs (only used when data.source: live)
# ---------------------------------------------------------------------------
def _fetch_gamma_markets(settings: Settings) -> pd.DataFrame:
    """Page the Gamma /markets endpoint into a normalized frame. Pulls a broad
    universe; relevance filtering is Stage 1's job, not the fetcher's."""
    import requests  # imported lazily so the fixtures path needs no network stack

    cfg = settings.data
    base = str(cfg.get("gamma_base_url", "https://gamma-api.polymarket.com")).rstrip("/")
    max_markets = int(cfg.get("max_markets", 500))
    closed = bool(settings.selector.get("closed_only", True))
    headers = {"User-Agent": str(cfg.get("user_agent", "falnama-research/0.1 (read-only)"))}
    timeout = int(cfg.get("request_timeout_seconds", 30))

    rows: list[dict] = []
    offset, page = 0, min(500, max(50, max_markets))
    while len(rows) < max_markets:
        params = {"closed": str(closed).lower(), "limit": page, "offset": offset}
        resp = requests.get(f"{base}/markets", params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        records = _as_records(resp.json())
        if not records:
            break
        rows.extend(records)
        offset += len(records)
        if len(records) < page:
            break
        time.sleep(0.2)  # be polite to the public API
    normalized = [_normalize_market(r) for r in rows[:max_markets]]
    return pd.DataFrame(normalized, columns=MARKET_COLUMNS)


def _fetch_clob_history(settings: Settings, market_ids: list[str]) -> pd.DataFrame:
    """Fetch price history for each market's first (Yes) CLOB token. Live-only."""
    import requests

    cfg = settings.data
    base = str(cfg.get("clob_base_url", "https://clob.polymarket.com")).rstrip("/")
    timeout = int(cfg.get("request_timeout_seconds", 30))
    lookback_days = int(settings.anomaly.get("price_history_lookback_days", 14))

    # We need each market's Yes token id + name; re-pull the universe to get them.
    universe = _fetch_gamma_markets(settings)
    wanted = universe[universe["market_id"].astype(str).isin({str(m) for m in market_ids})]

    frames: list[pd.DataFrame] = []
    for _, m in wanted.iterrows():
        token_ids = _parse_json(m.get("clob_token_ids"), [])
        if not token_ids:
            continue
        params = {"market": token_ids[0], "interval": "1h", "fidelity": 60}
        resp = requests.get(f"{base}/prices-history", params=params, timeout=timeout)
        resp.raise_for_status()
        points = resp.json().get("history", [])
        if not points:
            continue
        hist = pd.DataFrame(points).rename(columns={"t": "timestamp", "p": "price"})
        hist["timestamp"] = pd.to_datetime(hist["timestamp"], unit="s", utc=True)
        hist["market_id"] = m["market_id"]
        hist["market_name"] = m["market_name"]
        hist["volume"] = m.get("volume")
        hist["liquidity"] = m.get("liquidity")
        hist["close_time"] = m.get("close_time")
        frames.append(hist)
        time.sleep(0.2)
    if not frames:
        return pd.DataFrame(columns=["market_id", "market_name", "timestamp", "price", "volume", "liquidity", "close_time"])
    out = pd.concat(frames, ignore_index=True)
    cutoff = out["timestamp"].max() - pd.Timedelta(days=lookback_days)
    return out[out["timestamp"] >= cutoff].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Wallet concentration — the pure math, plus a best-effort live fetch
# ---------------------------------------------------------------------------
def compute_concentration(trades: pd.DataFrame, wallet_col: str = "wallet",
                          size_col: str = "size") -> dict:
    """Turn a table of trades (a wallet id + a USD size per row) into a
    concentration record: top-k wallet volume shares, Gini, and HHI. Pure math,
    unit-testable without any network call."""
    if trades.empty or wallet_col not in trades or size_col not in trades:
        return {"available": False, "reason": "no trades available for the window"}
    by_wallet = trades.groupby(wallet_col)[size_col].sum().sort_values(ascending=False)
    total = float(by_wallet.sum())
    if total <= 0:
        return {"available": False, "reason": "zero trade volume"}
    shares = by_wallet / total
    return {
        "available": True,
        "top1_share": float(shares.iloc[0]),
        "top3_share": float(shares.iloc[:3].sum()),
        "top5_share": float(shares.iloc[:5].sum()),
        "gini": _gini(by_wallet.to_numpy()),
        "hhi": float((shares ** 2).sum()),
        "wallet_count": int(by_wallet.size),
        "trade_count": int(len(trades)),
    }


def _gini(values) -> float:
    """Gini coefficient of nonnegative values (0 = perfectly even, 1 = one wallet
    holds everything)."""
    v = np.sort(np.asarray(values, dtype=float))
    n = v.size
    if n == 0 or v.sum() == 0:
        return 0.0
    cumulative = np.cumsum(v)
    return float((n + 1 - 2 * (cumulative.sum() / cumulative[-1])) / n)


def _fetch_live_concentration(settings: Settings, market_ids: list[str]) -> dict[str, dict]:
    """Best-effort wallet concentration from the public trades API. Coverage is
    partial and rate-limited, so a failure for any single market is recorded as
    'unavailable' rather than aborting the run. Live-only; fixtures are the tested
    path. First pass measures recent trades; scoping to the exact anomaly window
    is a worthwhile refinement."""
    import requests

    cfg = settings.data
    base = str(cfg.get("data_api_base_url", "https://data-api.polymarket.com")).rstrip("/")
    timeout = int(cfg.get("request_timeout_seconds", 30))
    out: dict[str, dict] = {}
    for market_id in market_ids:
        try:
            resp = requests.get(f"{base}/trades", params={"market": market_id, "limit": 1000}, timeout=timeout)
            resp.raise_for_status()
            raw = resp.json()
            records = raw if isinstance(raw, list) else raw.get("data", [])
            trades = pd.DataFrame([{
                "wallet": t.get("proxyWallet") or t.get("maker") or t.get("taker"),
                "size": float(t.get("size") or t.get("usdcSize") or 0),
            } for t in records if isinstance(t, dict)])
            out[str(market_id)] = compute_concentration(trades)
        except Exception as exc:  # network / rate-limit / parse — record, don't crash
            out[str(market_id)] = {"available": False, "reason": f"trade fetch failed: {exc}"}
        time.sleep(0.2)
    return out


# ---------------------------------------------------------------------------
# Small normalization helpers (shared by the live path)
# ---------------------------------------------------------------------------
def _as_records(payload) -> list[dict]:
    """Gamma returns either a bare list or an object wrapping a list."""
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("data", "markets", "results"):
            if isinstance(payload.get(key), list):
                return [x for x in payload[key] if isinstance(x, dict)]
    return []


def _parse_json(value, default):
    """Polymarket encodes some list fields as JSON strings; decode leniently."""
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return default


def _first(record: dict, keys: list[str], default=None):
    for key in keys:
        if record.get(key) not in (None, ""):
            return record[key]
    return default


def _normalize_market(m: dict) -> dict:
    """Map a raw Gamma market object onto MARKET_COLUMNS."""
    events = _parse_json(m.get("events"), [])
    event = events[0] if events and isinstance(events[0], dict) else {}
    tags = _parse_json(m.get("tags"), [])
    tag_names = "|".join(str(t.get("label") or t.get("name") or t) for t in tags) if isinstance(tags, list) else str(tags)
    slug = _first(m, ["slug", "market_slug"])
    return {
        "market_id": str(_first(m, ["id", "marketId", "market_id"], "")) or None,
        "market_slug": slug,
        "market_name": str(_first(m, ["question", "title", "market_name"], "Unknown market")),
        "question": str(_first(m, ["question", "title"], "")),
        "description": _first(m, ["description"]),
        "category": _first(m, ["category"]),
        "tags": tag_names,
        "event_slug": _first(m, ["eventSlug"], event.get("slug")),
        "event_title": _first(m, ["eventTitle"], event.get("title")),
        "volume": _first(m, ["volume", "volumeNum"]),
        "liquidity": _first(m, ["liquidity", "liquidityNum"]),
        "closed": _first(m, ["closed"]),
        "close_time": _first(m, ["closedTime", "endDate", "end_date"]),
        "clob_token_ids": json.dumps(_parse_json(m.get("clobTokenIds"), [])),
        "market_url": f"https://polymarket.com/market/{slug}" if slug else None,
    }
