"""Stage 2a — wallet fingerprints: WHO placed the bets, not just how the price moved.

WHAT:     For each market in the screened universe, looks at the large long-shot
          bets placed over the recent window and asks, wallet by wallet, whether
          the bettor fits the fingerprint investigative reporting keeps finding
          behind insider-flavored trades: a FRESH wallet, with a NARROW history,
          placing a LARGE bet on an outcome the market thinks is UNLIKELY.
CONSUMES: the screened universe (screen.universe_path) + Polymarket's public,
          keyless data API (trades per market; activity + markets-traded per wallet)
PRODUCES: outputs/wallets/ — wallet_evidence_<run_id>.csv (every wallet examined,
          with the facts behind its verdict) and market_fingerprints_<run_id>.csv
          (one row per market: its fingerprint tier)
REVIEWER: a human deciding which markets and which wallets deserve a closer look
ROLE:     an independent SENSOR beside the price detector. Stage 2 (anomaly) asks
          "did the price move unusually?"; this stage asks "did unusual people
          bet?". The two can disagree, and a disagreement is itself informative.

Why a CLUSTER, not a single wallet, raises the alarm (measured, not assumed):
  Fresh wallets placing long-shot bets are routine — gamblers open accounts to
  chase a headline all the time. Across matched controls (same Iran-strike event
  family, deadlines that resolved NO) we found 0-2 fully-matching wallets per
  market; the Feb-2026 strike market that resolved YES had 8, most staking
  $20K+ each at 11-20 cents a day or two before the strike. A single match is
  therefore NOT evidence on its own — the documented lone Maduro insider (one
  fresh wallet, $32K at 8 cents) looks exactly like the losing gamblers in the
  control. So the tiers are:
      cluster  — several matching wallets on the SAME side → red flag (+bonus)
      watch    — one or two matches → surfaced for human review, no score effect
      none     — examined, nothing matched
      unavailable — no wallet data (never a penalty, same rule as concentration)

The per-wallet test is four plain yes/no facts, each kept on the evidence row so
a reviewer can see exactly why a wallet matched:
    long-shot   bought an outcome at or below longshot_max_price
    large       staked at least min_stake_usd on it inside the window
    fresh       its first-ever Polymarket activity was at most max_wallet_age_days
                before its first bet here
    focused     it has traded at most max_markets_traded markets in its lifetime
Thresholds live in config (the `wallets:` section) and were set against the
backtest cases above — treat them as a first calibration, not settled truth.

Known limits, stated so nobody over-reads the output:
  * `markets_traded` is the wallet's lifetime count AS OF THE FETCH, not as of the
    bet. Live that is the same thing; in a historical replay it can include later
    trading, which only makes a wallet look LESS focused (errs toward no match).
  * The strongest fingerprint in the reporting — funded from a fresh exchange
    withdrawal hours before the bet, drained after — lives on-chain (Polygon USDC
    transfers), not in Polymarket's API. First Polymarket activity is our keyless
    proxy for "funded"; on-chain funding is a natural upgrade seam.
"""

from __future__ import annotations

import time

import pandas as pd

from .config import Settings
from .io import RunContext

# The data API refuses to page past this many trades for one query, so a very busy
# market's window may be truncated to its most recent 10,000 qualifying trades.
_MAX_TRADE_OFFSET = 10000
_PAGE = 500

# Columns of the two output tables — fixed so empty runs still write a header.
EVIDENCE_COLUMNS = [
    "market_id", "market_name", "wallet", "wallet_name", "outcome", "stake_usd",
    "avg_entry_price", "payoff_multiple", "trade_count", "first_bet_utc",
    "wallet_first_activity_utc", "wallet_age_days", "markets_traded",
    "is_longshot", "is_large", "is_fresh", "is_focused", "fingerprint_match",
    "profile_url", "error",
]
MARKET_COLUMNS = [
    "run_id", "market_id", "market_name", "fingerprint_available", "fingerprint_tier",
    "fingerprint_red_flag", "fingerprint_bonus", "matched_wallets", "matched_stake_usd",
    "matched_outcome", "candidate_wallets", "window_trade_usd", "window_start_utc",
    "window_end_utc", "fingerprint_reason",
]


# ---------------------------------------------------------------------------
# The fingerprint itself (pure, unit-testable — no network)
# ---------------------------------------------------------------------------
def candidate_bets(trades: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Collapse raw trades into one row per (wallet, outcome) long-shot position,
    keeping only positions large enough to matter.

    This is the cheap, wide filter: it runs on every trade we pulled, so the
    expensive per-wallet lookups that follow only happen for the few wallets that
    made a big bet on an unlikely outcome. A trade's `price` is the price of the
    outcome it bought, so "long-shot" works the same for a Yes or a No buyer.
    """
    cols = ["wallet", "wallet_name", "outcome", "stake_usd", "avg_entry_price",
            "trade_count", "first_bet_utc"]
    if trades.empty:
        return pd.DataFrame(columns=cols)
    max_price = float(cfg.get("longshot_max_price", 0.35))
    buys = trades[(trades["side"].str.upper() == "BUY") & (trades["price"] <= max_price)].copy()
    if buys.empty:
        return pd.DataFrame(columns=cols)
    buys["usd"] = buys["size"] * buys["price"]
    grouped = buys.groupby(["wallet", "outcome"], sort=False).agg(
        wallet_name=("wallet_name", "first"), stake_usd=("usd", "sum"),
        shares=("size", "sum"), trade_count=("usd", "size"), first_bet_utc=("timestamp", "min"),
    ).reset_index()
    # Dollar-weighted entry price: what the wallet actually paid per share overall.
    grouped["avg_entry_price"] = grouped["stake_usd"] / grouped["shares"]
    large = grouped[grouped["stake_usd"] >= float(cfg.get("min_stake_usd", 10000))]
    top = large.sort_values("stake_usd", ascending=False).head(int(cfg.get("max_candidates_per_market", 25)))
    return top[cols].reset_index(drop=True)


def fingerprint_wallet(bet: dict, profile: dict, cfg: dict) -> dict:
    """Judge one candidate position against the four-part fingerprint.

    `bet` is a row from candidate_bets; `profile` holds the wallet's first-ever
    activity time and lifetime markets-traded count (either may be missing, in
    which case that test simply fails — an unknown age is never read as fresh).
    """
    first_bet = pd.to_datetime(bet["first_bet_utc"], utc=True)
    first_seen = pd.to_datetime(profile.get("first_activity_utc"), utc=True, errors="coerce")
    age_days = (first_bet - first_seen).total_seconds() / 86400.0 if pd.notna(first_seen) else None
    traded = profile.get("markets_traded")
    traded = int(traded) if traded is not None and pd.notna(traded) else None
    price = float(bet["avg_entry_price"])

    is_longshot = price <= float(cfg.get("longshot_max_price", 0.35))
    is_large = float(bet["stake_usd"]) >= float(cfg.get("min_stake_usd", 10000))
    is_fresh = age_days is not None and age_days <= float(cfg.get("max_wallet_age_days", 14))
    is_focused = traded is not None and traded <= int(cfg.get("max_markets_traded", 25))

    return {
        "wallet": bet["wallet"], "wallet_name": bet.get("wallet_name"), "outcome": bet["outcome"],
        "stake_usd": round(float(bet["stake_usd"]), 2),
        "avg_entry_price": round(price, 4),
        "payoff_multiple": round(1.0 / price, 1) if price > 0 else None,  # $ back per $ if it wins
        "trade_count": int(bet["trade_count"]),
        "first_bet_utc": first_bet.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wallet_first_activity_utc": first_seen.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(first_seen) else None,
        "wallet_age_days": round(age_days, 2) if age_days is not None else None,
        "markets_traded": traded,
        "is_longshot": is_longshot, "is_large": is_large, "is_fresh": is_fresh, "is_focused": is_focused,
        "fingerprint_match": bool(is_longshot and is_large and is_fresh and is_focused),
        "profile_url": f"https://polymarket.com/profile/{bet['wallet']}",
        "error": profile.get("error", ""),
    }


def summarize_market(evidence: pd.DataFrame, cfg: dict) -> dict:
    """Turn a market's wallet evidence into its tier (see the module docstring).

    Matches are counted PER OUTCOME: the alarm is several fresh wallets piling
    onto the SAME unlikely side, not fresh wallets scattered across both."""
    matched = evidence[evidence["fingerprint_match"]] if not evidence.empty else evidence
    if matched.empty:
        return {"fingerprint_tier": "none", "fingerprint_red_flag": False, "fingerprint_bonus": 0.0,
                "matched_wallets": 0, "matched_stake_usd": 0.0, "matched_outcome": None,
                "fingerprint_reason": "no wallet fit the fresh + focused + large long-shot fingerprint"}
    by_side = matched.groupby("outcome")["stake_usd"].agg(["size", "sum"]).sort_values(["size", "sum"], ascending=False)
    side, count, stake = by_side.index[0], int(by_side.iloc[0]["size"]), float(by_side.iloc[0]["sum"])
    if count >= int(cfg.get("min_cluster_wallets", 3)):
        tier, flag, bonus = "cluster", True, float(cfg.get("cluster_bonus", 15))
        reason = f"{count} fresh, focused wallets staked ${stake:,.0f} on long-shot {side!r}"
    else:
        tier, flag, bonus = "watch", False, 0.0
        reason = (f"{count} fingerprinted wallet(s) on {side!r} (${stake:,.0f}) — below the cluster "
                  "threshold; a lone match also looks like an ordinary gambler, so review by hand")
    return {"fingerprint_tier": tier, "fingerprint_red_flag": flag, "fingerprint_bonus": bonus,
            "matched_wallets": count, "matched_stake_usd": round(stake, 2), "matched_outcome": side,
            "fingerprint_reason": reason}


# ---------------------------------------------------------------------------
# One market, and the whole universe
# ---------------------------------------------------------------------------
def fingerprint_market(settings: Settings, market: dict, window_end: pd.Timestamp | None,
                       profile_cache: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Fingerprint one market's recent bettors. Returns (evidence rows, market record).

    `window_end` is the moment we look back from — the run time live, or the
    settlement onset in a historical replay; None means "use every trade the
    source has" (the fixtures, which are already a frozen window). Any failure is
    recorded on the market record as 'unavailable' and never raised, so one
    stubborn market cannot sink the run.
    """
    cfg = settings.section("wallets")
    profile_cache = {} if profile_cache is None else profile_cache
    hours = float(cfg.get("lookback_hours", 168))
    start = window_end - pd.Timedelta(hours=hours) if window_end is not None else None
    record = {
        "market_id": market.get("market_id"), "market_name": market.get("market_name"),
        "fingerprint_available": False, "fingerprint_tier": "unavailable",
        "fingerprint_red_flag": False, "fingerprint_bonus": 0.0, "matched_wallets": 0,
        "matched_stake_usd": 0.0, "matched_outcome": None, "candidate_wallets": 0,
        "window_trade_usd": None,
        "window_start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ") if start is not None else None,
        "window_end_utc": window_end.strftime("%Y-%m-%dT%H:%M:%SZ") if window_end is not None else None,
        "fingerprint_reason": "",
    }
    try:
        trades = _fetch_trades(settings, market, start, window_end)
        record["window_trade_usd"] = round(float((trades["size"] * trades["price"]).sum()), 2) if not trades.empty else 0.0
        candidates = candidate_bets(trades, cfg)
        rows = []
        for bet in candidates.to_dict(orient="records"):
            if bet["wallet"] not in profile_cache:
                profile_cache[bet["wallet"]] = _fetch_profile(settings, bet["wallet"])
            rows.append(fingerprint_wallet(bet, profile_cache[bet["wallet"]], cfg))
        evidence = pd.DataFrame(rows, columns=[c for c in EVIDENCE_COLUMNS if c not in ("market_id", "market_name")])
        evidence.insert(0, "market_name", market.get("market_name"))
        evidence.insert(0, "market_id", market.get("market_id"))
        record.update({"fingerprint_available": True, "candidate_wallets": int(len(evidence)),
                       **summarize_market(evidence, cfg)})
        return evidence, record
    except Exception as exc:  # network / rate limit / missing id — recorded, never fatal
        record["fingerprint_reason"] = f"wallet data unavailable: {type(exc).__name__}: {exc}"
        return pd.DataFrame(columns=EVIDENCE_COLUMNS), record


def fingerprint_universe(settings: Settings, markets: pd.DataFrame,
                         window_end: pd.Timestamp | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fingerprint every market. Wallet lookups are cached across markets: an
    active bettor often appears in several markets of the same event family."""
    cache: dict = {}
    evidence, records = [], []
    for market in markets.to_dict(orient="records"):
        ev, rec = fingerprint_market(settings, market, window_end, cache)
        if not ev.empty:
            evidence.append(ev)
        records.append(rec)
    evidence_df = pd.concat(evidence, ignore_index=True) if evidence else pd.DataFrame(columns=EVIDENCE_COLUMNS)
    return evidence_df, pd.DataFrame(records, columns=[c for c in MARKET_COLUMNS if c != "run_id"])


# ---------------------------------------------------------------------------
# Data access — fixtures or Polymarket's public data API (isolated for mocking)
# ---------------------------------------------------------------------------
def _fetch_trades(settings: Settings, market: dict, start: pd.Timestamp | None,
                  end: pd.Timestamp | None) -> pd.DataFrame:
    """Trades of at least `min_trade_usd` in [start, end] for one market, as
    columns wallet, wallet_name, side, outcome, price, size, timestamp."""
    if settings.data_source == "fixtures":
        path = settings.fixtures_dir / "wallet_trades.csv"
        df = pd.read_csv(path) if path.exists() else pd.DataFrame()
        if df.empty:
            return df
        return df[df["market_id"].astype(str) == str(market.get("market_id"))].reset_index(drop=True)

    import requests

    condition = market.get("condition_id")
    if not isinstance(condition, str) or not condition.startswith("0x"):
        # The data API keys markets by their on-chain conditionId; Gamma's numeric
        # market id silently returns nothing, so refuse rather than report "none".
        raise LookupError("market has no conditionId")
    cfg, data = settings.section("wallets"), settings.data
    base = str(data.get("data_api_base_url", "https://data-api.polymarket.com")).rstrip("/")
    params = {"market": condition, "limit": _PAGE, "filterType": "CASH",
              "filterAmount": float(cfg.get("min_trade_usd", 500))}
    if start is not None:
        params["start"] = int(start.timestamp())
    if end is not None:
        params["end"] = int(end.timestamp())
    rows: list[dict] = []
    offset = 0
    while offset < _MAX_TRADE_OFFSET:
        resp = requests.get(f"{base}/trades", params={**params, "offset": offset},
                            timeout=int(data.get("request_timeout_seconds", 30)))
        resp.raise_for_status()
        page = resp.json()
        if not isinstance(page, list) or not page:
            break
        rows.extend(page)
        offset += len(page)
        if len(page) < _PAGE:
            break
        time.sleep(0.1)  # be polite to the public API
    return pd.DataFrame([{
        "wallet": t.get("proxyWallet"), "wallet_name": t.get("name") or t.get("pseudonym"),
        "side": t.get("side", ""), "outcome": t.get("outcome"),
        "price": float(t.get("price") or 0), "size": float(t.get("size") or 0),
        "timestamp": pd.Timestamp(int(t["timestamp"]), unit="s", tz="UTC"),
    } for t in rows if t.get("proxyWallet") and t.get("timestamp")],
        columns=["wallet", "wallet_name", "side", "outcome", "price", "size", "timestamp"])


def _fetch_profile(settings: Settings, wallet: str) -> dict:
    """A wallet's first-ever Polymarket activity and lifetime markets-traded count.
    Returns what it could get; a failure is recorded on the profile, not raised,
    so one wallet cannot knock out its whole market."""
    if settings.data_source == "fixtures":
        path = settings.fixtures_dir / "wallet_profiles.csv"
        df = pd.read_csv(path) if path.exists() else pd.DataFrame()
        hit = df[df["wallet"] == wallet] if not df.empty else df
        return hit.iloc[0].to_dict() if not hit.empty else {"error": "no profile in fixtures"}

    import requests

    data = settings.data
    base = str(data.get("data_api_base_url", "https://data-api.polymarket.com")).rstrip("/")
    timeout = int(data.get("request_timeout_seconds", 30))
    profile: dict = {}
    try:
        first = requests.get(f"{base}/activity", params={"user": wallet, "limit": 1, "sortBy": "TIMESTAMP",
                                                         "sortDirection": "ASC"}, timeout=timeout).json()
        if isinstance(first, list) and first:
            profile["first_activity_utc"] = pd.Timestamp(int(first[0]["timestamp"]), unit="s", tz="UTC")
        traded = requests.get(f"{base}/traded", params={"user": wallet}, timeout=timeout).json()
        if isinstance(traded, dict) and traded.get("traded") is not None:
            profile["markets_traded"] = int(traded["traded"])
    except Exception as exc:
        profile["error"] = f"profile fetch failed: {type(exc).__name__}: {exc}"
    time.sleep(0.1)
    return profile


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def market_fingerprints_path(settings: Settings):
    """Where Stage 2 (anomaly) finds this stage's per-market verdicts."""
    return settings.output_dir("wallets") / "market_fingerprints_latest.csv"


def run(ctx: RunContext) -> pd.DataFrame:
    """Fingerprint the screened universe, write the evidence, record the manifest.

    When disabled, writes an empty verdict table for this run so Stage 2 can never
    pick up a stale verdict from an earlier run."""
    from . import io, screen

    settings = ctx.settings
    cfg = settings.section("wallets")
    out_dir = settings.output_dir("wallets")
    if not cfg.get("enabled", False):
        io.write_table(pd.DataFrame(columns=MARKET_COLUMNS), out_dir, "market_fingerprints", ctx.run_id)
        io.update_manifest(ctx, "wallet_fingerprints", {"enabled": False})
        return pd.DataFrame(columns=MARKET_COLUMNS)

    markets = io.read_table(screen.universe_path(settings))
    # Live: look back from now. Fixtures: the committed trades ARE the window.
    window_end = None if settings.data_source == "fixtures" else pd.Timestamp(ctx.run_time_utc)
    evidence, table = fingerprint_universe(settings, markets, window_end)
    table.insert(0, "run_id", ctx.run_id)

    io.write_table(evidence, out_dir, "wallet_evidence", ctx.run_id, also_latest=False)
    path = io.write_table(table, out_dir, "market_fingerprints", ctx.run_id)
    tiers = table["fingerprint_tier"].value_counts().to_dict() if not table.empty else {}
    io.update_manifest(ctx, "wallet_fingerprints", {
        "enabled": True,
        "markets_examined": int(len(table)),
        "wallets_examined": int(len(evidence)),
        "wallets_matched": int(evidence["fingerprint_match"].sum()) if not evidence.empty else 0,
        "tiers": {str(k): int(v) for k, v in tiers.items()},
        "market_fingerprints_file": str(path.relative_to(settings.project_root)),
    })
    return table
