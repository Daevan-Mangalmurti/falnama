"""Stage 2a wallet fingerprints: the four-part per-wallet test, the same-side
cluster rule, fail-safe behavior on missing data, and the hand-off to Stage 2."""

from __future__ import annotations

import pandas as pd
import pytest

from falnama import wallets
from falnama.anomaly import fingerprint_overlay
from falnama.config import load_config

S = load_config()
S.raw["data"]["source"] = "fixtures"  # never touch the network from unit tests
CFG = {
    "longshot_max_price": 0.35, "min_stake_usd": 10000, "max_wallet_age_days": 14,
    "max_markets_traded": 25, "max_candidates_per_market": 25,
    "min_cluster_wallets": 3, "cluster_bonus": 15,
}
S.raw["wallets"] = {**CFG, "enabled": True, "lookback_hours": 168, "min_trade_usd": 500}


def _trades(rows):
    return pd.DataFrame(rows, columns=["wallet", "wallet_name", "side", "outcome", "price", "size", "timestamp"])


def _bet(**over):
    bet = {"wallet": "0xabc", "wallet_name": "w", "outcome": "Yes", "stake_usd": 20000.0,
           "avg_entry_price": 0.12, "trade_count": 2, "first_bet_utc": "2026-02-27T00:00:00Z"}
    return {**bet, **over}


FRESH = {"first_activity_utc": "2026-02-25T00:00:00Z", "markets_traded": 4}


# ---- candidate selection: the cheap, wide filter ---------------------------
def test_candidates_keep_only_large_longshot_buys():
    t = _trades([
        ("A", "a", "BUY", "Yes", 0.10, 200000, "2026-02-27T01:00:00Z"),  # $20K long-shot → keep
        ("B", "b", "BUY", "Yes", 0.90, 50000, "2026-02-27T02:00:00Z"),   # buying the favorite → drop
        ("C", "c", "SELL", "No", 0.10, 500000, "2026-02-27T03:00:00Z"),  # a sale, not a bet → drop
        ("D", "d", "BUY", "Yes", 0.10, 20000, "2026-02-27T04:00:00Z"),   # $2K → too small
    ])
    c = wallets.candidate_bets(t, CFG)
    assert list(c["wallet"]) == ["A"]
    assert c.iloc[0]["stake_usd"] == pytest.approx(20000)


def test_candidates_sum_a_wallets_bets_and_weight_the_entry_price():
    t = _trades([
        ("A", "a", "BUY", "Yes", 0.10, 100000, "2026-02-27T05:00:00Z"),  # $10K
        ("A", "a", "BUY", "Yes", 0.20, 50000, "2026-02-27T01:00:00Z"),   # $10K
    ])
    row = wallets.candidate_bets(t, CFG).iloc[0]
    assert row["stake_usd"] == pytest.approx(20000)
    assert row["avg_entry_price"] == pytest.approx(20000 / 150000)  # $ paid per share overall
    assert str(row["first_bet_utc"]) == "2026-02-27T01:00:00Z"
    assert row["trade_count"] == 2


# ---- the per-wallet fingerprint --------------------------------------------
def test_fresh_focused_large_longshot_matches():
    ev = wallets.fingerprint_wallet(_bet(), FRESH, CFG)
    assert ev["fingerprint_match"] is True
    assert ev["wallet_age_days"] == pytest.approx(2.0)
    assert ev["payoff_multiple"] == pytest.approx(8.3)


@pytest.mark.parametrize("bet_over, profile, failing", [
    ({}, {"first_activity_utc": "2025-01-01T00:00:00Z", "markets_traded": 4}, "is_fresh"),
    ({}, {"first_activity_utc": "2026-02-25T00:00:00Z", "markets_traded": 900}, "is_focused"),
    ({"avg_entry_price": 0.60}, FRESH, "is_longshot"),
    ({"stake_usd": 3000.0}, FRESH, "is_large"),
])
def test_each_fact_is_required(bet_over, profile, failing):
    ev = wallets.fingerprint_wallet(_bet(**bet_over), profile, CFG)
    assert ev[failing] is False and ev["fingerprint_match"] is False


def test_unknown_history_is_never_read_as_fresh_or_focused():
    ev = wallets.fingerprint_wallet(_bet(), {"error": "profile fetch failed"}, CFG)
    assert ev["is_fresh"] is False and ev["is_focused"] is False
    assert ev["fingerprint_match"] is False and "failed" in ev["error"]


# ---- market tiers: a cluster needs several wallets on the SAME side --------
def _evidence(sides_matched):
    return pd.DataFrame([{"outcome": side, "stake_usd": 20000.0, "fingerprint_match": m}
                         for side, m in sides_matched])


def test_three_matches_on_one_side_is_a_cluster():
    s = wallets.summarize_market(_evidence([("Yes", True)] * 3 + [("Yes", False)]), CFG)
    assert s["fingerprint_tier"] == "cluster" and s["fingerprint_red_flag"] is True
    assert s["fingerprint_bonus"] == 15 and s["matched_wallets"] == 3 and s["matched_outcome"] == "Yes"


def test_matches_split_across_sides_are_only_a_watch():
    s = wallets.summarize_market(_evidence([("Yes", True), ("Yes", True), ("No", True)]), CFG)
    assert s["fingerprint_tier"] == "watch" and s["fingerprint_red_flag"] is False
    assert s["fingerprint_bonus"] == 0 and s["matched_wallets"] == 2


def test_a_lone_match_is_a_watch_and_no_match_is_none():
    assert wallets.summarize_market(_evidence([("Yes", True)]), CFG)["fingerprint_tier"] == "watch"
    assert wallets.summarize_market(_evidence([("Yes", False)]), CFG)["fingerprint_tier"] == "none"
    assert wallets.summarize_market(pd.DataFrame(columns=["outcome", "stake_usd", "fingerprint_match"]),
                                    CFG)["fingerprint_tier"] == "none"


# ---- fail-safe data handling ----------------------------------------------
def test_a_fetch_failure_is_unavailable_not_none(monkeypatch):
    def boom(*a, **k):
        raise TimeoutError("data API timed out")
    monkeypatch.setattr(wallets, "_fetch_trades", boom)
    ev, rec = wallets.fingerprint_market(S, {"market_id": "m1", "market_name": "M"}, None)
    assert ev.empty
    assert rec["fingerprint_tier"] == "unavailable" and rec["fingerprint_available"] is False
    assert "timed out" in rec["fingerprint_reason"]


def test_live_path_refuses_a_numeric_market_id():
    """The data API silently returns [] for Gamma's numeric id; that must read as
    'unavailable', never as a clean 'none' (the bug that kept concentration dark)."""
    live = load_config()
    live.raw["data"]["source"] = "live"
    live.raw["wallets"] = S.raw["wallets"]
    _, rec = wallets.fingerprint_market(live, {"market_id": "540843", "condition_id": None},
                                        pd.Timestamp("2026-02-28T00:00:00Z"))
    assert rec["fingerprint_tier"] == "unavailable" and "conditionId" in rec["fingerprint_reason"]


class _Resp:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def _page(rows, cursor=None):
    return _Resp({"data": rows, "pagination": {"limit": 1000, "offset": 0, "has_more": cursor is not None,
                                               "next_cursor": cursor}})


def _live_settings():
    live = load_config()
    live.raw["data"]["source"] = "live"
    live.raw["wallets"] = S.raw["wallets"]
    return live


def test_live_trade_fetch_follows_cursors_and_windows(monkeypatch):
    """Data API v2 pages by cursor and ignores start/end for a market query, so
    the window is applied here: newest-first pages, trades after `end` dropped,
    and paging stops at the first page that reaches back before `start`."""
    end = pd.Timestamp("2026-02-28T00:00:00Z")
    lo, hi = int((end - pd.Timedelta(hours=168)).timestamp()), int(end.timestamp())
    pages = {
        None: _page([_trade(i, hi + 100) for i in range(2)] + [_trade(i, hi - 1000) for i in range(998)], "c1"),
        "c1": _page([_trade(i, lo + 10) for i in range(5)] + [_trade(i, lo - 10) for i in range(3)], "c2"),
        "c2": _page([_trade(0, lo - 99)]),  # must never be fetched
    }
    calls = []

    def fake_get(url, params, **kw):
        calls.append((url, dict(params)))
        return pages[params.get("cursor")]

    import requests
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(wallets.polymarket.time, "sleep", lambda s: None)
    df = wallets._fetch_trades(_live_settings(), {"condition_id": "0xdead"}, end - pd.Timedelta(hours=168), end)
    assert len(df) == 998 + 5 and len(calls) == 2
    url, first = calls[0]
    assert url == "https://data-api.polymarket.com/v2/trades"
    assert first["condition"] == "0xdead" and first["filter_type"] == "CASH" and first["filter_amount"] == 500
    assert not {"market", "offset", "start", "end"} & set(first)  # v2 rejects market/offset, ignores start/end
    assert calls[1][1]["cursor"] == "c1"
    assert df["timestamp"].between(pd.Timestamp(lo, unit="s", tz="UTC"), end).all()


def _trade(i, ts):
    return {"proxy_wallet": f"0x{i}", "side": "BUY", "outcome": "Yes", "price": 0.1, "size": 10000,
            "timestamp": ts, "name": f"w{i}"}


def test_live_profile_uses_v2_activity_and_user_stats(monkeypatch):
    calls = {}

    def fake_get(url, params, **kw):
        route = url.rsplit("/v2/", 1)[1]
        calls[route] = params
        if route == "activity":
            return _page([{"timestamp": 1772000000, "type": "TRADE"}])
        return _Resp({"data": {"proxy_wallet": params["user"], "trades": 7, "join_date": None}})

    import requests
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(wallets.time, "sleep", lambda s: None)
    profile = wallets._fetch_profile(_live_settings(), "0xabc")
    assert profile == {"first_activity_utc": pd.Timestamp(1772000000, unit="s", tz="UTC"), "markets_traded": 7}
    assert calls["activity"]["sort_direction"] == "ASC" and calls["activity"]["start"] == 1  # full history
    assert set(calls) == {"activity", "user-stats"}


def test_a_retired_profile_route_is_an_error_not_a_blank(monkeypatch):
    """v1 answered 410 with an error body; reading it as JSON used to leave the
    profile silently empty. A failure must be recorded on the profile."""
    import requests

    class Gone:
        status_code = 410

        def raise_for_status(self):
            raise requests.HTTPError("410 Client Error: Gone")

    monkeypatch.setattr(requests, "get", lambda url, **kw: Gone())
    monkeypatch.setattr(wallets.time, "sleep", lambda s: None)
    profile = wallets._fetch_profile(_live_settings(), "0xabc")
    assert "410" in profile["error"] and "markets_traded" not in profile


def test_stage_warns_when_most_markets_are_dark(monkeypatch, tmp_path):
    import copy

    from falnama import screen
    from falnama.config import Settings
    from falnama.io import RunContext

    settings = Settings(raw=copy.deepcopy(S.raw), project_root=tmp_path)
    universe = tmp_path / "universe.csv"
    pd.DataFrame({"market_id": ["m1", "m2", "m3"], "market_name": ["A", "B", "C"]}).to_csv(universe, index=False)
    monkeypatch.setattr(screen, "universe_path", lambda s: universe)

    def boom(*a, **k):
        raise ConnectionError("data API down")
    monkeypatch.setattr(wallets, "_fetch_trades", boom)
    ctx = RunContext.start(settings)
    table = wallets.run(ctx)
    assert (table["fingerprint_tier"] == "unavailable").all()
    assert len(ctx.warnings) == 1 and "wallet fingerprints available for only 0 of 3" in ctx.warnings[0]
    assert "data API down" in ctx.warnings[0]


# ---- the fixture universe exercises every tier -----------------------------
def test_fixture_universe_tiers():
    markets = pd.read_csv(S.fixtures_dir / "markets.csv")
    _, table = wallets.fingerprint_universe(S, markets.head(6), None)
    tiers = dict(zip(table["market_id"], table["fingerprint_tier"]))
    assert tiers["mil-iran-strike"] == "cluster"   # 4 fresh wallets on Yes; the veteran whale doesn't count
    assert tiers["taiwan-blockade"] == "watch"     # a lone fresh wallet
    assert tiers["trade-china-evs"] == "watch"     # fresh wallets, but split 2 Yes / 1 No
    assert tiers["sanc-russia-q3"] == "none"       # big long-shot bets from established wallets


# ---- the hand-off to Stage 2 ------------------------------------------------
def test_overlay_adds_the_bonus_only_for_a_cluster():
    assert fingerprint_overlay({"fingerprint_tier": "cluster", "fingerprint_bonus": 15,
                                "matched_wallets": 4})["fingerprint_bonus"] == 15
    for tier in ("watch", "none", "unavailable"):
        o = fingerprint_overlay({"fingerprint_tier": tier, "fingerprint_bonus": 15})
        assert o["fingerprint_bonus"] == 0 and o["fingerprint_red_flag"] is False
    assert fingerprint_overlay(None)["fingerprint_tier"] == "unavailable"
