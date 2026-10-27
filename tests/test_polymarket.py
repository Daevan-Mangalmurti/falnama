"""The live Polymarket fetches: Gamma paging, and surviving flaky requests.

This guards the pagination bug that silently truncated every live run to 100
markets: the Gamma /markets endpoint caps a page at 100 rows no matter what limit
we request, so the old "a short page means the last page" check fired on the very
first page. These tests use a fake Gamma that reproduces the cap, injected in
place of the lazily-imported `requests`, so they run offline.
"""

from __future__ import annotations

import sys

import pandas as pd

from falnama import polymarket
from falnama.config import load_config


class _FakeResponse:
    status_code = 200

    def __init__(self, rows: list[dict]):
        self._rows = rows

    def raise_for_status(self):
        pass

    def json(self):
        return self._rows


class _RequestsStub:
    """Stands in for the `requests` module: `.get` mimics Gamma, honouring offset
    but never returning more than 100 rows per page, whatever `limit` is asked."""

    def __init__(self, pool_size: int):
        self._pool = [{"id": i, "question": f"Market {i}"} for i in range(pool_size)]

    def get(self, url, params=None, headers=None, timeout=None):
        params = params or {}
        offset = int(params.get("offset", 0))
        limit = min(int(params.get("limit", 100)), 100)  # the API's hard ceiling
        return _FakeResponse(self._pool[offset:offset + limit])


def _use_fake_gamma(monkeypatch, pool_size: int):
    # The fetch does `import requests` lazily, so the stub must live in sys.modules
    # for that import statement to pick it up.
    monkeypatch.setitem(sys.modules, "requests", _RequestsStub(pool_size))


def _settings(max_markets: int):
    s = load_config()
    s.raw["data"]["source"] = "live"
    s.raw["data"]["max_markets"] = max_markets
    return s


def test_fetch_pages_past_the_hundred_row_ceiling(monkeypatch):
    # 250 wanted, pages capped at 100 → must make 3 requests, not stop at 100.
    # (Under the old bug this returned exactly 100.)
    _use_fake_gamma(monkeypatch, 500)
    markets = polymarket._fetch_gamma_markets(_settings(250))
    assert isinstance(markets, pd.DataFrame)
    assert len(markets) == 250


def test_fetch_stops_when_the_pool_is_exhausted(monkeypatch):
    # Only 140 markets exist but 500 wanted → return all 140 and stop cleanly,
    # rather than looping forever on empty pages.
    _use_fake_gamma(monkeypatch, 140)
    markets = polymarket._fetch_gamma_markets(_settings(500))
    assert len(markets) == 140


# ---------------------------------------------------------------------------
# Flaky network: retries, per-market skips, and the outage breaker
# ---------------------------------------------------------------------------
class _Timeout(Exception):
    pass


class _FlakyClob:
    """Stands in for `requests` against CLOB /prices-history. `plan` maps a token
    id to how many calls time out before it answers (None = never answers)."""

    Timeout = _Timeout
    ConnectionError = ConnectionError

    def __init__(self, plan: dict):
        self.plan = dict(plan)
        self.calls: dict[str, int] = {}

    def get(self, url, params=None, headers=None, timeout=None):
        token = params["market"]
        self.calls[token] = self.calls.get(token, 0) + 1
        fails = self.plan.get(token, 0)
        if fails is None or self.calls[token] <= fails:
            raise _Timeout("The read operation timed out")
        return _FakeResponse({"history": [{"t": 1_761_000_000, "p": 0.4}]})


def _universe(*tokens):
    return pd.DataFrame([{"market_id": str(i), "market_name": f"M{i}", "clob_token_ids": f'["{t}"]',
                          "volume": 1.0, "liquidity": 1.0, "close_time": None}
                         for i, t in enumerate(tokens)])


def _flaky(monkeypatch, plan):
    stub = _FlakyClob(plan)
    monkeypatch.setitem(sys.modules, "requests", stub)
    monkeypatch.setattr(polymarket.time, "sleep", lambda s: None)  # no real backoff waits
    return stub


def test_a_timeout_is_retried_then_succeeds(monkeypatch):
    stub = _flaky(monkeypatch, {"a": 1})  # first call times out, second answers
    history, failures = polymarket._fetch_clob_history(_settings(10), ["0"], _universe("a"))
    assert failures == [] and len(history) == 1 and stub.calls["a"] == 2


def test_a_market_that_never_answers_is_skipped_not_fatal(monkeypatch):
    # The 2026-10-27 failure: one CLOB read timeout used to end the whole run.
    stub = _flaky(monkeypatch, {"b": None})
    history, failures = polymarket._fetch_clob_history(_settings(10), ["0", "1", "2"], _universe("a", "b", "c"))
    assert sorted(history["market_id"]) == ["0", "2"]
    assert [f["market_id"] for f in failures] == ["1"] and "timed out" in failures[0]["error"]
    assert stub.calls["b"] == 3  # the first try plus max_retries (2)


def test_several_failures_in_a_row_are_an_outage(monkeypatch):
    import pytest
    _flaky(monkeypatch, {"a": None, "b": None, "c": None, "d": 0})
    with pytest.raises(RuntimeError, match="outage"):
        polymarket._fetch_clob_history(_settings(10), ["0", "1", "2", "3"], _universe("a", "b", "c", "d"))


def test_the_passed_universe_avoids_a_second_gamma_pull(monkeypatch):
    stub = _flaky(monkeypatch, {})
    monkeypatch.setattr(polymarket, "_fetch_gamma_markets",
                        lambda s: (_ for _ in ()).throw(AssertionError("re-pulled Gamma")))
    history, _ = polymarket._fetch_clob_history(_settings(10), ["0"], _universe("a"))
    assert len(history) == 1 and stub.calls == {"a": 1}
