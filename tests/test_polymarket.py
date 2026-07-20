"""The live Gamma fetch pages correctly past the API's 100-row ceiling.

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
