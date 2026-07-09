"""Stage 2: the anomaly scorer flags a sharp, persistent jump and stays calm
on gentle noise. Also checks the too-little-data guard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from falnama.config import load_config
from falnama.anomaly import score_market

SETTINGS = load_config()


def _series(prices: list[float], close_in_hours: int | None = None) -> pd.DataFrame:
    start = pd.Timestamp("2026-06-01T00:00:00Z")
    times = pd.date_range(start, periods=len(prices), freq="1h")
    close = (times[-1] + pd.Timedelta(hours=close_in_hours)) if close_in_hours else None
    return pd.DataFrame({
        "market_id": "m", "market_name": "test market",
        "timestamp": [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in times],
        "price": prices,
        "close_time": close.strftime("%Y-%m-%dT%H:%M:%SZ") if close is not None else None,
    })


def test_sharp_jump_scores_strong():
    # 40 calm hours near 0.15, then a fast +0.4 jump that holds.
    calm = list(0.15 + np.zeros(40))
    jumped = list(0.55 + np.zeros(20))
    result = score_market(_series(calm + jumped), SETTINGS)
    assert result is not None
    assert result["score_magnitude"] >= 90
    assert result["anomaly_score"] >= SETTINGS.anomaly["strong_threshold"]


def test_calm_market_scores_low():
    rng = np.random.default_rng(0)
    prices = list(np.clip(0.4 + np.cumsum(rng.normal(0, 0.003, 60)), 0.05, 0.95))
    result = score_market(_series(prices), SETTINGS)
    assert result is not None
    assert result["anomaly_score"] < SETTINGS.anomaly["weak_threshold"]


def test_too_little_history_returns_none():
    assert score_market(_series([0.2, 0.3, 0.4]), SETTINGS) is None


def test_time_to_close_bonus_applies_near_resolution():
    calm = list(0.15 + np.zeros(40))
    jumped = list(0.45 + np.zeros(10))
    near = score_market(_series(calm + jumped, close_in_hours=6), SETTINGS)
    far = score_market(_series(calm + jumped, close_in_hours=1000), SETTINGS)
    assert near["time_to_close_bonus"] > far["time_to_close_bonus"]
