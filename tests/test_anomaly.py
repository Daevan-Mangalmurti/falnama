"""Stage 2: the anomaly scorer flags a sharp, persistent jump and stays calm
on gentle noise. Also checks the too-little-data guard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from falnama.config import load_config
from falnama.anomaly import UNUSUALNESS_VOL_FLOOR, _unusualness, score_market

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


def test_persistence_is_none_when_the_move_is_the_latest_observation():
    # A jump on the FINAL observation has no aftermath to observe, so persistence
    # is undetermined (None) rather than a fabricated 100, and the composite is
    # renormalized over the components we could actually measure.
    result = score_market(_series([0.20] * 55 + [0.40]), SETTINGS)
    assert result is not None
    assert result["score_persistence"] is None
    assert 0.0 <= result["anomaly_score"] <= 100.0          # finite, not NaN
    # A fresh, sharp move should still score — not be penalized for being recent.
    assert result["anomaly_score"] >= SETTINGS.anomaly["weak_threshold"]


def _prices_to_series(prices: list[float]) -> pd.Series:
    idx = pd.date_range("2026-06-01T00:00:00Z", periods=len(prices), freq="1h")
    return pd.Series(prices, index=idx, dtype=float)


def test_unusualness_does_not_saturate_on_a_mostly_flat_market():
    # The bug this guards: a market flat ~90% of the time with tiny ticks used to
    # score 100 because the std (dominated by zeros) was ~0. It should now be
    # modest — a big-for-this-market move, not an off-the-charts one.
    prices = [0.30] * 90 + [0.31, 0.31, 0.32, 0.31, 0.32] * 2  # long flat, then tiny ticks
    u = _unusualness(_prices_to_series(prices))
    assert 0.0 < u < 60.0  # discriminates, nowhere near the old saturated 100


def test_unusualness_is_zero_for_a_never_moving_market():
    assert _unusualness(_prices_to_series([0.25] * 50)) == 0.0


def test_unusualness_survives_a_single_step_market_via_the_floor():
    # One lone move and otherwise flat: std over active steps is undefined, so the
    # floor must carry it — a finite score, not a divide-by-zero blow-up to 100.
    prices = [0.20] * 40 + [0.26] * 20          # exactly one non-zero step (+0.06)
    u = _unusualness(_prices_to_series(prices))
    assert u == round(100.0 * min(1.0, (0.06 / UNUSUALNESS_VOL_FLOOR) / 6.0), 1)


def test_same_move_scores_higher_in_a_quieter_market():
    # The bounded rarity credit the floor provides: the SAME 0.05 jump is more
    # unusual in a market whose other moves are tiny than in a churny one.
    quiet = _unusualness(_prices_to_series([0.30] * 59 + [0.35]))   # flat, then one +0.05
    noisy_steps = np.array([0.03, -0.03] * 29 + [0.05])            # active vol ~0.03, peak 0.05
    noisy_prices = 0.30 + np.concatenate([[0.0], np.cumsum(noisy_steps)])
    noisy = _unusualness(_prices_to_series(list(noisy_prices)))
    assert quiet > noisy


def test_time_to_close_bonus_applies_near_resolution():
    calm = list(0.15 + np.zeros(40))
    jumped = list(0.45 + np.zeros(10))
    near = score_market(_series(calm + jumped, close_in_hours=6), SETTINGS)
    far = score_market(_series(calm + jumped, close_in_hours=1000), SETTINGS)
    assert near["time_to_close_bonus"] > far["time_to_close_bonus"]
