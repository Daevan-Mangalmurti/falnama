"""io foundation: id coercion and empty-tolerant table reads.

These guard the two things that broke the first live run: numeric Polymarket ids
surviving a CSV round-trip, and a run that scores no anomalies writing an empty
CSV that the next stage then reads."""

from __future__ import annotations

from falnama import io
from falnama.config import load_config


def test_clean_id_coerces_numeric_to_string():
    assert io.clean_id(540843) == "540843"        # int, as a live market id arrives
    assert io.clean_id(540843.0) == "540843"      # float, from a CSV column with NaNs
    assert io.clean_id("mil-iran-strike") == "mil-iran-strike"


def test_clean_id_treats_missing_as_none():
    assert io.clean_id(None) is None
    assert io.clean_id(float("nan")) is None
    assert io.clean_id("") is None
    assert io.clean_id("nan") is None


def test_read_table_tolerates_missing_and_empty(tmp_path):
    assert io.read_table(tmp_path / "does_not_exist.csv").empty
    empty = tmp_path / "empty.csv"
    empty.write_text("")  # a 0-anomaly run writes a header-less, empty file
    assert io.read_table(empty).empty


def test_sensor_warning_fires_only_below_the_availability_floor():
    ctx = io.RunContext.start(load_config())
    ctx.settings.raw["data"]["min_sensor_availability"] = 0.5
    io.warn_if_sensor_dark(ctx, "s", 0, 0, [])                 # asked about nothing: no verdict
    io.warn_if_sensor_dark(ctx, "s", 5, 10, ["x"] * 5)          # exactly half: fine
    assert ctx.warnings == []
    io.warn_if_sensor_dark(ctx, "s", 1, 3, ["timeout", "410 Gone", "410 Gone"])
    assert ctx.warnings == ["WARNING: s available for only 1 of 3 markets (below 50%); "
                            "most common reason: 410 Gone"]
