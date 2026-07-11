"""io foundation: id coercion and empty-tolerant table reads.

These guard the two things that broke the first live run: numeric Polymarket ids
surviving a CSV round-trip, and a run that scores no anomalies writing an empty
CSV that the next stage then reads."""

from __future__ import annotations

from falnama import io


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
