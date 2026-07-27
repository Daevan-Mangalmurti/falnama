"""review.py compiles outputs/ into tables for the analyst notebook."""

from __future__ import annotations

import json

import pytest

from falnama import review
from falnama.config import Settings, load_config


def _settings(tmp_path):
    # A Settings rooted at tmp_path so review reads only from there.
    return Settings(raw=load_config().raw, project_root=tmp_path.resolve())


def _write_run(settings, run_id, selected, strong):
    logs = settings.output_dir("run_logs")
    (logs / f"run_manifest_{run_id}.json").write_text(json.dumps({
        "run_id": run_id, "run_time_utc": f"{run_id[:4]}-07-15T09:00:00Z",
        "data_source": "live", "card_mode": "mock",
        "market_selector": {"selected_count": selected, "rejected_count": 3},
        "anomaly_detector": {"markets_scored": selected, "strong_count": strong, "concentration_red_flags": 1},
        "recommender": {"recommended_count": 0, "rejected_count": 1},
    }))
    (logs / f"pipeline_health_{run_id}.json").write_text(json.dumps({"run_id": run_id, "success": True}))
    (settings.output_dir("relevant_markets") / f"relevant_markets_{run_id}.csv").write_text("market_name,primary_topic\nX,elections\n")
    (settings.output_dir("anomalies") / f"ranked_anomalies_{run_id}.csv").write_text("market_name,anomaly_score\nX,72\n")


def test_run_history_is_oldest_first(tmp_path):
    s = _settings(tmp_path)
    _write_run(s, "20260714T090000Z", 30, 0)
    _write_run(s, "20260715T090000Z", 31, 1)
    hist = review.run_history(s)
    assert len(hist) == 2
    assert list(hist["selected"]) == [30, 31]  # sorted oldest -> newest


def test_load_run_defaults_to_latest(tmp_path):
    s = _settings(tmp_path)
    _write_run(s, "20260714T090000Z", 30, 0)
    _write_run(s, "20260715T090000Z", 31, 1)
    assert review.latest_run_id(s) == "20260715T090000Z"
    run = review.load_run(s)
    assert run.run_id == "20260715T090000Z"
    assert len(run.relevant_markets) == 1 and len(run.anomalies) == 1
    assert run.recommendations.empty  # none written -> empty frame, not a crash


def test_unknown_run_id_fails_loudly(tmp_path):
    # The bug this guards: a run_id that matches no manifest used to load an EMPTY
    # manifest, so the notebook's summary cell blew up later with KeyError. It must
    # instead raise immediately, and name the valid ids.
    s = _settings(tmp_path)
    _write_run(s, "20260715T090000Z", 31, 1)
    with pytest.raises(FileNotFoundError, match="20260715T090000Z"):
        review.load_run(s, run_id="20260715T999999Z")


def test_load_run_tolerates_pasted_decorations(tmp_path):
    # Copy-paste slips that should still resolve to the right run rather than error.
    s = _settings(tmp_path)
    _write_run(s, "20260715T090000Z", 31, 1)
    for messy in ["run_manifest_20260715T090000Z", "20260715T090000Z.json",
                  "  20260715T090000Z  ", "outputs/run_logs/run_manifest_20260715T090000Z.json"]:
        run = review.load_run(s, run_id=messy)
        assert run.run_id == "20260715T090000Z"
        assert run.manifest["market_selector"]["selected_count"] == 31
