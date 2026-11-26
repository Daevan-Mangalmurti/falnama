"""Stage 5: news-lag distinguishes moves explained by prior public news from
moves that prior news does NOT explain — enforcing the strictly-before-trigger
timing rule, and failing SAFE (unexplained, not dismissed) when the checker breaks."""

from __future__ import annotations

import pandas as pd
import pytest

from falnama import newslag
from falnama.config import load_config
from falnama.schema import validate

S = load_config()
S.raw["newslag"] = {
    "enabled": True, "mode": "mock", "lookback_hours": 48, "top_k_articles": 6,
    "embedding": {"provider": "lexical"},
}

# One strong anomaly: an implied-probability jump on an Iran-strike market at noon.
ANOMALY = {
    "market_id": "42", "market_name": "Will the US strike Iran before August?",
    "anomaly_trigger_time_utc": "2026-07-27T12:00:00Z", "anomaly_score": 90.0,
    "price_before": 0.20, "price_after": 0.55, "max_abs_move": 0.35,
}


def _art(title: str, hours_from_trigger: float, **extra) -> dict:
    t = pd.Timestamp("2026-07-27T12:00:00Z") + pd.Timedelta(hours=hours_from_trigger)
    return {"title": title, "snippet": title, "url": "http://x", "source": "wire",
            "published_time_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), **extra}


def test_confirmation_before_trigger_explains_the_move():
    # A confirming report published 3h BEFORE the trigger → high public info,
    # low residual: the market was digesting public news, not leaking.
    articles = [_art("US officially announces strike on Iran targets", -3)]
    a = newslag.assess(ANOMALY, articles, S)
    assert validate(a, "news_lag") == []
    assert a["information_state"] == "confirmation"
    assert a["public_information_score"] >= 70
    assert a["residual_anomaly_score"] <= 30            # 90 * (1 - high)


def test_no_prior_news_leaves_a_high_residual():
    # Nothing before the trigger → nothing explains the move → residual ~= the
    # anomaly's own strength. This is the interesting case Falnama hunts.
    a = newslag.assess(ANOMALY, [], S)
    assert a["public_information_score"] == 0
    assert a["residual_anomaly_score"] == ANOMALY["anomaly_score"]
    assert a["information_state"] == "none"
    assert "retrieval may be incomplete" in a["uncertainty_note"]  # honest about the limit


def test_post_trigger_news_cannot_explain_the_move():
    # The anti-ex-post rule: a confirming article published AFTER the trigger must
    # be ignored — you cannot explain a move with news that came later.
    articles = [_art("US officially announces strike on Iran targets", +2)]  # 2h AFTER
    a = newslag.assess(ANOMALY, articles, S)
    assert a["evidence"] == []                          # filtered out
    assert a["residual_anomaly_score"] == ANOMALY["anomaly_score"]


def test_articles_outside_the_lookback_window_are_dropped():
    # A confirmation from 10 days before the trigger is stale — outside the 48h
    # lookback, so it does not count as explaining THIS move.
    articles = [_art("US officially announces strike on Iran targets", -240)]  # 10 days
    a = newslag.assess(ANOMALY, articles, S)
    assert a["evidence"] == []
    assert a["residual_anomaly_score"] == ANOMALY["anomaly_score"]


def test_speculation_only_partially_explains():
    # Pre-trigger speculation that a strike MIGHT happen is not confirmation that it
    # did — a partial explanation, so the residual stays materially high.
    articles = [_art("US may weigh options as Iran tensions rise", -5)]
    a = newslag.assess(ANOMALY, articles, S)
    assert a["information_state"] == "speculation"
    assert 20 <= a["public_information_score"] <= 50
    assert a["residual_anomaly_score"] > 30


def test_evidence_only_carries_pre_trigger_articles():
    # A mix of before/after: only the before-articles survive into the evidence,
    # each strictly before the trigger (the schema's contract).
    articles = [_art("Iran tensions escalate, analysts say", -6),
                _art("Strike confirmed after the fact", +1)]
    a = newslag.assess(ANOMALY, articles, S)
    assert len(a["evidence"]) == 1
    trigger = pd.Timestamp(ANOMALY["anomaly_trigger_time_utc"])
    assert pd.Timestamp(a["evidence"][0]["published_time_utc"]) < trigger


def test_live_path_adjudicates_via_the_mocked_llm(monkeypatch):
    # Exercise the live adjudication WITHOUT the network by mocking the one LLM
    # boundary; ranking stays lexical (offline).
    live = load_config()
    live.raw["newslag"] = {**S.raw["newslag"], "mode": "live"}
    fake = newslag.NewsLagAnalysis(
        strongest_state="official_announcement", public_information_score=82.0,
        needs_deeper_review=False, uncertainty_note="official statement predates the move",
        articles=[newslag.ArticleJudgment(index=0, relevant=True,
                  information_state="official_announcement", explains_move=True, note="ok")])
    monkeypatch.setattr(newslag, "_call_newslag_llm", lambda *a, **k: fake)

    articles = [_art("US announces military action against Iran", -2)]
    a = newslag.assess(ANOMALY, articles, live)
    assert a["mode"] == "live"
    assert a["information_state"] == "official_announcement"
    assert a["public_information_score"] == 82.0
    assert a["evidence"][0]["information_state"] == "official_announcement"


def test_live_path_fails_safe_when_the_llm_errors(monkeypatch):
    # A broken checker must never claim a move was explained: residual stays at the
    # anomaly's strength, and the failure is recorded in the note.
    live = load_config()
    live.raw["newslag"] = {**S.raw["newslag"], "mode": "live"}

    def boom(*a, **k):
        raise RuntimeError("no API key")

    monkeypatch.setattr(newslag, "_call_newslag_llm", boom)
    a = newslag.assess(ANOMALY, [_art("US announces strike on Iran", -2)], live)
    assert a["public_information_score"] == 0
    assert a["residual_anomaly_score"] == ANOMALY["anomaly_score"]
    assert "unavailable" in a["uncertainty_note"] and "no API key" in a["uncertainty_note"]


def test_escalates_to_opus_only_when_flagged_ambiguous(monkeypatch):
    # Haiku triages; a needs_deeper_review flag routes the case to Opus. Verify the
    # escalation fires and the second (deep) verdict wins.
    live = load_config()
    live.raw["newslag"] = {**S.raw["newslag"], "mode": "live", "escalate_to_opus": True}
    live.raw["llm"] = {**live.raw["llm"], "screener_model": "haiku-x", "model": "opus-x"}
    calls = []

    def fake_llm(system, user, model, settings, *, use_thinking):
        calls.append(model)
        if model == "haiku-x":
            return newslag.NewsLagAnalysis(strongest_state="rumor", public_information_score=40.0,
                needs_deeper_review=True, uncertainty_note="thin snippets", articles=[])
        return newslag.NewsLagAnalysis(strongest_state="confirmation", public_information_score=88.0,
            needs_deeper_review=False, uncertainty_note="deep read", articles=[])

    monkeypatch.setattr(newslag, "_call_newslag_llm", fake_llm)
    a = newslag.assess(ANOMALY, [_art("Reports swirl of imminent Iran strike", -4)], live)
    assert calls == ["haiku-x", "opus-x"]              # triage then escalate
    assert a["information_state"] == "confirmation" and a["public_information_score"] == 88.0


def test_embedding_ranking_falls_back_to_lexical_on_error(monkeypatch):
    # If the embedding provider errors, ranking must degrade to lexical rather than
    # dropping the stage.
    cfg = {**S.raw["newslag"], "embedding": {"provider": "voyage"}}
    v = load_config()
    v.raw["newslag"] = cfg
    monkeypatch.setattr(newslag, "_embed", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no voyage key")))
    ranked = newslag._rank_candidates("US strike on Iran",
                                      [_art("Iran strike imminent", -1), _art("Unrelated sports news", -1)], v)
    assert ranked[0]["title"] == "Iran strike imminent"   # lexical still ranks it first


def test_disabled_newslag_is_a_no_op(tmp_path, monkeypatch):
    from falnama.io import RunContext

    off = load_config()
    off.raw["newslag"] = {**S.raw["newslag"], "enabled": False}
    monkeypatch.setattr(type(off), "output_dir", lambda self, key: tmp_path)
    ctx = RunContext.start(off)
    assert newslag.run(ctx, pd.DataFrame([ANOMALY])) == []


def test_run_writes_assessments_and_manifest(tmp_path, monkeypatch):
    from falnama.io import RunContext

    s = load_config()
    s.raw["newslag"] = {**S.raw["newslag"], "enabled": True, "mode": "mock"}
    monkeypatch.setattr(type(s), "output_dir", lambda self, key: tmp_path)
    ctx = RunContext.start(s)

    results = newslag.run(ctx, pd.DataFrame([ANOMALY]))
    assert len(results) == 1
    assert (tmp_path / "news_lag_latest.json").exists()
    for r in results:
        assert validate(r.assessment, "news_lag") == []


@pytest.mark.parametrize("state,expected_floor,expected_ceiling", [
    ("confirmation", 95, 100), ("speculation", 25, 35), ("none", 0, 0),
])
def test_state_maps_to_a_sensible_public_score(state, expected_floor, expected_ceiling):
    power = newslag._STATE_EXPLANATORY_POWER[state]
    assert expected_floor <= round(100 * power) <= expected_ceiling


def test_query_builder_keeps_anchors_and_drops_noise():
    # The query must keep topical anchors (actors/places/actions) and drop the
    # incidental words that crush GDELT recall — month names and generic verbs.
    q = newslag._news_query({"market_name": "Will the US strike Iran before August?"}).lower()
    assert "iran" in q and "strike" in q and "august" not in q
    q2 = newslag._news_query({"market_name": "Putin out as President of Russia by December 31, 2026?"}).lower()
    assert "putin" in q2 and "russia" in q2
    assert "out" not in q2.split() and "december" not in q2


class _FakeResp:
    def __init__(self, status: int, body: str):
        self.status_code, self.text = status, body

    def raise_for_status(self):
        if self.status_code >= 400 and self.status_code != 429:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        import json
        return json.loads(self.text)  # raises ValueError on a non-JSON body


def test_gdelt_retries_through_both_throttle_forms(monkeypatch):
    # GDELT throttles two ways: a hard 429, and a soft 200-with-plain-text-body.
    # _fetch_gdelt must retry BOTH and then parse the eventual JSON — real failure
    # modes seen against the live API.
    import sys
    import time as _time

    live = load_config()
    live.raw["newslag"] = {**S.raw["newslag"], "mode": "live", "gdelt_retry_backoff_seconds": 0}
    monkeypatch.setattr(_time, "sleep", lambda *a, **k: None)  # don't actually wait

    ok = '{"articles":[{"title":"US strike confirmed","url":"http://x","domain":"reuters","seendate":"20260728T120000Z"}]}'
    responses = iter([
        _FakeResp(429, ""),                                # hard throttle
        _FakeResp(200, "Your query rate is too high"),     # soft throttle (200 + non-JSON)
        _FakeResp(200, ok),                                # finally, real JSON
    ])
    monkeypatch.setitem(sys.modules, "requests", type("R", (), {"get": staticmethod(lambda *a, **k: next(responses))}))

    a = {"market_name": "US strike Iran", "anomaly_trigger_time_utc": "2026-07-29T00:00:00Z"}
    arts = newslag._fetch_gdelt(a, live)
    assert len(arts) == 1
    assert arts[0]["published_time_utc"] == "2026-07-28T12:00:00Z"   # seendate parsed
    assert arts[0]["source"] == "reuters"


def test_gdelt_gives_up_safely_after_exhausting_retries(monkeypatch):
    # Persistent throttle → the source is reported UNAVAILABLE (None, distinct from
    # [] = "answered, no coverage"), never a crash of the stage.
    import sys
    import time as _time

    live = load_config()
    live.raw["newslag"] = {**S.raw["newslag"], "mode": "live",
                           "gdelt_max_retries": 2, "gdelt_retry_backoff_seconds": 0}
    monkeypatch.setattr(_time, "sleep", lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "requests",
                        type("R", (), {"get": staticmethod(lambda *a, **k: _FakeResp(429, ""))}))

    a = {"market_name": "US strike Iran", "anomaly_trigger_time_utc": "2026-07-29T00:00:00Z"}
    with pytest.raises(newslag.NewsSourceUnavailable):
        newslag._fetch_gdelt(a, live)
    assert newslag._fetch_news(a, live) is None


# ---------------------------------------------------------------------------
# Reuse: each move is checked once, not once per run
# ---------------------------------------------------------------------------
def _ctx(tmp_path, monkeypatch, mode: str = "mock", run_id: str = "20260728T000000Z"):
    from falnama.io import RunContext

    s = load_config()
    s.raw["newslag"] = {**S.raw["newslag"], "enabled": True, "mode": mode}
    monkeypatch.setattr(type(s), "output_dir", lambda self, key: tmp_path)
    return RunContext(settings=s, run_id=run_id, run_time_utc="2026-07-28T00:00:00Z")


def _counting_fetch(monkeypatch, result=None):
    calls = []

    def fetch(anomaly, settings):
        calls.append(anomaly["market_id"])
        return newslag._mock_news(anomaly) if result == "mock" else result
    monkeypatch.setattr(newslag, "_fetch_news", fetch)
    return calls


def test_a_repeat_move_reuses_the_earlier_check(tmp_path, monkeypatch):
    calls = _counting_fetch(monkeypatch, "mock")
    first = newslag.run(_ctx(tmp_path, monkeypatch, run_id="20260728T000000Z"), pd.DataFrame([ANOMALY]))
    # Next run: same move, but the anomaly score drifted down.
    drifted = {**ANOMALY, "anomaly_score": 45.0}
    second = newslag.run(_ctx(tmp_path, monkeypatch, run_id="20260729T000000Z"), pd.DataFrame([drifted]))

    assert calls == ["42"]                                   # news fetched (and judged) once
    a1, a2 = first[0].assessment, second[0].assessment
    assert a2["reused_from_run_id"] == "20260728T000000Z"
    assert a2["public_information_score"] == a1["public_information_score"]
    assert a2["residual_anomaly_score"] == round(45.0 * (1 - a1["public_information_score"] / 100), 1)
    assert validate(a2, "news_lag") == []


def test_reuse_chains_point_at_the_original_check(tmp_path, monkeypatch):
    _counting_fetch(monkeypatch, "mock")
    for run_id in ("20260728T000000Z", "20260729T000000Z", "20260730T000000Z"):
        out = newslag.run(_ctx(tmp_path, monkeypatch, run_id=run_id), pd.DataFrame([ANOMALY]))
    assert out[0].assessment["reused_from_run_id"] == "20260728T000000Z"


def test_a_failed_news_fetch_is_checked_again_next_run(tmp_path, monkeypatch):
    calls = _counting_fetch(monkeypatch, None)              # source unavailable
    ctx = _ctx(tmp_path, monkeypatch, run_id="20260728T000000Z")
    first = newslag.run(ctx, pd.DataFrame([ANOMALY]))
    assert first[0].assessment["news_source_status"] == "unavailable"
    assert first[0].assessment["residual_anomaly_score"] == ANOMALY["anomaly_score"]   # fail-safe
    assert ctx.errors == [] and any("news source unavailable" in w for w in ctx.warnings)

    newslag.run(_ctx(tmp_path, monkeypatch, run_id="20260729T000000Z"), pd.DataFrame([ANOMALY]))
    assert calls == ["42", "42"]


def test_checks_from_before_the_status_field_are_not_reused(tmp_path, monkeypatch):
    import json
    # An old-format assessment cannot tell us whether its GDELT fetch worked.
    old = {**newslag.assess(ANOMALY, [], S)}
    old.pop("news_source_status")
    (tmp_path / "news_lag_20260720T000000Z.json").write_text(json.dumps([old]))
    calls = _counting_fetch(monkeypatch, "mock")
    newslag.run(_ctx(tmp_path, monkeypatch), pd.DataFrame([ANOMALY]))
    assert calls == ["42"]


def test_a_mock_check_is_not_reused_by_a_live_run(tmp_path, monkeypatch):
    _counting_fetch(monkeypatch, "mock")
    newslag.run(_ctx(tmp_path, monkeypatch, mode="mock", run_id="20260728T000000Z"), pd.DataFrame([ANOMALY]))
    assert newslag.prior_assessments(_ctx(tmp_path, monkeypatch, mode="live").settings) == {}
