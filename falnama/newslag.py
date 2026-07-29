"""Stage 5 — news-lag assessment (the epistemic counterweight to Stage 2).

WHAT:     For a strong anomaly, asks whether public news available STRICTLY BEFORE
          the trigger time already explained the move. A move that is both unusual
          (Stage 2) and unexplained by prior public news is the interesting case.
CONSUMES: strong anomalies (Stage 2) + candidate news articles + settings.newslag
PRODUCES: outputs/anomalies/ news-lag assessments (schema: news_lag), one per
          anomaly, plus a manifest record.
REVIEWER: a human gauging how surprising a signal really was — did the market lead
          the news, or just digest it?

The whole module turns on ONE comparison: article publication time vs. the anomaly
trigger time. Only news published BEFORE the trigger can *explain* a move (using
after-the-fact news to explain it is the ex-post rationalization the whole system
guards against). So the timing filter is the backbone, and it is deterministic and
free; the LLM only ever judges CONTENT, never timing.

Two layers, cheap-then-expensive (the same "cast wide, judge narrow" shape as the
rest of the pipeline):

  Layer 1 — retrieval (cheap, wide). Pull timestamped candidate articles from a
    news source (GDELT by default), then rank them by similarity to the market
    question. Ranking uses embeddings when configured (more discriminating than
    keyword overlap) and falls back to lexical overlap otherwise, so the module
    runs with no extra key.

  Layer 2 — adjudication (expensive, narrow). Only the top-ranked pre-trigger
    articles reach the LLM, which judges the strongest INFORMATION STATE present:
    speculation that X *might* happen is not confirmation that it *did*. A cheap
    model (Haiku) triages; genuinely ambiguous cases escalate to the expensive
    model (Opus).

Output per anomaly: a `public_information_score` (how well prior public news
explains the move) and a `residual_anomaly_score` (how much remains unexplained —
the metric that ranks "genuinely surprising"). Like cards, it has a deterministic
mock path and a live path, and it FAILS SAFE: if the news source or the LLM is
unavailable, it does NOT claim the move was explained — the residual stays high and
the failure is recorded, so a broken checker flags for review rather than dismissing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from . import io
from .config import Settings
from .io import RunContext
from .schema import validate_or_raise

# The information-state ladder, weakest → strongest. The whole point of Layer 2 is
# to place prior coverage on this ladder: only the strong end genuinely explains a
# move. Kept in code (it is the analytic taxonomy) and mirrored in the schema enum.
_STATES = Literal[
    "none", "market_commentary", "retrospective", "speculation",
    "rumor", "denial", "official_announcement", "confirmation",
]

# How much explanatory power each state carries when it PREDATES the trigger and
# points the same way the market moved. A confirmation fully explains; speculation
# barely does. These are the dial that turns a state into a public-information score.
_STATE_EXPLANATORY_POWER = {
    "none": 0.0, "market_commentary": 0.05, "retrospective": 0.05,
    "speculation": 0.30, "rumor": 0.45, "denial": 0.50,
    "official_announcement": 0.85, "confirmation": 1.0,
}

_STOPWORDS = {
    "will", "the", "a", "an", "be", "to", "of", "in", "on", "by", "before", "after",
    "is", "are", "was", "were", "and", "or", "for", "at", "with", "as", "that",
    "this", "it", "his", "her", "their", "than", "then", "any", "no", "not",
}


@dataclass
class NewsLagResult:
    """One anomaly's news-lag assessment (matches schemas/news_lag_schema.json)."""

    assessment: dict


# ---------------------------------------------------------------------------
# The structured shape the LLM fills (valid-by-construction)
# ---------------------------------------------------------------------------
class ArticleJudgment(BaseModel):
    """The model's read on one candidate article."""

    index: int = Field(description="The [n] index of the article being judged, as given.")
    relevant: bool = Field(description="Does this article actually bear on the market question?")
    information_state: _STATES = Field(description="The article's strongest information state.")
    explains_move: bool = Field(description="Could this article explain a move in the SAME direction the market moved?")
    note: str = Field(description="One short sentence of justification.")


class NewsLagAnalysis(BaseModel):
    """The model's full read on one anomaly against its pre-trigger articles."""

    strongest_state: _STATES = Field(description="The strongest information state across the relevant pre-trigger articles.")
    public_information_score: float = Field(description="0-100: how well pre-trigger public news explains the move.")
    needs_deeper_review: bool = Field(description="True if the snippets are ambiguous and full article text or a stronger model would help.")
    uncertainty_note: str = Field(description="Caveats — especially that absent coverage may be a retrieval gap, not a real absence.")
    articles: list[ArticleJudgment] = Field(description="One judgment per article given, in any order.")


# ---------------------------------------------------------------------------
# The prompt — the analytical heart of this stage. Edit here to retune judgment.
# ---------------------------------------------------------------------------
NEWSLAG_SYSTEM_PROMPT = """\
You are the news-lag analyst for Falnama, a paper-only research pipeline that \
studies whether prediction markets move BEFORE the news is public. You are given \
ONE market anomaly — a sharp price move at a known trigger time — and news \
articles published STRICTLY BEFORE that trigger. Decide whether prior public news \
already explains the move.

Judge the INFORMATION STATE, not mere topical overlap. An article speculating that \
something MIGHT happen is not a report that it DID. Rank coverage on this ladder, \
weakest to strongest: none < market_commentary < retrospective < speculation < \
rumor < denial < official_announcement < confirmation.

For each article decide: is it genuinely relevant to this market; what is its \
information state; and could it explain a move in the SAME direction the market \
moved (news that explains a move must match its sign — opposite-direction or \
unrelated coverage is coincidence, not explanation).

Then give public_information_score, 0-100: how well pre-trigger public news \
explains this specific move. Score HIGH (>70) only when a confirmation or official \
announcement, in the right direction, predates the trigger. Speculation or rumor \
alone is PARTIAL (20-50). Topical-but-non-moving coverage is LOW (<20). Genuinely \
no relevant prior news is 0.

Be skeptical and honest about your own limits. "No prior news found" is weak \
evidence of anything — it is easily just a retrieval gap (a paywalled or \
non-English outlet, or coverage that broke on social media rather than in the \
news). When the snippets are thin or ambiguous, set needs_deeper_review and say so \
in the uncertainty note rather than guessing. This is research, not investment advice.\
"""


def _build_user_prompt(anomaly: dict[str, Any], articles: list[dict[str, Any]]) -> str:
    """Assemble the per-anomaly user message: the move, then the indexed articles."""
    direction = "up" if _as_float(anomaly.get("max_abs_move")) and \
        _as_float(anomaly.get("price_after")) >= _as_float(anomaly.get("price_before")) else "down"
    lines = [
        "Assess whether prior public news explains this prediction-market move.",
        "",
        f"Market: {anomaly.get('market_name') or 'Unknown market'}",
        f"Trigger time (UTC): {anomaly.get('anomaly_trigger_time_utc')}",
        f"Implied-probability move: {_as_float(anomaly.get('price_before')):.3f} -> "
        f"{_as_float(anomaly.get('price_after')):.3f} (direction: {direction})",
        "",
        "Articles published STRICTLY BEFORE the trigger, most relevant first:",
    ]
    for i, art in enumerate(articles):
        lines.append(f"[{i}] ({art.get('published_time_utc')}) {art.get('title') or ''}".rstrip())
        snippet = str(art.get("snippet") or "").strip().replace("\n", " ")
        if snippet:
            lines.append(f"    {snippet[:300]}")
    lines += ["", f"Return exactly {len(articles)} article judgments, one per index above."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Assess one anomaly (the pure-ish core: timing filter -> rank -> judge -> score)
# ---------------------------------------------------------------------------
def assess(anomaly: dict[str, Any], articles: list[dict[str, Any]], settings: Settings) -> dict[str, Any]:
    """Assess one anomaly against candidate articles. Returns a schema-valid dict.

    Enforces the anti-ex-post timing rule (only pre-trigger articles can explain a
    move), ranks the survivors, routes mock/live, scores, and validates. The one
    network boundary in the live path is the LLM seam, mocked in tests.
    """
    cfg = settings.newslag
    mode = str(cfg.get("mode", "mock"))
    lookback_hours = float(cfg.get("lookback_hours", 48))
    trigger = pd.to_datetime(anomaly.get("anomaly_trigger_time_utc"), utc=True, errors="coerce")

    # --- Timing filter: only news STRICTLY BEFORE the trigger, within lookback ---
    window_start = trigger - pd.Timedelta(hours=lookback_hours) if pd.notna(trigger) else None
    pre_trigger = _strictly_before(articles, trigger, window_start)

    # --- Rank, keep the top-K for the LLM to actually read ---
    top_k = int(cfg.get("top_k_articles", 6))
    ranked = _rank_candidates(str(anomaly.get("market_name") or ""), pre_trigger, settings)[:top_k]

    # --- Adjudicate: none-case shortcut, else route mock/live ---
    if not ranked:
        analysis, error = _empty_analysis(), ""
    elif mode == "mock":
        analysis, error = _mock_analysis(ranked), ""
    else:
        analysis, error = _live_analysis(anomaly, ranked, settings)

    return _build_assessment(anomaly, ranked, analysis, settings, mode=mode, error=error)


def _strictly_before(articles: list[dict], trigger, window_start) -> list[dict]:
    """Keep only articles whose publish time is in [trigger - lookback, trigger).
    Articles with an unparseable time are dropped — we cannot vouch for their
    timing, and the anti-ex-post rule must not be satisfied on a guess."""
    if pd.isna(trigger):
        return []
    kept = []
    for art in articles:
        published = pd.to_datetime(art.get("published_time_utc"), utc=True, errors="coerce")
        if pd.notna(published) and window_start <= published < trigger:
            kept.append(art)
    return kept


def _build_assessment(anomaly: dict, ranked: list[dict], analysis: dict, settings: Settings,
                      *, mode: str, error: str) -> dict[str, Any]:
    """Turn an analysis into the schema-valid assessment: scores + evidence."""
    anomaly_score = _as_float(anomaly.get("anomaly_score"), 0.0)
    public = max(0.0, min(100.0, _as_float(analysis.get("public_information_score"), 0.0)))
    # Residual = the anomaly's own strength, discounted by how well prior news
    # explains it. A strong move nothing explains stays a strong residual; a
    # well-explained or weak move falls away. This is the "genuinely surprising" rank.
    residual = round(anomaly_score * (1.0 - public / 100.0), 1)

    # Per-article states the model returned, keyed by the index we sent.
    states = {j["index"]: j for j in analysis.get("articles", []) if isinstance(j, dict)}
    evidence = [{
        "snippet": str(art.get("snippet") or art.get("title") or "")[:600],
        "source": io.clean_id(art.get("source")),
        "url": io.clean_id(art.get("url")),
        "published_time_utc": str(art.get("published_time_utc")),
        "information_state": states.get(i, {}).get("information_state"),
    } for i, art in enumerate(ranked)]

    note = str(analysis.get("uncertainty_note") or _default_uncertainty(bool(ranked)))
    if error:
        note = f"News-lag checker unavailable ({error}); move left UNEXPLAINED pending review. {note}"

    assessment = {
        "market_name": str(anomaly.get("market_name") or "Unknown market"),
        "market_id": io.clean_id(anomaly.get("market_id")),
        "anomaly_trigger_time_utc": str(anomaly.get("anomaly_trigger_time_utc")),
        "lookback_hours": float(settings.newslag.get("lookback_hours", 48)),
        "public_information_score": round(public, 1),
        "residual_anomaly_score": residual,
        "information_state": str(analysis.get("strongest_state") or "none"),
        "evidence": evidence,
        "uncertainty_note": note,
        "mode": mode,
    }
    validate_or_raise(assessment, "news_lag")  # never emit an off-contract assessment
    return assessment


# ---------------------------------------------------------------------------
# Layer 1 — retrieval (the news source seam) + ranking
# ---------------------------------------------------------------------------
def _fetch_news(anomaly: dict[str, Any], settings: Settings) -> list[dict[str, Any]]:
    """Return candidate articles in the anomaly's lookback window. Routes mock/live.
    The single retrieval boundary — mock it (or run mock mode) to keep CI offline."""
    if str(settings.newslag.get("mode", "mock")) == "mock":
        return _mock_news(anomaly)
    try:
        return _fetch_gdelt(anomaly, settings)
    except Exception:  # network / parse / rate-limit — degrade to no candidates
        return []      # assess() then reports "no coverage found" (fail-safe: not "explained")


def _fetch_gdelt(anomaly: dict[str, Any], settings: Settings) -> list[dict[str, Any]]:
    """Live path — GDELT DOC 2.0 article search, time-boxed to the lookback window.

    GDELT is free, keyless, timestamped, and geopolitics-native — a good cheap-wide
    surveillance substrate. We ask only for article metadata (title, url, domain,
    seen-time); full text is fetched later only if a case reaches deep review.
    """
    import time

    import requests  # lazy import so the mock path needs no network stack

    cfg = settings.newslag
    trigger = pd.to_datetime(anomaly.get("anomaly_trigger_time_utc"), utc=True, errors="coerce")
    if pd.isna(trigger):
        return []
    start = trigger - pd.Timedelta(hours=float(cfg.get("lookback_hours", 48)))
    params = {
        "query": _news_query(anomaly),
        "mode": "ArtList", "format": "json",
        "maxrecords": int(cfg.get("max_candidates", 40)),
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": trigger.strftime("%Y%m%d%H%M%S"),
        "sort": "DateDesc",
    }
    base = str(cfg.get("gdelt_base_url", "https://api.gdeltproject.org/api/v2/doc/doc"))
    timeout = int(cfg.get("request_timeout_seconds", 30))
    # GDELT's free tier throttles hard (~1 request / 5s). It signals this TWO ways:
    # an HTTP 429, OR — its soft throttle — a 200 whose body is a plain-text warning
    # instead of JSON. Both must be retried; since we make one call per anomaly,
    # a polite backoff is worth it rather than fail-safing to "no coverage" on a
    # transient throttle. A real error (bad query -> 4xx) still raises and is caught
    # upstream; exhausting the retries returns [] (fail-safe: no coverage found).
    retries = int(cfg.get("gdelt_max_retries", 3))
    backoff = float(cfg.get("gdelt_retry_backoff_seconds", 5.0))
    headers = {"User-Agent": "falnama-research/0.1 (read-only)"}
    articles: list = []
    for attempt in range(retries + 1):
        resp = requests.get(base, params=params, timeout=timeout, headers=headers)
        if resp.status_code in (429, 500, 502, 503):
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
                continue
            return []  # exhausted -> fail-safe, don't crash the stage
        resp.raise_for_status()
        try:
            articles = resp.json().get("articles", []) if resp.text.strip() else []
            break
        except ValueError:  # 200 + non-JSON body = GDELT's soft throttle
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
                continue
            return []
    return [{
        "title": a.get("title"),
        "snippet": a.get("title"),           # DOC ArtList gives titles, not bodies
        "url": a.get("url"),
        "source": a.get("domain"),
        "published_time_utc": _parse_gdelt_time(a.get("seendate")),
    } for a in articles]


def _mock_news(anomaly: dict[str, Any]) -> list[dict[str, Any]]:
    """Deterministic canned articles for the mock path, timed around the trigger so
    the strictly-before filter is exercised. A fixture, not real retrieval — it lets
    the full pipeline run offline and demonstrate real timing/scoring behavior."""
    trigger = pd.to_datetime(anomaly.get("anomaly_trigger_time_utc"), utc=True, errors="coerce")
    if pd.isna(trigger):
        return []
    subject = _news_query(anomaly)[:60] or "the market subject"

    def art(title: str, hours: float) -> dict:
        return {"title": title, "snippet": title, "url": "https://example.test/mock",
                "source": "mock-wire",
                "published_time_utc": (trigger + pd.Timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")}

    return [
        art(f"Analysts weigh whether {subject} could unfold", -12),   # commentary/speculation, pre
        art(f"Officials reportedly consider action on {subject}", -5),  # speculation, pre
        art(f"Confirmed: {subject} announced", +1),                    # confirmation, POST (filtered)
    ]


def _news_query(anomaly: dict[str, Any]) -> str:
    """Build a GDELT keyword query from the market's salient terms."""
    name = str(anomaly.get("market_name") or "")
    terms = [t for t in re.findall(r"[A-Za-z][A-Za-z'-]+", name) if t.lower() not in _STOPWORDS]
    # Keep the distinctive terms; GDELT ANDs space-separated words.
    return " ".join(terms[:8]) or name


def _parse_gdelt_time(value) -> str | None:
    """GDELT seendate is 'YYYYMMDDTHHMMSSZ'. Normalize to ISO-8601, or None."""
    ts = pd.to_datetime(value, utc=True, errors="coerce", format="%Y%m%dT%H%M%SZ")
    if pd.isna(ts):
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(ts) else ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _rank_candidates(market_text: str, articles: list[dict], settings: Settings) -> list[dict]:
    """Rank articles by similarity to the market question, most relevant first.

    Uses embeddings when a provider is configured (more discriminating than keyword
    overlap, and far cheaper than an LLM call per article); falls back to lexical
    overlap otherwise, so ranking always works with no extra key. A failed embedding
    call also falls back rather than dropping the stage."""
    if len(articles) <= 1:
        return list(articles)
    provider = str(settings.newslag.get("embedding", {}).get("provider", "lexical"))
    scores = None
    if provider not in ("lexical", "none", ""):
        try:
            scores = _embedding_scores(market_text, articles, settings)
        except Exception:
            scores = None  # fall through to lexical — never fail the stage on ranking
    if scores is None:
        scores = [_lexical_overlap(market_text, f"{a.get('title') or ''} {a.get('snippet') or ''}")
                  for a in articles]
    return [a for _, a in sorted(zip(scores, articles), key=lambda p: p[0], reverse=True)]


def _lexical_overlap(a: str, b: str) -> float:
    """A cheap similarity: fraction of the market's content words the article shares."""
    wa = {w for w in re.findall(r"[a-z0-9]+", a.lower()) if w not in _STOPWORDS and len(w) > 2}
    wb = {w for w in re.findall(r"[a-z0-9]+", b.lower()) if w not in _STOPWORDS and len(w) > 2}
    return 0.0 if not wa else len(wa & wb) / len(wa)


def _embedding_scores(market_text: str, articles: list[dict], settings: Settings) -> list[float]:
    """Cosine similarity of the market question to each article, via the embedding
    provider (Voyage — Anthropic's recommended embeddings partner; NOT a first-party
    Anthropic endpoint, so it uses its own key). Isolated so tests can mock it."""
    emb_cfg = settings.newslag.get("embedding", {})
    texts = [market_text] + [f"{a.get('title') or ''} {a.get('snippet') or ''}" for a in articles]
    vectors = _embed(texts, emb_cfg)
    q, docs = vectors[0], vectors[1:]
    return [_cosine(q, d) for d in docs]


def _embed(texts: list[str], emb_cfg: dict) -> list[list[float]]:
    """The embedding network boundary. Voyage via its SDK; mock this in tests."""
    import voyageai  # lazy import; only needed when embedding.provider: voyage

    client = voyageai.Client()  # resolves VOYAGE_API_KEY from the environment
    model = str(emb_cfg.get("model", "voyage-3.5"))
    return client.embed(texts, model=model, input_type="document").embeddings


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


# ---------------------------------------------------------------------------
# Layer 2 — adjudication (mock + live LLM, with Haiku->Opus escalation)
# ---------------------------------------------------------------------------
def _live_analysis(anomaly: dict, ranked: list[dict], settings: Settings) -> tuple[dict, str]:
    """Judge the articles with the cheap model, escalating ambiguous cases to the
    expensive one. Returns (analysis dict, error string — '' when all is well)."""
    system, user = NEWSLAG_SYSTEM_PROMPT, _build_user_prompt(anomaly, ranked)
    triage_model = str(settings.llm.get("screener_model") or settings.llm.get("model"))
    try:
        analysis = _call_newslag_llm(system, user, triage_model, settings, use_thinking=False)
    except Exception as exc:  # auth / rate limit / network / malformed — FAIL SAFE
        return _empty_analysis(), f"{type(exc).__name__}: {exc}"

    # Escalate only the genuinely-ambiguous cases to the expensive model.
    if analysis.needs_deeper_review and settings.newslag.get("escalate_to_opus", True):
        deep_model = str(settings.llm.get("model"))
        if deep_model and deep_model != triage_model:
            try:
                analysis = _call_newslag_llm(system, user, deep_model, settings, use_thinking=True)
            except Exception:
                pass  # keep the triage verdict; it is already usable
    return _analysis_to_dict(analysis), ""


def _call_newslag_llm(system: str, user: str, model: str, settings: Settings,
                      *, use_thinking: bool) -> NewsLagAnalysis:
    """The LLM boundary. Structured output via messages.parse (same proven pattern
    as cards/screen), so the response is a validated pydantic object."""
    import anthropic  # lazy import so the mock path needs no SDK / key

    client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY from the environment
    timeout = int(settings.llm.get("timeout_seconds", 60))
    kwargs: dict[str, Any] = dict(model=model, max_tokens=4000, system=system,
                                  messages=[{"role": "user", "content": user}],
                                  output_format=NewsLagAnalysis)
    if use_thinking:
        kwargs["thinking"] = {"type": "adaptive"}  # let the deep model reason before committing
    response = client.with_options(timeout=timeout).messages.parse(**kwargs)
    if response.parsed_output is None:
        raise ValueError("news-lag LLM returned no parseable structured output")
    return response.parsed_output


def _analysis_to_dict(analysis: NewsLagAnalysis) -> dict[str, Any]:
    return {
        "strongest_state": analysis.strongest_state,
        "public_information_score": analysis.public_information_score,
        "needs_deeper_review": analysis.needs_deeper_review,
        "uncertainty_note": analysis.uncertainty_note,
        "articles": [j.model_dump() for j in analysis.articles],
    }


def _mock_analysis(ranked: list[dict]) -> dict[str, Any]:
    """Deterministic stand-in for the LLM: reads the strongest information state off
    simple cue words in the ranked titles, and scores it via the same ladder the
    real scorer uses. A fixture, not a classifier — enough to exercise the plumbing
    and demonstrate real behavior offline."""
    cues = [
        ("confirmation", ("confirms", "confirmed", "announces", "announced", "official", "signs", "declares")),
        ("speculation", ("may", "could", "weighs", "considers", "mulls", "reportedly", "expected to")),
    ]
    judgments, strongest = [], "none"
    for i, art in enumerate(ranked):
        text = f"{art.get('title') or ''} {art.get('snippet') or ''}".lower()
        state = "market_commentary"
        for candidate, words in cues:
            if any(w in text for w in words):
                state = candidate
                break
        if _STATE_EXPLANATORY_POWER[state] > _STATE_EXPLANATORY_POWER[strongest]:
            strongest = state
        judgments.append({"index": i, "relevant": True, "information_state": state,
                          "explains_move": state == "confirmation", "note": "mock keyword read"})
    return {
        "strongest_state": strongest,
        "public_information_score": round(100.0 * _STATE_EXPLANATORY_POWER[strongest], 1),
        "needs_deeper_review": False,
        "uncertainty_note": "MOCK news-lag assessment — deterministic keyword read, not a real judgment.",
        "articles": judgments,
    }


def _empty_analysis() -> dict[str, Any]:
    """The no-evidence / fail-safe analysis: nothing explains the move."""
    return {"strongest_state": "none", "public_information_score": 0.0,
            "needs_deeper_review": False,
            "uncertainty_note": "No relevant pre-trigger coverage found — this is weak evidence "
                                "(retrieval may be incomplete), not proof the move was non-public.",
            "articles": []}


def _default_uncertainty(had_articles: bool) -> str:
    return ("LLM read of article snippets; confirm against full text before acting."
            if had_articles else _empty_analysis()["uncertainty_note"])


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run(ctx: RunContext, anomalies: pd.DataFrame | None = None) -> list[NewsLagResult]:
    """Assess each strong anomaly against prior public news, write the assessments,
    and record the batch. A clean no-op while disabled (the default), so the
    pipeline runs end-to-end without this module."""
    settings = ctx.settings
    if not settings.newslag.get("enabled", False):
        io.update_manifest(ctx, "newslag", {"enabled": False, "note": "news-lag disabled"})
        return []

    if anomalies is None:
        anomalies = io.read_table(settings.output_dir("anomalies") / "strong_anomalies_latest.csv")

    results, errors = [], 0
    for record in anomalies.to_dict(orient="records"):
        articles = _fetch_news(record, settings)
        assessment = assess(record, articles, settings)
        if "unavailable" in assessment["uncertainty_note"]:
            errors += 1
        results.append(NewsLagResult(assessment))

    out_dir = settings.output_dir("anomalies")
    io.write_json(out_dir / f"news_lag_{ctx.run_id}.json", [r.assessment for r in results])
    if results:
        io.write_json(out_dir / "news_lag_latest.json", [r.assessment for r in results])

    if errors:
        ctx.errors.append(f"newslag: {errors} assessment(s) left unexplained (checker unavailable)")
    io.update_manifest(ctx, "newslag", {
        "enabled": True, "mode": str(settings.newslag.get("mode", "mock")),
        "assessed": len(results),
        "unexplained_residual": int(sum(r.assessment["residual_anomaly_score"] >= 50 for r in results)),
        "checker_errors": errors,
    })
    return results


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default
