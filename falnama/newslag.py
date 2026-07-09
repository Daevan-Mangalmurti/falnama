"""Stage 5 — news-lag assessment (the research module, still maturing).

WHAT:     For an anomaly, asks whether public news available STRICTLY BEFORE the
          trigger time already explained the move.
CONSUMES: strong anomalies (Stage 2) + candidate news articles + settings.newslag
PRODUCES: outputs/anomalies/ news-lag assessments (schema: news_lag)
REVIEWER: a human gauging how "surprising" a signal really was
ROLE:     the epistemic counterweight to Stage 2. Stage 2 asks "did the market
          move unusually?"; Stage 5 asks "was it already public?". A move that is
          both unusual AND unexplained by prior public news is the interesting case.

The hard part is information STATE, not keyword matching. An article saying a
strike MIGHT happen (speculation) is not a report that it DID (confirmation).
The module therefore classifies the strongest pre-trigger information state and
reports a public-information score, a residual-anomaly score, and the evidence —
each article strictly timestamped before the trigger. Like cards, it has a mock
path (deterministic) and a live path (the LLM seam).

NOTE: off by default (newslag.enabled). Implemented in Phase 5. This stub
documents the intended interface.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .io import RunContext

_NOT_YET = "Implemented in Phase 5 (news-lag)."


@dataclass
class NewsLagResult:
    """One anomaly's news-lag assessment (matches schemas/news_lag_schema.json)."""

    assessment: dict


def assess(anomaly: dict, articles: list[dict], settings: Settings) -> dict:
    """Assess one anomaly against pre-trigger articles. Returns a news-lag dict.
    Routes mock/live on settings.newslag['mode']. Pure and unit-testable."""
    raise NotImplementedError(_NOT_YET)


def run(ctx: RunContext, anomalies=None) -> list[NewsLagResult]:
    """Stage entry point. A clean no-op while disabled (the default), so the
    pipeline runs end-to-end without this deferred module. Enabling it before it
    is implemented fails loudly rather than silently skipping."""
    if not ctx.settings.newslag.get("enabled", False):
        return []
    raise NotImplementedError(
        "News-lag is enabled in config but not implemented yet (deferred module). "
        "Set newslag.enabled: false to run the rest of the pipeline."
    )
