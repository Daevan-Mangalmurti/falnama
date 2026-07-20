"""The orchestrator — run the stages in order and write the audit trail.

WHAT:     Runs the signal chain end to end: select → detect → cards → recommend →
          (execution) → (news-lag), writing the run manifest and health report.
CONSUMES: a `Settings` object (loaded from config if not supplied)
PRODUCES: every stage's artifacts, tied together by one run_id, plus the
          manifest and health files under outputs/run_logs/
REVIEWER: anyone running Falnama; this is the top-level entry point
ROLE:     glue, and nothing more. The orchestrator decides ORDER and records what
          happened; all judgment lives in the stage modules. Keeping it thin is
          the point — the pipeline should read like the diagram in the README.

Stages are decoupled through the filesystem: each stage writes a
'<name>_latest' artifact and the next stage reads it. The one in-memory hand-off
is recommend → execution (the recommendation list), because the paper-sim seam
acts on the recommendations directly.
"""

from __future__ import annotations

from . import io
from .config import Settings, load_config
from .io import RunContext

# The signal chain, in order. `screen` is the optional LLM relevance gate; it sits
# right after keyword selection so everything downstream works on a smaller, better
# universe (and is a no-op pass-through when disabled).
STAGE_ORDER = ["select", "screen", "anomaly", "cards", "recommend", "execution", "newslag"]


def run(settings: Settings | None = None, stages: list[str] | None = None) -> RunContext:
    """Execute the pipeline and return the finished RunContext.

    `stages` restricts which stages run (default: all, in order). On any stage
    failure the error is recorded, the health report is still written, and the
    exception propagates so callers/CI see a non-zero exit.
    """
    from . import anomaly, cards, execution, newslag, recommend, screen, select

    settings = settings or load_config()
    ctx = RunContext.start(settings)
    order = [s for s in STAGE_ORDER if s in set(stages or STAGE_ORDER)]
    status = {stage: False for stage in order}
    recommendations: list[dict] = []

    try:
        for stage in order:
            if stage == "select":
                select.run(ctx)
            elif stage == "screen":
                screen.run(ctx)
            elif stage == "anomaly":
                anomaly.run(ctx)
            elif stage == "cards":
                cards.run(ctx)
            elif stage == "recommend":
                recommendations = recommend.run(ctx).recommended
            elif stage == "execution":
                execution.run(ctx, recommendations)
            elif stage == "newslag":
                newslag.run(ctx)
            status[stage] = True
    except Exception as exc:
        ctx.errors.append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        io.write_health(ctx, status)

    return ctx
