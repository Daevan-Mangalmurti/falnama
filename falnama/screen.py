"""Stage 1.5 — an LLM relevance gate over the keyword-selected universe.

WHAT:     Asks a language model to judge each selected market on two independent
          axes — would its resolution move public financial markets, and could a
          small group plausibly know the outcome first — then drops the failures.
CONSUMES: outputs/relevant_markets/relevant_markets_latest.csv (Stage 1) +
          settings.screener + the LLM seam
PRODUCES: outputs/relevant_markets/screened_markets_latest.csv (the survivors)
          and screen_verdicts_<run_id>.csv (EVERY market, with its verdict and a
          one-line rationale — the calibration record)
REVIEWER: anyone asking "why did the pipeline spend money on THIS market?"
ROLE:     precision. Stage 1's keyword rules are cheap, deterministic, and
          offline — good properties for a first pass, but they read text without
          understanding it. They cannot tell that "the PRIMARY resolution source
          for this market" is boilerplate rather than an election, or that a 2028
          nomination race is decided by public opinion rather than private
          knowledge. This stage adds the judgment that keywords cannot express,
          and does it BEFORE the expensive stages (price history, scenario cards,
          and later news-lag) rather than after.

Why two axes instead of one relevance number:

    economic_salience     — would this outcome MOVE PUBLIC ASSETS?
    information_asymmetry — could a FEW PEOPLE know it before everyone else?

They fail differently and a single score hides that. A celebrity sentencing scores
low on the first; a presidential primary two years out scores respectably on the
first and near zero on the second (nobody knows yet — it is decided by millions of
voters). Falnama's thesis needs BOTH, so a market must clear both floors.

A note on why the first axis is not called "geopolitical relevance". Falnama's
founding documents scope the project to geopolitics, but that was always a PROXY:
geopolitical questions tend to combine market impact with privately-held knowledge,
so one label captured both properties at once. Now that the two are scored
separately, the proxy costs more than it earns — it would exclude an antitrust
ruling against a mega-cap or a surprise central-bank move, which can be far more
economically salient than a minor diplomatic gesture. Naming the real test lets
the screen widen without losing focus, because the asymmetry axis still holds the
line on its own.

The keyword score from Stage 1 is preserved next to the LLM's verdict rather than
overwritten. Seeing "keywords said 75, the model said 10" side by side is what
makes the screen tunable instead of mysterious.

FAIL-OPEN: if the model errors, times out, or returns a malformed batch, this
stage keeps every market and records the failure. A broken filter must never
silently empty the universe — a noisy run is recoverable, a blank one looks like
"no signal today" and quietly wastes a day of screening.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from . import io
from .config import Settings
from .io import RunContext

# How much of a market's description the model sees. Enough for context, short
# enough that a 25-market batch stays small. The tail of a Polymarket description
# is almost always resolution boilerplate, which is exactly what we want to skip.
_DESCRIPTION_CHARS = 260


@dataclass
class ScreenResult:
    """Outcome of Stage 1.5: the survivors, every verdict, and a small summary."""

    kept: pd.DataFrame
    verdicts: pd.DataFrame
    diagnostics: dict


# ---------------------------------------------------------------------------
# The structured shape the model fills
# ---------------------------------------------------------------------------
class MarketVerdict(BaseModel):
    """The model's judgment of one market.

    `index` is the market's position in the batch we sent. We map results back by
    position rather than by id or name, because a model re-typing a long id or
    question is a transcription error waiting to happen; an integer is not.
    """

    index: int = Field(description="The [n] index of the market being judged, exactly as given.")
    economic_salience: int = Field(description="0-100. Would this outcome move public financial markets?")
    information_asymmetry: int = Field(description="0-100. Could a small group know the outcome before the public?")
    corrected_topic: str = Field(description="The topic this market really belongs to, in snake_case.")
    rationale: str = Field(description="One short sentence justifying both scores.")


class ScreenBatch(BaseModel):
    """One verdict per market in the batch, in the order they were given."""

    verdicts: list[MarketVerdict]


# ---------------------------------------------------------------------------
# The prompt — the analytical heart of this stage. Edit here to retune judgment.
# ---------------------------------------------------------------------------
SCREEN_SYSTEM_PROMPT = """\
You are the relevance screener for Falnama, a paper-only research pipeline that \
studies whether geopolitical prediction markets move BEFORE the news is public. \
An earlier keyword pass has already produced a rough candidate list. Your job is \
to remove what does not belong, so the expensive downstream analysis is spent \
only on markets that could carry a real information signal.

Score every market on two INDEPENDENT axes, 0-100.

1. economic_salience — if this question resolved tomorrow, would it MOVE PUBLIC \
FINANCIAL MARKETS? Score HIGH when the outcome plausibly moves equities, indices, \
sector ETFs, commodities, currencies, or interest rates: armed conflict, \
sanctions, tariffs and trade policy, export controls, control of a major \
government or economy, central-bank and macro policy, energy supply, and large \
regulatory, antitrust, or legal decisions against major firms. Score LOW when the \
outcome is culturally interesting but financially inert: sport, entertainment, \
product releases, celebrity legal proceedings, and ordinary domestic crime, \
however famous the defendant.

Judge the ASSET IMPACT, not the topic label. Falnama's founding documents scope \
the project to "geopolitics", but that was a proxy — geopolitical questions \
usually combine market impact with private knowledge, so one word captured both. \
You are now scoring those two properties separately, so apply the real test \
directly: a major antitrust ruling or a surprise central-bank decision can be far \
more economically salient than a minor diplomatic gesture, and should score \
higher. Do not reject something merely because it is not statecraft.

2. information_asymmetry — could a SMALL NUMBER OF PEOPLE plausibly know this \
outcome before the general public? Score HIGH when the outcome is decided by a \
handful of actors who know their own intentions: a cabinet, a war room, a \
sanctions committee, a negotiating team, a prosecutor, a central-bank board, a \
corporate board, a regulator. Score LOW when the outcome is decided in public and \
in aggregate: elections and primaries decided by millions of voters, poll-driven \
questions, scheduled data releases, sporting contests. A far-off election is the \
clearest case of low asymmetry — no insider knows the answer either, because it \
does not exist yet.

The two axes are genuinely independent, and this is the point. A presidential \
primary two years out is economically meaningful (respectable) but knowable by \
nobody (low). A celebrity sentencing may have real insiders (moderate) but moves \
no asset (low). Both should be dropped, for different reasons — score each axis \
honestly on its own terms and let the pipeline's thresholds decide. Do not blend \
them into a single impression of how interesting the market is.

CRITICAL — ignore resolution boilerplate. Polymarket descriptions end with \
legalese about how the market settles ("the primary resolution source", \
"fulfilling the duties of the specified position", "resolves according to"). \
That text is administrative, not subject matter. Judge the QUESTION being asked, \
never the settlement mechanics. The keyword pass you are correcting was fooled \
by exactly this.

Also return corrected_topic: the topic the market truly belongs to, in \
snake_case (military_conflict, sanctions, trade_policy, technology_controls, \
cabinet_government, diplomacy_treaty, elections, central_bank_macro, energy, \
civil_unrest, regulatory_policy, legal_judicial, corporate_action, sport, \
entertainment, other). The keyword pass often gets this wrong; you are the \
correction.

Return exactly one verdict per market, echoing the given index. Be decisive but \
not harsh: when genuinely unsure, score near the middle rather than guessing at \
an extreme. One short sentence of rationale each.\
"""


def _build_batch_prompt(batch: list[dict]) -> str:
    """Render one batch of markets as an indexed list for the model to judge."""
    lines = ["Screen these prediction markets.", ""]
    for i, market in enumerate(batch):
        lines.append(f"[{i}] {market.get('market_name') or 'Unknown market'}")
        description = str(market.get("description") or "").strip().replace("\n", " ")
        if description:
            lines.append(f"    context: {description[:_DESCRIPTION_CHARS]}")
        topic = market.get("primary_topic")
        if topic:
            lines.append(f"    keyword pass guessed: {topic} (may be wrong — correct it)")
    lines += ["", f"Return exactly {len(batch)} verdicts, one per index above."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Screening (routes on mode; pure enough to test without the network)
# ---------------------------------------------------------------------------
def screen_markets(markets: pd.DataFrame, settings: Settings) -> ScreenResult:
    """Judge every market, then split into survivors and drops.

    Returns all three artifacts: the kept markets, the full verdict table (kept
    AND dropped, so exclusions stay auditable), and a summary.
    """
    cfg = settings.screener
    if markets.empty:
        return ScreenResult(markets, markets, {"screened": 0, "kept": 0, "dropped": 0})

    records = markets.to_dict(orient="records")
    verdicts, error = _judge_all(records, settings)

    min_salience = float(cfg.get("min_economic_salience", 60))
    min_asym = float(cfg.get("min_information_asymmetry", 45))

    rows = []
    for record, verdict in zip(records, verdicts):
        salience = float(verdict["economic_salience"])
        asym = float(verdict["information_asymmetry"])
        # A market must clear BOTH floors. Recording which one failed is what
        # makes the threshold tunable later — "dropped" alone teaches nothing.
        failed = []
        if salience < min_salience:
            failed.append(f"salience {salience:g} < {min_salience:g}")
        if asym < min_asym:
            failed.append(f"asymmetry {asym:g} < {min_asym:g}")
        rows.append({
            **record,
            "screen_economic_salience": salience,
            "screen_information_asymmetry": asym,
            "screen_topic": verdict["corrected_topic"],
            "screen_rationale": verdict["rationale"],
            "screen_verdict": "drop" if failed else "keep",
            "screen_drop_reason": "; ".join(failed),
        })

    all_verdicts = pd.DataFrame(rows)
    kept = all_verdicts[all_verdicts["screen_verdict"] == "keep"].reset_index(drop=True)
    dropped = all_verdicts[all_verdicts["screen_verdict"] == "drop"]

    diagnostics = {
        "mode": str(cfg.get("mode", "mock")),
        "screened": int(len(all_verdicts)),
        "kept": int(len(kept)),
        "dropped": int(len(dropped)),
        "min_economic_salience": min_salience,
        "min_information_asymmetry": min_asym,
        "dropped_by_topic": dropped["screen_topic"].astype(str).value_counts().to_dict() if not dropped.empty else {},
        "error": error,
    }
    return ScreenResult(kept=kept, verdicts=all_verdicts, diagnostics=diagnostics)


def _judge_all(records: list[dict], settings: Settings) -> tuple[list[dict], str]:
    """Return one verdict per record, plus an error string ('' when all is well).

    This is where FAIL-OPEN lives: any failure produces permissive verdicts that
    keep every market, and the reason is surfaced in the diagnostics and manifest
    rather than swallowed.
    """
    mode = str(settings.screener.get("mode", "mock"))
    if mode == "mock":
        return [_mock_verdict(r) for r in records], ""

    batch_size = max(1, int(settings.screener.get("batch_size", 25)))
    verdicts: list[dict] = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        try:
            verdicts.extend(_judge_batch(batch, settings))
        except Exception as exc:  # network, auth, malformed batch — all fail open
            # Keep this batch AND every remaining one, unjudged, then stop. We
            # abandon the rest deliberately: the likeliest causes (a missing key,
            # a bad model name) would fail identically on every later batch, and
            # burning twenty more calls to confirm that helps nobody.
            verdicts.extend(_permissive_verdict(r) for r in records[start:])
            return verdicts, f"{type(exc).__name__}: {exc}"
    return verdicts, ""


def _judge_batch(batch: list[dict], settings: Settings) -> list[dict]:
    """Screen one batch through the model and map the verdicts back by index."""
    parsed = _call_screen_llm(SCREEN_SYSTEM_PROMPT, _build_batch_prompt(batch), settings)

    # Map by the echoed index, then verify we got exactly one usable verdict per
    # market. A short, duplicated, or out-of-range batch is a model error, not
    # something to paper over — raising here triggers the fail-open path.
    by_index = {v.index: v for v in parsed.verdicts if 0 <= v.index < len(batch)}
    if len(by_index) != len(batch):
        raise ValueError(
            f"screener returned {len(by_index)} usable verdicts for a batch of {len(batch)}"
        )
    return [{
        "economic_salience": _clamp(by_index[i].economic_salience),
        "information_asymmetry": _clamp(by_index[i].information_asymmetry),
        "corrected_topic": str(by_index[i].corrected_topic or "other").strip().lower(),
        "rationale": str(by_index[i].rationale or "").strip(),
    } for i in range(len(batch))]


def _call_screen_llm(system: str, user: str, settings: Settings) -> ScreenBatch:
    """The single network boundary. Isolated so tests can mock it and keep CI
    offline — the same seam pattern as cards._call_scenario_llm.

    Note the model: screening is fast classification over many markets, not deep
    reasoning about one, so it runs on a small model by default (see
    llm.screener_model). Scenario cards keep the large model.
    """
    import anthropic  # imported lazily so the mock path needs no SDK / key

    client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY from the environment
    model = str(settings.llm.get("screener_model") or settings.llm.get("model"))
    timeout = int(settings.llm.get("timeout_seconds", 60))
    response = client.with_options(timeout=timeout).messages.parse(
        model=model,
        max_tokens=4000,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_format=ScreenBatch,
    )
    if response.parsed_output is None:
        raise ValueError("screener returned no parseable structured output")
    return response.parsed_output


def _mock_verdict(record: dict) -> dict:
    """Deterministic stand-in for the model, so the stage is testable and the
    pipeline still demonstrates real filtering with no key and no network.

    These few rules imitate the judgments the live screener makes on the cases we
    have actually seen go wrong — they are a fixture, not a serious classifier.
    """
    name = str(record.get("market_name") or "").lower()
    salience, asym = 80, 70
    topic = str(record.get("primary_topic") or "other")
    note = "mock: assumed in scope"

    if any(t in name for t in ("prison", "sentenced", "world cup", "gta", "oscar", "super bowl")):
        salience, topic, note = 10, "legal_judicial" if "prison" in name or "sentenced" in name else "sport", \
            "mock: moves no public asset"
    elif any(t in name for t in ("2028", "nomination", "primary")):
        asym, topic, note = 15, "elections", "mock: far-off race decided by public opinion"

    return {"economic_salience": salience, "information_asymmetry": asym,
            "corrected_topic": topic, "rationale": note}


def _permissive_verdict(record: dict) -> dict:
    """The fail-open verdict: keep the market, and say plainly why it was kept."""
    return {"economic_salience": 100, "information_asymmetry": 100,
            "corrected_topic": str(record.get("primary_topic") or "other"),
            "rationale": "screener unavailable — market kept without judgment"}


def _clamp(value) -> float:
    try:
        return float(max(0, min(100, float(value))))
    except (TypeError, ValueError):
        return 100.0  # unreadable score → permissive, consistent with fail-open


# ---------------------------------------------------------------------------
# Where downstream stages find the current universe
# ---------------------------------------------------------------------------
def universe_path(settings: Settings) -> Path:
    """The market list the expensive stages should read.

    Stages are decoupled through the filesystem, so this one function is how
    Stage 2 and Stage 3 learn whether an LLM screen ran. With the screener off,
    Stage 1's keyword-selected list is the universe; with it on, the screened
    subset is. Stage 1's own artifact is never modified either way.
    """
    directory = settings.output_dir("relevant_markets")
    if settings.screener.get("enabled", False):
        screened = directory / "screened_markets_latest.csv"
        if screened.exists():
            return screened
    return directory / "relevant_markets_latest.csv"


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run(ctx: RunContext, markets: pd.DataFrame | None = None) -> ScreenResult:
    """Screen the selected universe, write artifacts, and record what happened.

    When the screener is disabled this is a no-op pass-through, so the stage can
    sit permanently in the pipeline order (the same pattern as news-lag).
    """
    settings = ctx.settings
    out_dir = settings.output_dir("relevant_markets")

    if markets is None:
        markets = io.read_table(out_dir / "relevant_markets_latest.csv")

    if not settings.screener.get("enabled", False):
        io.update_manifest(ctx, "screen", {"enabled": False, "note": "LLM relevance screen disabled"})
        return ScreenResult(kept=markets, verdicts=pd.DataFrame(), diagnostics={"enabled": False})

    result = screen_markets(markets, settings)

    screened_path = io.write_table(result.kept, out_dir, "screened_markets", ctx.run_id)
    io.write_table(result.verdicts, out_dir, "screen_verdicts", ctx.run_id, also_latest=False)
    io.write_json(settings.output_dir("diagnostics") / f"screen_{ctx.run_id}.json", result.diagnostics)

    if result.diagnostics.get("error"):
        ctx.errors.append(f"screen: {result.diagnostics['error']}")

    io.update_manifest(ctx, "screen", {
        "enabled": True,
        **{k: result.diagnostics.get(k) for k in ("mode", "screened", "kept", "dropped", "error")},
        "screened_file": str(screened_path.relative_to(settings.project_root)),
    })
    return result
