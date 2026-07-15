# Falnama

**Detect anomalous geopolitical prediction-market activity that precedes public
news, preserve the evidence, and turn it into auditable research artifacts.**

Falnama treats prediction markets as *noisy information sensors*. A market price
is not "truth" — it is a signal produced by traders with different information,
incentives, and biases. The narrow question Falnama asks is:

> Did this market move *too abruptly, too concentrated, or too close to
> resolution* to be explained by the public news available at the time?

Falnama is a **research and intelligence-analysis tool, not a trading bot.** It
produces paper recommendations and rejected-signal logs for a human to review.
It does not place real orders, and there is no broker code in this repository.

> Guiding principle: **software is plumbing; information flow is the mission.**
> Every file exists to clarify, preserve, or test the signal chain below.

## The signal chain

```
Polymarket activity
   → 1. select       build a small, high-quality geopolitical market universe
   → 2. detect       score each market for anomalous behavior (interpretable, not a black box)
   → 3. interpret    write an immutable "scenario index card" mapping the event to assets (LLM)
   → 4. recommend    turn eligible anomalies + cards into PAPER recommendations
   → 5. news-lag     check whether public news already explained the move (research module)
        + audit       every step writes a traceable artifact
```

Each stage is one module in `falnama/`, and every run leaves an evidence trail
in `outputs/`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run the whole pipeline in mock mode (no network, no API key needed):
python scripts/run.py

# Run the test suite:
pytest
```

By default Falnama runs against committed sample data in `data/fixtures/` and
generates **mock** index cards, so a collaborator can clone, run, and inspect
every artifact with no credentials. To pull live Polymarket data or generate
real LLM index cards, see the config notes below.

## How to read the code

The engine lives in `falnama/`. Open any module and the header docstring tells
you, in plain language, **what it does, what it reads, what it produces, and who
reviews the output** — so you can understand its role in the signal chain in
about a minute, even with only a moderate coding background.

| File | Stage | Responsibility |
|------|-------|----------------|
| `config.py`     | —      | Load and validate `config/falnama.yaml` into one typed settings object. |
| `io.py`         | —      | Paths, run IDs, timestamped writes, run manifest + health report. |
| `schema.py`     | —      | Validate artifacts against the JSON schemas in `schemas/`. |
| `polymarket.py` | data   | Read-only client for the Polymarket APIs, plus snapshot-to-disk and a fixtures fallback. |
| `select.py`     | 1      | Build the geopolitical market universe; log every accept/reject. |
| `anomaly.py`    | 2      | Interpretable deterministic anomaly scores (no opaque model). |
| `cards.py`      | 3      | Generate immutable, hashed scenario index cards via the LLM seam (mock by default). |
| `recommend.py`  | 4      | Paper recommendation engine: eligibility, anti-ex-post timing, sizing, no-trade bias. |
| `execution.py`  | 4      | Optional **paper-simulation** seam (config-gated). No real broker. |
| `newslag.py`    | 5      | News-lag assessment: was the move already public? (research module) |
| `pipeline.py`   | —      | Orchestrator: run the stages in order, write the run manifest and health. |
| `review.py`     | —      | Load a run's artifacts + the cross-run history for the analyst notebook. |

## Configuration

Everything tunable lives in [`config/falnama.yaml`](config/falnama.yaml), which
is heavily commented. The settings you are most likely to change:

- `card_mode: mock` → `live` — generate real LLM index cards (needs `ANTHROPIC_API_KEY`).
- `data.source: fixtures` → `live` — pull live data from Polymarket.
- `recommend.max_position_usd` — the paper position-size cap.

## Reviewing a run

Rather than reading raw files in an editor, open the analyst notebook:

```bash
pip install -e ".[viz]"
jupyter notebook notebooks/analyst_review.ipynb
```

It loads the latest run (or any past `run_id`) and shows a compiled view: the run
summary, the selected universe by topic/region, the ranked anomalies and
concentration flags, recommendations and rejections, and cross-run trends. All
the parsing is in [`falnama/review.py`](falnama/review.py), so you can also use it
from a plain script: `from falnama import review; review.run_history()`.

## What an "index card" is (and the next milestone)

A **scenario index card** is the heart of Falnama's defense against *ex-post
rationalization*. Before a trade is ever recommended, the system writes an
immutable, timestamped, content-hashed JSON card that maps a prediction-market
question to its plausible downstream asset implications (asset, direction,
magnitude, confidence, time plan, evidence, failure modes). Because the card is
created *before* the anomaly is acted on and can never be edited in place, a
recommendation can always be traced back to a prior, frozen hypothesis.

This scaffold ships the card **interface, schema, and a deterministic mock**.
The immediate next milestone is to realize the live LLM card generator in
`cards.py` (Claude via the `anthropic` SDK, with the card schema enforced so
output is valid by construction).

## Outputs (the evidence trail)

A successful run does **not** have to produce a trade — producing only rejected
signals is a valid, useful outcome (rejections are data, not waste). Artifacts
land in `outputs/`:

```
outputs/relevant_markets/     selected market universe + selection diagnostics
outputs/anomalies/            ranked + strong anomalies, concentration diagnostics
outputs/index_cards/          immutable, hashed scenario cards
outputs/recommended_trades/   paper recommendations (.xlsx workbook + JSON)
outputs/rejected_signals/     why candidates were rejected
outputs/run_logs/             per-run manifest + health report
outputs/diagnostics/          card↔anomaly matching and trade diagnostics
```
