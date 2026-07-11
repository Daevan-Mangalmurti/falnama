"""Shared input/output: run identity, timestamped writes, and the audit trail.

WHAT:     One place for the small, repetitive jobs every stage needs — stamping
          a run with an ID and time, writing timestamped CSV/JSON without
          clobbering older files, and recording the run manifest + health report.
CONSUMES: a `Settings` object (for output paths)
PRODUCES: files under outputs/run_logs/ (manifest + health), plus helpers used
          by other stages to write their own artifacts
REVIEWER: anyone auditing "what happened in run X" — start at the manifest
ROLE:     foundation. Keeping all of this here means the stages stay focused on
          analysis, and the evidence trail has a single, consistent format.

The audit trail is two files per run:
  * run_manifest_<run_id>.json — what each stage read and produced (the "what")
  * pipeline_health_<run_id>.json — which stages succeeded, warnings, errors,
    and timing (the "did it work")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Settings


# ---------------------------------------------------------------------------
# Time and identity
# ---------------------------------------------------------------------------
def utc_now() -> str:
    """Current UTC time as an ISO-8601 string, e.g. '2026-06-27T08:30:00Z'.

    A single canonical timestamp format is used everywhere so artifacts sort and
    compare cleanly.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def compact_stamp(iso_time: str) -> str:
    """Turn '2026-06-27T08:30:00Z' into '20260627T083000Z' for use in filenames."""
    return iso_time.replace(":", "").replace("-", "")


def new_run_id(iso_time: str | None = None) -> str:
    """A run ID is just the compact timestamp of when the run started."""
    return compact_stamp(iso_time or utc_now())


def clean_id(value) -> str | None:
    """Coerce an id / slug / url field to a clean string, or None.

    Guards the schema boundary: our JSON schemas require these fields to be
    string-or-null, but real Polymarket market ids are NUMERIC, and a CSV
    round-trip through pandas turns "540843" into an int (or a float, or NaN when
    the cell is empty). This normalizes all of those back to a plain string/None
    so artifacts stay schema-valid whatever the data source looks like.
    """
    if value is None:
        return None
    if isinstance(value, float):
        if value != value:  # NaN
            return None
        if value.is_integer():
            value = int(value)  # 540843.0 -> 540843, not "540843.0"
    text = str(value).strip()
    return None if text.lower() in ("", "nan", "none") else text


# ---------------------------------------------------------------------------
# RunContext — the small bundle of identity that flows through the pipeline
# ---------------------------------------------------------------------------
@dataclass
class RunContext:
    """Carries the identity of one pipeline run so every stage tags its outputs
    consistently and appends to the same manifest/health files."""

    settings: Settings
    run_id: str
    run_time_utc: str
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @classmethod
    def start(cls, settings: Settings) -> RunContext:
        now = utc_now()
        return cls(settings=settings, run_id=new_run_id(now), run_time_utc=now)


# ---------------------------------------------------------------------------
# Writing artifacts without overwriting history
# ---------------------------------------------------------------------------
def unique_path(path: Path) -> Path:
    """Return `path`, or `path` with a numeric suffix if it already exists.

    Falnama never silently overwrites an artifact — old evidence is preserved.
    """
    if not path.exists():
        return path
    for i in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{i:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not allocate a unique path near {path}")


def write_json(path: Path, data: Any) -> Path:
    """Write `data` as pretty JSON, creating parent folders as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def read_json(path: Path, default: Any = None) -> Any:
    """Read JSON if the file exists, else return `default`."""
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV artifact, tolerating a missing or empty file (returns an empty
    DataFrame). Intermediate artifacts can legitimately be empty — e.g. a run
    that scores no anomalies — and that must never crash the next stage."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_table(df: pd.DataFrame, directory: Path, basename: str, run_id: str,
                also_latest: bool = True) -> Path:
    """Write a DataFrame as a timestamped CSV, optionally refreshing a stable
    '<basename>_latest.csv' pointer that the next stage can find easily.

    Returns the path of the timestamped file (the durable record).
    """
    directory.mkdir(parents=True, exist_ok=True)
    stamped = unique_path(directory / f"{basename}_{run_id}.csv")
    df.to_csv(stamped, index=False)
    if also_latest:
        df.to_csv(directory / f"{basename}_latest.csv", index=False)
    return stamped


# ---------------------------------------------------------------------------
# The audit trail: run manifest + pipeline health
# ---------------------------------------------------------------------------
def _run_logs_dir(settings: Settings) -> Path:
    return settings.output_dir("run_logs")


def update_manifest(ctx: RunContext, section: str, updates: dict[str, Any]) -> Path:
    """Merge `updates` into a named section of this run's manifest.

    The manifest answers "what did each stage read and write?" — the factual
    record of the run. Each stage calls this once with its inputs/outputs/counts.
    """
    path = _run_logs_dir(ctx.settings) / f"run_manifest_{ctx.run_id}.json"
    manifest = read_json(path, default=None) or {
        "run_id": ctx.run_id,
        "run_time_utc": ctx.run_time_utc,
        "card_mode": ctx.settings.card_mode,
        "data_source": ctx.settings.data_source,
    }
    manifest.setdefault(section, {})
    manifest[section].update(updates)
    return write_json(path, manifest)


def write_health(ctx: RunContext, stage_status: dict[str, bool]) -> Path:
    """Write this run's health report: per-stage success plus accumulated
    warnings and errors. Answers "did the run work, and what should I look at?"
    """
    path = _run_logs_dir(ctx.settings) / f"pipeline_health_{ctx.run_id}.json"
    health = {
        "run_id": ctx.run_id,
        "run_time_utc": ctx.run_time_utc,
        "card_mode": ctx.settings.card_mode,
        "stages": stage_status,
        "success": all(stage_status.values()) if stage_status else False,
        "warnings": ctx.warnings,
        "errors": ctx.errors,
        "last_updated_utc": utc_now(),
    }
    return write_json(path, health)
