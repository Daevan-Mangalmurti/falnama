"""Load and validate Falnama's configuration.

WHAT:     Reads config/falnama.yaml and turns it into one typed `Settings`
          object that the rest of the pipeline uses.
CONSUMES: config/falnama.yaml
PRODUCES: a validated `Settings` instance (nothing is written to disk here)
REVIEWER: anyone changing pipeline behavior — this is the single control panel
ROLE:     foundation. Every stage receives the same `Settings`, so behavior is
          fully determined by one file plus the input data.

Design choice: configuration is a plain validated dictionary wrapped in a small
typed object. The wrapper exposes the handful of settings used everywhere
(safety flags, the LLM model, output paths) as named properties, and lets each
stage read its own nested section (`settings.selector`, `settings.anomaly`, ...)
as a dictionary. This stays readable for collaborators without a deep typing
background, while still failing loudly on the mistakes that matter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# The settings we insist on. If any are missing or wrong, we fail at startup
# with a clear message rather than producing a subtly broken run.
_VALID_CARD_MODES = {"mock", "live"}
_VALID_DATA_SOURCES = {"fixtures", "live"}
_VALID_EXECUTION_MODES = {"none", "paper_sim"}
_VALID_SCREENER_MODES = {"mock", "live"}


class ConfigError(ValueError):
    """Raised when the config is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class Settings:
    """A validated view over config/falnama.yaml.

    `raw` holds the full parsed YAML. `project_root` is the repository root, used
    to resolve the relative output paths. Convenience properties cover the
    cross-cutting settings; per-stage sections are read as dictionaries.
    """

    raw: dict[str, Any]
    project_root: Path

    # ---- cross-cutting safety / mode flags ---------------------------------
    @property
    def paper_trading_only(self) -> bool:
        return bool(self.raw.get("paper_trading_only", True))

    @property
    def backfill_mode(self) -> bool:
        return bool(self.raw.get("backfill_mode", False))

    @property
    def card_mode(self) -> str:
        return str(self.raw.get("card_mode", "mock"))

    @property
    def data_source(self) -> str:
        return str(self.section("data").get("source", "fixtures"))

    @property
    def execution_mode(self) -> str:
        return str(self.section("execution").get("mode", "none"))

    # ---- LLM seam ----------------------------------------------------------
    @property
    def llm(self) -> dict[str, Any]:
        return self.section("llm")

    # ---- per-stage sections (read as dictionaries) -------------------------
    def section(self, name: str) -> dict[str, Any]:
        """Return a named config section as a dict (empty dict if absent)."""
        value = self.raw.get(name, {})
        return value if isinstance(value, dict) else {}

    @property
    def selector(self) -> dict[str, Any]:
        return self.section("selector")

    @property
    def screener(self) -> dict[str, Any]:
        return self.section("screener")

    @property
    def anomaly(self) -> dict[str, Any]:
        return self.section("anomaly")

    @property
    def cards(self) -> dict[str, Any]:
        return self.section("cards")

    @property
    def recommend(self) -> dict[str, Any]:
        return self.section("recommend")

    @property
    def newslag(self) -> dict[str, Any]:
        return self.section("newslag")

    @property
    def data(self) -> dict[str, Any]:
        return self.section("data")

    # ---- output paths ------------------------------------------------------
    def output_dir(self, key: str) -> Path:
        """Resolve a named output directory (e.g. 'index_cards') to an absolute
        path, creating it if needed. Falls back to outputs/<key>."""
        outputs = self.section("outputs")
        rel = outputs.get(key, f"outputs/{key}")
        path = (self.project_root / rel).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def snapshots_dir(self) -> Path:
        path = (self.project_root / "data" / "snapshots").resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def fixtures_dir(self) -> Path:
        return (self.project_root / "data" / "fixtures").resolve()


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from `start` until we find the repo root (the folder holding
    config/falnama.yaml). Lets scripts and notebooks run from any subdirectory."""
    start = Path(start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "config" / "falnama.yaml").exists():
            return candidate
    raise ConfigError(
        "Could not locate the Falnama project root (no config/falnama.yaml found "
        f"at or above {start})."
    )


def load_config(path: str | Path | None = None) -> Settings:
    """Load, validate, and return the settings.

    With no argument, finds config/falnama.yaml automatically. Raises
    ConfigError with an actionable message if anything is wrong.
    """
    if path is None:
        project_root = find_project_root()
        path = project_root / "config" / "falnama.yaml"
    else:
        path = Path(path).resolve()
        project_root = path.parent.parent  # <root>/config/falnama.yaml

    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Config must be a YAML mapping at the top level: {path}")

    settings = Settings(raw=raw, project_root=project_root)
    _validate(settings)
    return settings


def _validate(settings: Settings) -> None:
    """Check the invariants that protect the integrity of a run."""
    errors: list[str] = []

    # The core safety tripwire. Falnama is paper-only by design.
    if settings.paper_trading_only is not True:
        errors.append("paper_trading_only must be true (Falnama never trades for real).")

    if settings.card_mode not in _VALID_CARD_MODES:
        errors.append(f"card_mode must be one of {sorted(_VALID_CARD_MODES)}.")

    if settings.data_source not in _VALID_DATA_SOURCES:
        errors.append(f"data.source must be one of {sorted(_VALID_DATA_SOURCES)}.")

    if settings.execution_mode not in _VALID_EXECUTION_MODES:
        errors.append(f"execution.mode must be one of {sorted(_VALID_EXECUTION_MODES)}.")

    screener_mode = str(settings.screener.get("mode", "mock"))
    if screener_mode not in _VALID_SCREENER_MODES:
        errors.append(f"screener.mode must be one of {sorted(_VALID_SCREENER_MODES)}.")

    newslag_mode = str(settings.newslag.get("mode", "mock"))
    if newslag_mode not in _VALID_SCREENER_MODES:  # same mock|live set
        errors.append(f"newslag.mode must be one of {sorted(_VALID_SCREENER_MODES)}.")

    # If real cards, a live screen, or live news-lag are requested, the model and
    # key var must be named.
    needs_llm = (settings.card_mode == "live"
                 or newslag_mode == "live"
                 or (settings.screener.get("enabled", False) and screener_mode == "live"))
    if needs_llm:
        if not settings.llm.get("model"):
            errors.append("llm.model is required when card_mode or newslag.mode is 'live'.")
        if not settings.llm.get("api_key_env_var"):
            errors.append("llm.api_key_env_var is required for live LLM calls.")

    if errors:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))
