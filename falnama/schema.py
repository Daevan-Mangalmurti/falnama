"""Validate artifacts against the JSON Schemas in schemas/.

WHAT:     Loads a named JSON Schema and checks an object against it.
CONSUMES: schemas/*.json
PRODUCES: nothing on disk — returns a list of human-readable problems (empty =
          valid), or raises if you ask it to.
REVIEWER: anyone who changes the shape of an index card or recommendation —
          update the schema here and validation flows everywhere automatically.
ROLE:     foundation. Index cards and recommendations are the artifacts a future
          reviewer relies on, so they are validated by an explicit contract
          rather than by hoping the producing code stayed correct.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from jsonschema import Draft202012Validator

from .config import find_project_root


class SchemaError(ValueError):
    """Raised when an artifact does not match its schema (with all problems listed)."""


@lru_cache(maxsize=None)
def load_schema(name: str) -> dict[str, Any]:
    """Load schemas/<name>_schema.json (cached). Pass the bare name, e.g. 'index_card'."""
    schema_path = find_project_root() / "schemas" / f"{name}_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    return json.loads(schema_path.read_text(encoding="utf-8"))


def validate(obj: Any, schema_name: str) -> list[str]:
    """Return a list of validation problems (empty list means the object is valid).

    Returning problems instead of raising lets a stage validate many artifacts,
    log every issue, and decide what to do — useful when one bad card shouldn't
    abort a whole run.
    """
    validator = Draft202012Validator(load_schema(schema_name))
    problems = []
    for error in sorted(validator.iter_errors(obj), key=lambda e: list(e.path)):
        location = "/".join(str(p) for p in error.path) or "(root)"
        problems.append(f"{location}: {error.message}")
    return problems


def validate_or_raise(obj: Any, schema_name: str) -> None:
    """Validate and raise SchemaError listing every problem if the object is invalid.

    Use this where an invalid artifact must never be written (e.g. an index card
    that downstream trade decisions will trust).
    """
    problems = validate(obj, schema_name)
    if problems:
        raise SchemaError(
            f"Object does not match schema '{schema_name}':\n  - " + "\n  - ".join(problems)
        )
