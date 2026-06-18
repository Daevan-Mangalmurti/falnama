from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


def canonical_hash(card: dict[str, Any]) -> str:
    payload = json.loads(json.dumps(card, default=str))
    payload["card_hash"] = ""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_diff_name_status(base_ref: str, pathspec: str) -> list[tuple[str, str]]:
    cmd = ["git", "diff", "--name-status", f"{base_ref}...HEAD", "--", pathspec]
    try:
        completed = subprocess.run(cmd, text=True, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Could not compute git diff against {base_ref}: {exc.stderr.strip() or exc.stdout.strip()}") from exc
    changes: list[tuple[str, str]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        path = parts[-1]
        changes.append((status, path))
    return changes


def check_git_immutability(base_ref: str, cards_dir: Path, allow_deletes: bool = False) -> list[str]:
    errors: list[str] = []
    pathspec = str(cards_dir).rstrip("/")
    for status, path in git_diff_name_status(base_ref, pathspec):
        code = status[0]
        if code == "A":
            continue
        if code == "D" and allow_deletes:
            continue
        errors.append(f"Immutable index-card file changed with status {status}: {path}. Existing cards may only be added, not modified, renamed, or deleted.")
    return errors


def validate_cards(cards_dir: Path, schema_path: Path, allow_empty: bool, require_immutable: bool, verify_hashes: bool) -> tuple[int, list[str]]:
    errors: list[str] = []
    schema = load_json(schema_path)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    paths = sorted(cards_dir.glob("*.json")) if cards_dir.exists() else []
    if not paths and not allow_empty:
        errors.append(f"No index-card JSON files found in {cards_dir}")
        return 0, errors

    for path in paths:
        try:
            card = load_json(path)
        except Exception as exc:
            errors.append(f"{path}: invalid JSON: {exc}")
            continue
        if not isinstance(card, dict):
            errors.append(f"{path}: card must be a JSON object")
            continue
        for error in sorted(validator.iter_errors(card), key=lambda e: list(e.path)):
            loc = ".".join(str(x) for x in error.path) or "<root>"
            errors.append(f"{path}: schema error at {loc}: {error.message}")
        if require_immutable and card.get("do_not_revise") is not True:
            errors.append(f"{path}: do_not_revise must be true")
        if verify_hashes:
            expected = canonical_hash(card)
            if card.get("card_hash") != expected:
                errors.append(f"{path}: card_hash mismatch, expected {expected}")
    return len(paths), errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate immutable Falnama index cards and prevent edits to existing cards.")
    parser.add_argument("--schema", default="schemas/index_card_schema.json")
    parser.add_argument("--cards", default="repositories/index_cards")
    parser.add_argument("--base-ref", default=None, help="Git base ref to compare against, e.g. origin/main.")
    parser.add_argument("--skip-git-diff", action="store_true")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--allow-deletes", action="store_true")
    parser.add_argument("--require-immutable", action="store_true", default=True)
    parser.add_argument("--verify-hashes", action="store_true", default=True)
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    cards_dir = (project_root / args.cards).resolve()
    schema_path = (project_root / args.schema).resolve()

    errors: list[str] = []
    count, validation_errors = validate_cards(cards_dir, schema_path, args.allow_empty, args.require_immutable, args.verify_hashes)
    errors.extend(validation_errors)

    if args.base_ref and not args.skip_git_diff:
        # Use path relative to repo root for the pathspec.
        try:
            rel_cards = str(cards_dir.relative_to(project_root))
        except ValueError:
            rel_cards = str(cards_dir)
        errors.extend(check_git_immutability(args.base_ref, Path(rel_cards), allow_deletes=args.allow_deletes))

    print(f"Checked {count} index-card file(s) in {cards_dir}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Index-card immutability OK")


if __name__ == "__main__":
    main()
