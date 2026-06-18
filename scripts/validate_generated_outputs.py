from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from jsonschema import Draft202012Validator, FormatChecker

try:
    from scenario_analysis_adapter import canonical_hash
except Exception:
    import hashlib

    def canonical_hash(card: dict[str, Any]) -> str:
        payload = json.loads(json.dumps(card, default=str))
        payload["card_hash"] = ""
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def repo_path(project_root: Path, config: dict[str, Any], key: str, default: str) -> Path:
    return project_root / config.get("repositories", {}).get(key, default)


def newest_files(path: Path, patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path.glob(pattern))
    return sorted(set(files), key=lambda p: p.stat().st_mtime, reverse=True)


def validate_json_object(obj: dict[str, Any], validator: Draft202012Validator, label: str, errors: list[str]) -> None:
    for error in sorted(validator.iter_errors(obj), key=lambda e: list(e.path)):
        loc = ".".join(str(x) for x in error.path) or "<root>"
        errors.append(f"{label}: schema error at {loc}: {error.message}")


def validate_index_cards(cards_dir: Path, schema_path: Path, allow_empty: bool) -> tuple[int, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    schema = load_json(schema_path)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    paths = sorted(cards_dir.glob("*.json")) if cards_dir.exists() else []
    if not paths:
        message = f"No index-card JSON files found in {cards_dir}"
        (warnings if allow_empty else errors).append(message)
        return 0, errors, warnings

    for path in paths:
        label = str(path)
        try:
            card = load_json(path)
        except Exception as exc:
            errors.append(f"{label}: could not parse JSON: {exc}")
            continue
        if not isinstance(card, dict):
            errors.append(f"{label}: card must be a JSON object")
            continue
        validate_json_object(card, validator, label, errors)
        if card.get("do_not_revise") is not True:
            errors.append(f"{label}: do_not_revise must be true")
        expected = canonical_hash(card)
        if card.get("card_hash") != expected:
            errors.append(f"{label}: card_hash mismatch, expected {expected}")
    return len(paths), errors, warnings


def coerce_row_for_jsonschema(row: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            cleaned[key] = None
        elif hasattr(value, "item"):
            cleaned[key] = value.item()
        else:
            cleaned[key] = value
    return cleaned


def validate_recommended_trade_file(path: Path, validator: Draft202012Validator, errors: list[str]) -> int:
    count = 0
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, sheet_name="Recommended Trades")
        records = df.to_dict(orient="records")
    elif path.suffix.lower() == ".csv":
        records = pd.read_csv(path).to_dict(orient="records")
    elif path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        payload = load_json(path)
        records = payload if isinstance(payload, list) else payload.get("recommended_trades", [])
    else:
        return 0

    for idx, row in enumerate(records, start=1):
        cleaned = coerce_row_for_jsonschema(row)
        validate_json_object(cleaned, validator, f"{path}: row {idx}", errors)
        count += 1
    return count


def validate_recommended_trades(trades_dir: Path, schema_path: Path, allow_empty: bool) -> tuple[int, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    schema = load_json(schema_path)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    files = newest_files(trades_dir, ["paper_recommended_trades_*.xlsx", "recommended_trades_*.csv", "recommended_trades_*.jsonl", "recommended_trades_*.json"])
    if not files:
        message = f"No recommended-trade output files found in {trades_dir}"
        (warnings if allow_empty else errors).append(message)
        return 0, errors, warnings

    latest = files[0]
    try:
        count = validate_recommended_trade_file(latest, validator, errors)
    except Exception as exc:
        errors.append(f"{latest}: could not validate recommended trades: {exc}")
        count = 0
    # Zero rows is valid: the strategy is intentionally conservative.
    return count, errors, warnings


def validate_csv_exists(path: Path, patterns: list[str], label: str, allow_empty: bool) -> tuple[int, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    files = newest_files(path, patterns)
    if not files:
        message = f"No {label} files found in {path}"
        (warnings if allow_empty else errors).append(message)
        return 0, errors, warnings
    latest = files[0]
    try:
        df = pd.read_csv(latest)
        return len(df), errors, warnings
    except Exception as exc:
        errors.append(f"{latest}: could not read {label}: {exc}")
        return 0, errors, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated Falnama artifacts.")
    parser.add_argument("--config", default="config/falnama_config.yaml")
    parser.add_argument("--index-card-schema", default="schemas/index_card_schema.json")
    parser.add_argument("--trade-schema", default="schemas/recommended_trade_schema.json")
    parser.add_argument("--artifacts", default="repositories")
    parser.add_argument("--allow-empty", action="store_true", help="Warn rather than fail when a class of artifact is absent.")
    parser.add_argument("--strict", action="store_true", help="Require all major pipeline artifact classes to exist.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    config = load_yaml(project_root / args.config)
    allow_empty = args.allow_empty or not args.strict

    cards_dir = repo_path(project_root, config, "index_cards", "repositories/index_cards")
    trades_dir = repo_path(project_root, config, "recommended_trades", "repositories/recommended_trades")
    rejected_dir = repo_path(project_root, config, "rejected_signals", "repositories/rejected_signals")
    relevant_dir = repo_path(project_root, config, "relevant_markets", "repositories/relevant_markets")

    errors: list[str] = []
    warnings: list[str] = []
    summary: dict[str, Any] = {}

    count, err, warn = validate_index_cards(cards_dir, project_root / args.index_card_schema, allow_empty)
    summary["index_cards"] = count
    errors += err
    warnings += warn

    count, err, warn = validate_recommended_trades(trades_dir, project_root / args.trade_schema, allow_empty)
    summary["recommended_trade_rows"] = count
    errors += err
    warnings += warn

    # Current notebooks write anomaly CSVs to repositories/relevant_markets. Support that until moved.
    count, err, warn = validate_csv_exists(relevant_dir, ["ranked_anomalies_*.csv"], "ranked anomaly", allow_empty)
    summary["ranked_anomaly_rows"] = count
    errors += err
    warnings += warn

    count, err, warn = validate_csv_exists(relevant_dir, ["strong_anomalies_*.csv"], "strong anomaly", allow_empty)
    summary["strong_anomaly_rows"] = count
    errors += err
    warnings += warn

    count, err, warn = validate_csv_exists(rejected_dir, ["rejected_signals_*.csv"], "rejected signal", allow_empty)
    summary["rejected_signal_rows"] = count
    errors += err
    warnings += warn

    result = {"ok": not errors, "summary": summary, "warnings": warnings, "errors": errors}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for warning in warnings:
            print(f"WARNING: {warning}")
        for error in errors:
            print(f"ERROR: {error}")
        print(json.dumps(summary, indent=2))
        print("Generated outputs OK" if not errors else "Generated output validation FAILED")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
