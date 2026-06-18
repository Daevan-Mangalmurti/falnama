from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import yaml
from jsonschema import Draft202012Validator

from scenario_analysis_adapter import generate_index_card, canonical_hash
from audit import (
    append_card_generation_events,
    update_run_manifest,
    update_pipeline_health,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def latest_file(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {directory}")
    return files[0]


def market_to_context(row: pd.Series, source_file: Path) -> dict:
    return {
        "market_id": row.get("market_id"),
        "market_slug": row.get("market_slug"),
        "market_name": row.get("market_name") or row.get("question"),
        "market_url": row.get("market_url"),
        "source_market_file": str(source_file),
        "outcomes": row.get("outcomes"),
        "volume": row.get("volume"),
        "liquidity": row.get("liquidity"),
        "end_date": row.get("end_date"),
    }


def card_path_for(card: dict, cards_dir: Path) -> Path:
    safe_id = (
        card["card_id"]
        .replace(":", "")
        .replace("/", "_")
        .replace("\\", "_")
    )
    return cards_dir / f"{safe_id}.json"


def write_exclusive_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Prevent accidental overwrite of an immutable card.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(tmp.read_text(encoding="utf-8"))
    tmp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--generation-mode", default=None)
    parser.add_argument("--exclusive-create-only", action="store_true")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    schema = json.loads((PROJECT_ROOT / args.schema).read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)

    run_time_utc = utc_now()
    run_id = run_time_utc.replace("-", "").replace(":", "")

    relevant_dir = PROJECT_ROOT / config["repositories"]["relevant_markets"]
    cards_dir = PROJECT_ROOT / config["repositories"]["index_cards"]

    market_file = latest_file(relevant_dir, "*.csv")
    markets = pd.read_csv(market_file)

    mode = args.generation_mode or config.get("card_generation_mode", "all_markets")
    if mode == "top_n":
        markets = markets.head(int(config.get("top_n_cards", 20)))
    elif mode == "manual_market_ids":
        manual_ids = set(map(str, config.get("manual_market_ids", [])))
        markets = markets[markets["market_id"].astype(str).isin(manual_ids)]
    elif mode != "all_markets":
        raise ValueError(f"Unknown card generation mode: {mode}")

    events = []
    created = []
    skipped = []
    errors = []

    for _, row in markets.iterrows():
        context = market_to_context(row, market_file)

        try:
            card = generate_index_card(
                context,
                config,
                created_time_utc=run_time_utc,
                generation_mode=mode,
            )

            expected_hash = canonical_hash(card)
            if card["card_hash"] != expected_hash:
                raise ValueError("card_hash does not match canonical hash")

            validator.validate(card)

            path = card_path_for(card, cards_dir)
            write_exclusive_json(path, card)

            created.append(str(path))
            events.append({
                "event": "created",
                "card_id": card["card_id"],
                "card_hash": card["card_hash"],
                "market_name": card["market_name"],
                "path": str(path),
                "created_time_utc": run_time_utc,
            })

        except FileExistsError:
            skipped.append(str(context.get("market_id") or context.get("market_slug")))
            events.append({
                "event": "skipped_existing",
                "market_name": context.get("market_name"),
                "created_time_utc": run_time_utc,
            })

        except Exception as exc:
            msg = f"{context.get('market_name')}: {exc}"
            errors.append(msg)
            events.append({
                "event": "error",
                "market_name": context.get("market_name"),
                "error": str(exc),
                "created_time_utc": run_time_utc,
            })

    append_card_generation_events(PROJECT_ROOT, config, events)

    update_run_manifest(
        PROJECT_ROOT,
        config,
        run_id,
        run_time_utc,
        "ai_communicator",
        {
            "cards_created": created,
            "cards_reused": [],
            "cards_skipped": skipped,
            "errors": errors,
        },
    )

    update_pipeline_health(
        PROJECT_ROOT,
        config,
        run_id,
        "ai_communicator",
        success=len(errors) == 0,
        errors=errors,
    )

    if errors:
        raise SystemExit(f"Scenario card generation completed with {len(errors)} errors")


if __name__ == "__main__":
    main()
    