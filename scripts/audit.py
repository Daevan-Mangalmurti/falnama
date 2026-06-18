from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NOTEBOOK_TO_HEALTH_KEY = {
    "market_selector": "selector_success",
    "anomaly_detector": "anomaly_detector_success",
    "ai_communicator": "communicator_success",
    "trading_notebook": "trading_success",
}


def mock_mode_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("mock_mode", config.get("scenario_analysis_mock_mode", False)))


def compact_timestamp(run_time_utc: str) -> str:
    return run_time_utc.replace(":", "").replace("-", "")


def repo_path(project_root: Path, config: dict[str, Any], key: str, default: str) -> Path:
    path = project_root / config.get("repositories", {}).get(key, default)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_audit_repositories(project_root: Path, config: dict[str, Any]) -> None:
    for key, default in {
        "anomalies": "repositories/anomalies",
        "run_logs": "repositories/run_logs",
        "index_cards": "repositories/index_cards",
        "trade_diagnostics": "repositories/trade_diagnostics",
    }.items():
        repo_path(project_root, config, key, default)
    (repo_path(project_root, config, "run_logs", "repositories/run_logs") / "manifests").mkdir(parents=True, exist_ok=True)


def get_or_create_run_id(project_root: Path, config: dict[str, Any], notebook_name: str, run_time_utc: str) -> str:
    run_logs = repo_path(project_root, config, "run_logs", "repositories/run_logs")
    active_path = run_logs / "active_run_id.txt"
    if notebook_name == "market_selector" or not active_path.exists():
        run_id = compact_timestamp(run_time_utc)
        active_path.write_text(run_id + "\n", encoding="utf-8")
        return run_id
    run_id = active_path.read_text(encoding="utf-8").strip()
    if not run_id:
        run_id = compact_timestamp(run_time_utc)
        active_path.write_text(run_id + "\n", encoding="utf-8")
    return run_id


def default_manifest(run_id: str, run_time_utc: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_time_utc": run_time_utc,
        "mock_mode": mock_mode_enabled(config),
        "market_selector": {
            "input_files": [],
            "output_files": [],
            "market_count": 0,
        },
        "anomaly_detector": {
            "input_files": [],
            "output_files": [],
            "ranked_anomaly_count": 0,
            "strong_anomaly_count": 0,
        },
        "ai_communicator": {
            "cards_created": [],
            "cards_reused": [],
            "cards_skipped": [],
        },
        "trading_notebook": {
            "recommended_trade_file": "",
            "recommended_trade_count": 0,
            "rejected_signal_file": "",
            "rejected_signal_count": 0,
        },
    }


def manifest_path(project_root: Path, config: dict[str, Any], run_id: str) -> Path:
    path = repo_path(project_root, config, "run_logs", "repositories/run_logs") / "manifests" / f"run_manifest_{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def update_run_manifest(
    project_root: Path,
    config: dict[str, Any],
    run_id: str,
    run_time_utc: str,
    section_name: str,
    section_updates: dict[str, Any],
) -> Path:
    path = manifest_path(project_root, config, run_id)
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else default_manifest(run_id, run_time_utc, config)
    manifest["mock_mode"] = mock_mode_enabled(config)
    manifest["run_id"] = run_id
    manifest.setdefault(section_name, {}).update(section_updates)
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def default_health(run_id: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "mock_mode": mock_mode_enabled(config),
        "selector_success": False,
        "anomaly_detector_success": False,
        "communicator_success": False,
        "trading_success": False,
        "warnings": [],
        "errors": [],
        "elapsed_seconds": 0,
    }


def pipeline_health_path(project_root: Path, config: dict[str, Any], run_id: str) -> Path:
    path = repo_path(project_root, config, "run_logs", "repositories/run_logs") / f"pipeline_health_{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def update_pipeline_health(
    project_root: Path,
    config: dict[str, Any],
    run_id: str,
    section_name: str,
    *,
    success: bool,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
    started_monotonic: float | None = None,
) -> Path:
    path = pipeline_health_path(project_root, config, run_id)
    health = json.loads(path.read_text(encoding="utf-8")) if path.exists() else default_health(run_id, config)
    health["mock_mode"] = mock_mode_enabled(config)
    health["last_updated_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    health_key = NOTEBOOK_TO_HEALTH_KEY.get(section_name)
    if health_key:
        health[health_key] = bool(success)
    for warning in warnings or []:
        if warning not in health["warnings"]:
            health["warnings"].append(warning)
    for error in errors or []:
        if error not in health["errors"]:
            health["errors"].append(error)
    if started_monotonic is not None:
        health["elapsed_seconds"] = round(float(health.get("elapsed_seconds", 0)) + max(0.0, time.monotonic() - started_monotonic), 3)
    path.write_text(json.dumps(health, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def append_card_generation_events(project_root: Path, config: dict[str, Any], events: list[dict[str, Any]]) -> Path:
    path = repo_path(project_root, config, "index_cards", "repositories/index_cards") / "card_generation_manifest.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    if not isinstance(existing, list):
        existing = []
    existing.extend(events)
    path.write_text(json.dumps(existing, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")
    return path
