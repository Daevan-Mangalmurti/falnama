from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat
import yaml
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

DEFAULT_NOTEBOOKS = [
    "pipeline/01_market_selector.ipynb",
    "pipeline/02_anomaly_detector.ipynb",
    "pipeline/03_ai_communicator.ipynb",
    "pipeline/04_trading_paper_recommendations.ipynb",
]
STAGE_TO_NOTEBOOK = {
    "market_selector": DEFAULT_NOTEBOOKS[0],
    "selector": DEFAULT_NOTEBOOKS[0],
    "anomaly_detector": DEFAULT_NOTEBOOKS[1],
    "anomaly": DEFAULT_NOTEBOOKS[1],
    "ai_communicator": DEFAULT_NOTEBOOKS[2],
    "scenario_cards": DEFAULT_NOTEBOOKS[2],
    "trading_notebook": DEFAULT_NOTEBOOKS[3],
    "trade_recommender": DEFAULT_NOTEBOOKS[3],
    "trading": DEFAULT_NOTEBOOKS[3],
}
ALLOWED_MODES = {"smoke", "mock", "closed_historical", "live_research"}
ALLOWED_EXECUTIONS = {"dry_run", "ibkr_paper"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def compact_timestamp(value: str) -> str:
    return value.replace(":", "").replace("-", "")


def find_project_root(start: Path | None = None) -> Path:
    start = Path(start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "pipeline").exists() and (candidate / "scripts").exists():
            return candidate
        if (candidate / "config" / "falnama_config.yaml").exists():
            return candidate
    raise RuntimeError("Could not locate Falnama project root. Run from inside the repository.")


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML object: {path}")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def ensure_canonical_config(project_root: Path, requested_config: Path, mode: str, execution: str, persist_config: bool) -> tuple[Path, str | None]:
    canonical = project_root / "config" / "falnama_config.yaml"
    requested = requested_config if requested_config.is_absolute() else project_root / requested_config
    config = load_yaml(requested)

    # Apply safe runtime overrides in the ephemeral GitHub runner. These keep
    # smoke runs deterministic and prevent accidental live broker execution.
    config.setdefault("paper_trading_only", True)
    config["allow_real_broker_execution"] = False
    config.setdefault("index_cards_immutable", True)
    config.setdefault("repositories", {})
    config["repositories"].setdefault("relevant_markets", "repositories/relevant_markets")
    config["repositories"].setdefault("anomalies", "repositories/anomalies")
    config["repositories"].setdefault("index_cards", "repositories/index_cards")
    config["repositories"].setdefault("recommended_trades", "repositories/recommended_trades")
    config["repositories"].setdefault("run_logs", "repositories/run_logs")
    config["repositories"].setdefault("rejected_signals", "repositories/rejected_signals")
    config["repositories"].setdefault("trade_diagnostics", "repositories/trade_diagnostics")

    if mode in {"smoke", "mock"}:
        config["mock_mode"] = True
        config["scenario_analysis_mock_mode"] = True
        config.setdefault("selector", {})
        config["selector"]["enable_network_fetch"] = False
        config["selector"]["use_mock_if_no_inputs"] = True
        config.setdefault("anomaly_detector", {})
        config["anomaly_detector"]["use_mock_if_no_inputs"] = True
        config["backfill_testing_mode"] = False
    elif mode == "closed_historical":
        config.setdefault("selector", {})
        config["selector"].setdefault("closed_only", True)
        config["backfill_testing_mode"] = True
    elif mode == "live_research":
        config["backfill_testing_mode"] = False

    config.setdefault("execution", {})
    config["execution"]["mode"] = execution
    config["execution"]["allow_live_broker_orders"] = False
    if execution == "dry_run":
        config["execution"]["allow_paper_broker_orders"] = False
    elif execution == "ibkr_paper":
        config["execution"]["allow_paper_broker_orders"] = True
        config["execution"]["require_paper_account"] = True
        config["execution"]["require_account_allowlist"] = True

    backup_text: str | None = None
    if canonical.exists() and canonical.resolve() != requested.resolve() and not persist_config:
        backup_text = canonical.read_text(encoding="utf-8")
    write_yaml(canonical, config)
    return canonical, backup_text


def restore_config(canonical: Path, backup_text: str | None) -> None:
    if backup_text is not None:
        canonical.write_text(backup_text, encoding="utf-8")


def notebook_paths(project_root: Path, stages: list[str] | None, notebooks: list[str] | None) -> list[Path]:
    selected: list[str]
    if notebooks:
        selected = notebooks
    elif stages:
        selected = []
        for stage in stages:
            key = stage.strip().lower()
            if key not in STAGE_TO_NOTEBOOK:
                raise ValueError(f"Unknown pipeline stage: {stage}. Known: {sorted(STAGE_TO_NOTEBOOK)}")
            selected.append(STAGE_TO_NOTEBOOK[key])
    else:
        selected = DEFAULT_NOTEBOOKS
    paths = [(project_root / item).resolve() for item in selected]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing notebook(s): " + ", ".join(missing))
    return paths


def run_guardrails(project_root: Path, config_path: Path, mode: str, execution: str) -> None:
    script = project_root / "scripts" / "check_guardrails.py"
    if not script.exists():
        return
    cmd = [sys.executable, str(script), "--config", str(config_path), "--require-live-disabled"]
    if mode == "closed_historical":
        cmd.append("--allow-backfill")
    if execution == "ibkr_paper":
        cmd.append("--allow-paper-execution-config")
    subprocess.run(cmd, cwd=project_root, check=True)


def execute_notebook(
    notebook_path: Path,
    *,
    project_root: Path,
    output_dir: Path,
    timeout: int,
    kernel_name: str,
    allow_errors: bool,
) -> dict[str, Any]:
    started = utc_now()
    executed_path = output_dir / notebook_path.name.replace(".ipynb", "__executed.ipynb")
    output_dir.mkdir(parents=True, exist_ok=True)

    with notebook_path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    nb.metadata.setdefault("execution", {})
    nb.metadata["execution"].update({"falnama_started_utc": started, "source_path": str(notebook_path)})

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(project_root)}},
        allow_errors=allow_errors,
    )
    try:
        client.execute()
        status = "success"
        error = None
    except CellExecutionError as exc:
        status = "failed"
        error = str(exc)
        nbformat.write(nb, executed_path)
        raise
    except Exception as exc:
        status = "failed"
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        nbformat.write(nb, executed_path)
        raise
    finally:
        nb.metadata.setdefault("execution", {})
        nb.metadata["execution"].update({"falnama_finished_utc": utc_now()})
        nbformat.write(nb, executed_path)

    return {"notebook": str(notebook_path.relative_to(project_root)), "executed_path": str(executed_path.relative_to(project_root)), "started_utc": started, "finished_utc": utc_now(), "status": status, "error": error}


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute the Falnama notebook pipeline for CI, scheduled research runs, or paper-bot dry runs.")
    parser.add_argument("--config", default="config/falnama_config.yaml")
    parser.add_argument("--mode", default="smoke", choices=sorted(ALLOWED_MODES))
    parser.add_argument("--execution", default="dry_run", choices=sorted(ALLOWED_EXECUTIONS))
    parser.add_argument("--stage", action="append", help="Run one named stage. May be repeated.")
    parser.add_argument("--notebook", action="append", help="Run one notebook path. May be repeated.")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--kernel-name", default="python3")
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument("--skip-guardrails", action="store_true")
    parser.add_argument("--persist-config", action="store_true", help="Do not restore a pre-existing canonical config after execution.")
    args = parser.parse_args()
    # Honor FALNAMA_ALLOW_PAPER env var: if set to a truthy value, switch execution to ibkr_paper
# before canonical config and guardrails are applied. Recognize common truthy values.
env_allow_paper = os.environ.get("FALNAMA_ALLOW_PAPER", "").strip().lower()
if env_allow_paper in {"1", "true", "yes", "y", "on"}:
    # Only override if the user didn't explicitly request another execution mode
    default_execution = parser.get_default("execution")
    if args.execution == default_execution:
        args.execution = "ibkr_paper"
        print("FALNAMA_ALLOW_PAPER set — switching execution to 'ibkr_paper'", flush=True)

# Ensure downstream code (notebooks, subprocesses) can inspect this state
os.environ["FALNAMA_ALLOW_PAPER"] = "true" if args.execution == "ibkr_paper" else "false"

    project_root = find_project_root()
    run_time = utc_now()
    run_id = compact_timestamp(run_time)
    output_dir = project_root / "repositories" / "run_logs" / "executed_notebooks" / run_id
    summary_path = project_root / "repositories" / "run_logs" / f"run_pipeline_summary_{run_id}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["FALNAMA_RUN_MODE"] = args.mode
    os.environ["FALNAMA_EXECUTION_MODE"] = args.execution
    os.environ["FALNAMA_ALLOW_LIVE_BROKER_ORDERS"] = "false"

    canonical_config, backup_text = ensure_canonical_config(project_root, Path(args.config), args.mode, args.execution, args.persist_config)
    results: list[dict[str, Any]] = []
    failed = False

    try:
        if not args.skip_guardrails:
            run_guardrails(project_root, canonical_config, args.mode, args.execution)

        paths = notebook_paths(project_root, args.stage, args.notebook)
        for notebook in paths:
            print(f"\n=== Executing {notebook.relative_to(project_root)} ===", flush=True)
            try:
                result = execute_notebook(
                    notebook,
                    project_root=project_root,
                    output_dir=output_dir,
                    timeout=args.timeout,
                    kernel_name=args.kernel_name,
                    allow_errors=args.allow_errors,
                )
                results.append(result)
                print(f"OK: {result['executed_path']}", flush=True)
            except Exception as exc:
                failed = True
                result = {
                    "notebook": str(notebook.relative_to(project_root)),
                    "executed_path": str((output_dir / notebook.name.replace(".ipynb", "__executed.ipynb")).relative_to(project_root)),
                    "started_utc": None,
                    "finished_utc": utc_now(),
                    "status": "failed",
                    "error": repr(exc),
                }
                results.append(result)
                print(f"FAILED: {notebook.relative_to(project_root)}: {exc}", file=sys.stderr, flush=True)
                if not args.allow_errors:
                    break
    finally:
        summary = {"run_id": run_id, "run_time_utc": run_time, "mode": args.mode, "execution": args.execution, "config": str(canonical_config.relative_to(project_root)), "results": results, "success": not failed}
        summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\nPipeline summary: {summary_path.relative_to(project_root)}")
        if not args.persist_config:
            restore_config(canonical_config, backup_text)

    if failed and not args.allow_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
