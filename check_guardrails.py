from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

TRUTHY = {"1", "true", "yes", "y", "on"}
LIVE_MODES = {"live", "real", "production", "ibkr_live", "live_trading", "broker_live"}
PAPER_MODES = {"dry_run", "mock", "closed_historical", "live_research", "ibkr_paper"}


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in TRUTHY


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML object: {path}")
    return data


def nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def add_error(errors: list[str], message: str) -> None:
    errors.append(f"ERROR: {message}")


def add_warning(warnings: list[str], message: str) -> None:
    warnings.append(f"WARNING: {message}")


def check_guardrails(config: dict[str, Any], args: argparse.Namespace) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    execution = config.get("execution", {}) if isinstance(config.get("execution"), dict) else {}
    risk = config.get("risk", {}) if isinstance(config.get("risk"), dict) else {}
    ibkr = config.get("ibkr", {}) if isinstance(config.get("ibkr"), dict) else {}

    paper_trading_only = config.get("paper_trading_only", True)
    if paper_trading_only is not True:
        add_error(errors, "paper_trading_only must be true when present.")

    legacy_real = bool(config.get("allow_real_broker_execution", False))
    nested_live = bool(execution.get("allow_live_broker_orders", False))
    env_live = truthy(os.environ.get("FALNAMA_ALLOW_LIVE_BROKER_ORDERS"))
    if legacy_real or nested_live or env_live:
        add_error(errors, "live broker execution/orders must be disabled.")

    mode = str(execution.get("mode") or os.environ.get("FALNAMA_EXECUTION_MODE") or "dry_run").strip().lower()
    if mode in LIVE_MODES or "live" in mode and mode not in {"live_research"}:
        add_error(errors, f"execution mode looks live-like and is forbidden: {mode}")
    if mode not in PAPER_MODES:
        add_warning(warnings, f"Unrecognized execution mode: {mode}. Expected one of {sorted(PAPER_MODES)}.")

    if config.get("index_cards_immutable", True) is not True:
        add_error(errors, "index_cards_immutable must be true.")

    if config.get("backfill_testing_mode", False) and not args.allow_backfill:
        add_error(errors, "backfill_testing_mode is true outside an explicit backfill/calibration workflow.")

    if args.require_live_disabled and (legacy_real or nested_live):
        add_error(errors, "--require-live-disabled was set and live broker execution is enabled.")

    allow_paper = bool(execution.get("allow_paper_broker_orders", False))
    if allow_paper:
        if not args.allow_paper_execution_config:
            add_error(errors, "Config allows paper broker orders, but this workflow did not opt into paper execution config.")
        if bool(execution.get("allow_live_broker_orders", False)):
            add_error(errors, "allow_paper_broker_orders cannot be paired with allow_live_broker_orders.")
        if execution.get("require_paper_account", True) is not True:
            add_error(errors, "IBKR paper execution must require a paper account.")
        if execution.get("require_account_allowlist", True) is not True:
            add_error(errors, "IBKR paper execution must require an account allowlist.")
        if ibkr.get("paper_only", True) is not True:
            add_error(errors, "ibkr.paper_only must be true for paper execution.")
        expected_account_env = execution.get("expected_account_id_env_var")
        if not expected_account_env:
            add_error(errors, "execution.expected_account_id_env_var is required for paper execution.")

    kill_switch_env = execution.get("kill_switch_env_var") or "FALNAMA_KILL_SWITCH"
    if truthy(os.environ.get(str(kill_switch_env))):
        add_error(errors, f"Kill switch is active via ${kill_switch_env}.")

    max_position = config.get("max_position_usd", risk.get("max_position_usd"))
    if max_position is not None:
        try:
            if float(max_position) <= 0:
                add_error(errors, "max_position_usd must be positive.")
        except Exception:
            add_error(errors, "max_position_usd must be numeric.")

    max_order_notional = risk.get("max_order_notional_usd")
    if allow_paper and max_order_notional is not None:
        try:
            if float(max_order_notional) <= 0:
                add_error(errors, "risk.max_order_notional_usd must be positive.")
        except Exception:
            add_error(errors, "risk.max_order_notional_usd must be numeric.")

    endpoint = config.get("scenario_analysis_api_endpoint")
    scenario_mock = bool(config.get("scenario_analysis_mock_mode", config.get("mock_mode", True)))
    if not scenario_mock and endpoint in (None, "", "PLACEHOLDER"):
        add_error(errors, "scenario_analysis_mock_mode is false but scenario_analysis_api_endpoint is not configured.")
    key_env = config.get("scenario_analysis_api_key_env_var", "SCENARIO_ANALYSIS_GPT_API_KEY")
    if args.require_scenario_key and not os.environ.get(str(key_env)):
        add_error(errors, f"Missing required Scenario Analysis GPT API key env var: {key_env}")
    elif not scenario_mock and endpoint not in (None, "", "PLACEHOLDER") and not os.environ.get(str(key_env)):
        add_warning(warnings, f"Scenario Analysis GPT appears enabled, but ${key_env} is not set in this environment.")

    selector = config.get("selector", {}) if isinstance(config.get("selector"), dict) else {}
    if args.ci and selector.get("enable_network_fetch", False):
        add_warning(warnings, "selector.enable_network_fetch is true in CI; smoke tests may depend on external network state.")

    return errors, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Falnama safety and execution guardrails.")
    parser.add_argument("config_positional", nargs="?", help="Optional config path for legacy workflow invocations.")
    parser.add_argument("--config", dest="config_flag", help="Path to falnama_config.yaml.")
    parser.add_argument("--require-live-disabled", action="store_true")
    parser.add_argument("--allow-paper-execution-config", action="store_true")
    parser.add_argument("--allow-backfill", action="store_true")
    parser.add_argument("--require-scenario-key", action="store_true")
    parser.add_argument("--ci", action="store_true", help="Apply CI-oriented warnings.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable result JSON.")
    args = parser.parse_args()

    config_path = Path(args.config_flag or args.config_positional or "config/falnama_config.yaml")
    config = load_yaml(config_path)
    errors, warnings = check_guardrails(config, args)

    result = {"config": str(config_path), "ok": not errors, "errors": errors, "warnings": warnings}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for warning in warnings:
            print(warning)
        for error in errors:
            print(error)
        print("Guardrails OK" if not errors else "Guardrails FAILED")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
