from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

TRUTHY = {"1", "true", "yes", "y", "on"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML object")
    return data


def env_truthy(name: str, default: str = "") -> bool:
    return os.environ.get(name, default).strip().lower() in TRUTHY


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def check_socket(host: str, port: int, timeout: float) -> tuple[bool, str | None]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, None
    except OSError as exc:
        return False, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight checks for the Falnama IBKR paper-trading host.")
    parser.add_argument("--config", default="config/falnama_config.yaml")
    parser.add_argument("--require-gateway-reachable", action="store_true")
    parser.add_argument("--require-paper-account-env", action="store_true")
    parser.add_argument("--forbid-live-account", action="store_true")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--socket-timeout", type=float, default=5.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = load_config(config_path)

    execution = config.get("execution", {}) or {}
    ibkr = config.get("ibkr", {}) or {}
    errors: list[str] = []
    warnings: list[str] = []

    require(config.get("allow_real_broker_execution", False) is False, "Top-level allow_real_broker_execution must be false.", errors)
    require(execution.get("allow_live_broker_orders", False) is False, "execution.allow_live_broker_orders must be false.", errors)
    require(ibkr.get("paper_only", True) is True, "ibkr.paper_only must be true.", errors)

    kill_var = str(execution.get("kill_switch_env_var", "FALNAMA_KILL_SWITCH"))
    if env_truthy(kill_var):
        errors.append(f"Kill switch is active: {kill_var}")

    account_env_var = str(execution.get("expected_account_id_env_var", "IBKR_PAPER_ACCOUNT_ID"))
    expected_account = os.environ.get(account_env_var, "").strip()
    if args.require_paper_account_env or execution.get("require_paper_account", False):
        require(bool(expected_account), f"Missing required paper account env var: {account_env_var}", errors)
    if expected_account and args.forbid_live_account and not expected_account.upper().startswith("DU"):
        errors.append(
            f"Configured account {account_env_var}={expected_account!r} does not look like an IBKR paper account. "
            "IBKR paper account IDs commonly start with 'DU'."
        )

    port_env_var = str(ibkr.get("port_env_var", "IBKR_TWS_PORT"))
    host = args.host or str(ibkr.get("host", "127.0.0.1"))
    port_raw = args.port if args.port is not None else os.environ.get(port_env_var, "")
    port: int | None
    try:
        port = int(port_raw) if port_raw not in (None, "") else None
    except ValueError:
        port = None
        errors.append(f"Invalid IBKR port in {port_env_var}: {port_raw!r}")

    socket_ok: bool | None = None
    socket_error: str | None = None
    if args.require_gateway_reachable:
        if port is None:
            errors.append(f"IBKR TWS/Gateway port is required. Set {port_env_var} or pass --port.")
        else:
            socket_ok, socket_error = check_socket(host, port, args.socket_timeout)
            if not socket_ok:
                errors.append(f"Could not reach IBKR TWS/Gateway at {host}:{port}: {socket_error}")

    result = {
        "checked_time_utc": utc_now(),
        "config": str(config_path),
        "execution_mode": execution.get("mode"),
        "allow_live_broker_orders": execution.get("allow_live_broker_orders", False),
        "allow_paper_broker_orders": execution.get("allow_paper_broker_orders", False),
        "paper_only": ibkr.get("paper_only", True),
        "account_env_var": account_env_var,
        "paper_account_present": bool(expected_account),
        "host": host,
        "port": port,
        "gateway_socket_ok": socket_ok,
        "warnings": warnings,
        "errors": errors,
        "ok": not errors,
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
