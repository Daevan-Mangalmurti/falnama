from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML object")
    return data


def repo_dir(project_root: Path, config: dict[str, Any], key: str, default: str) -> Path:
    path = project_root / config.get("repositories", {}).get(key, default)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    elif path.suffix == ".json":
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def collect_receipts(receipts_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(receipts_dir.glob("*.json*")):
        try:
            for record in read_json_records(path):
                record.setdefault("_source_file", str(path))
                out.append(record)
        except Exception:
            continue
    return out


def fetch_ibkr_state(config: dict[str, Any]) -> dict[str, Any]:
    try:
        from ib_insync import IB  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install ib_insync on the paper host to reconcile IBKR state") from exc

    execution = config.get("execution", {}) or {}
    ibkr = config.get("ibkr", {}) or {}
    account_var = str(execution.get("expected_account_id_env_var", "IBKR_PAPER_ACCOUNT_ID"))
    account = os.environ.get(account_var, "").strip()
    if not account or not account.upper().startswith("DU"):
        raise RuntimeError(f"Missing or non-paper account in {account_var}")

    host = str(ibkr.get("host", "127.0.0.1"))
    port = int(os.environ.get(str(ibkr.get("port_env_var", "IBKR_TWS_PORT")), "7497"))
    client_id = int(os.environ.get(str(ibkr.get("client_id_env_var", "IBKR_CLIENT_ID")), "89"))

    ib = IB()
    ib.connect(host, port, clientId=client_id, timeout=10)
    try:
        accounts = set(ib.managedAccounts())
        if account not in accounts:
            raise RuntimeError(f"Expected paper account {account} not found; connected accounts={sorted(accounts)}")
        positions = [
            {
                "account": p.account,
                "symbol": getattr(p.contract, "symbol", None),
                "secType": getattr(p.contract, "secType", None),
                "currency": getattr(p.contract, "currency", None),
                "position": p.position,
                "avgCost": p.avgCost,
            }
            for p in ib.positions()
            if p.account == account
        ]
        open_orders = [
            {
                "orderId": getattr(t.order, "orderId", None),
                "symbol": getattr(t.contract, "symbol", None),
                "action": getattr(t.order, "action", None),
                "totalQuantity": getattr(t.order, "totalQuantity", None),
                "status": getattr(t.orderStatus, "status", None),
            }
            for t in ib.openTrades()
        ]
        return {"paper_account_id": account, "positions": positions, "open_orders": open_orders}
    finally:
        ib.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile Falnama execution receipts against the IBKR paper account.")
    parser.add_argument("--config", default="config/falnama_config.yaml")
    parser.add_argument("--canary-only", action="store_true")
    parser.add_argument("--allow-offline", action="store_true", help="Write a receipts-only report if IBKR is unavailable.")
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    config = load_config(project_root / args.config)
    receipts_dir = repo_dir(project_root, config, "execution_receipts", "repositories/execution_receipts")
    reports_dir = repo_dir(project_root, config, "broker_reconciliation", "repositories/broker_reconciliation")
    receipts = collect_receipts(receipts_dir)
    if args.canary_only:
        receipts = [r for r in receipts if "canary" in str(r.get("_source_file", "")) or r.get("mode") in {"what_if", "tiny_paper_order"}]

    errors: list[str] = []
    ibkr_state: dict[str, Any] | None = None
    try:
        ibkr_state = fetch_ibkr_state(config)
    except Exception as exc:
        if args.allow_offline:
            errors.append(f"IBKR unavailable; receipts-only reconciliation: {exc}")
        else:
            raise

    report = {
        "time_utc": utc_now(),
        "status": "ok" if not errors else "warning",
        "canary_only": args.canary_only,
        "receipt_count": len(receipts),
        "receipts": receipts[-50:],
        "ibkr_state": ibkr_state,
        "errors": errors,
    }
    path = reports_dir / f"reconciliation_{utc_now().replace(':', '').replace('-', '')}.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"report_path": str(path), "status": report["status"], "receipt_count": len(receipts), "errors": errors}, indent=2))
    if errors and not args.allow_offline:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
