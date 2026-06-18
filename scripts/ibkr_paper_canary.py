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


def load_allowlist(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return {"SPY", "SGOV", "BIL", "MINT"}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(data, list):
        return {str(x).upper() for x in data}
    if isinstance(data, dict):
        values = data.get("tickers") or data.get("symbols") or []
        return {str(x).upper() for x in values}
    return set()


def repo_dir(project_root: Path, config: dict[str, Any], key: str, default: str) -> Path:
    path = project_root / config.get("repositories", {}).get(key, default)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_receipt(project_root: Path, config: dict[str, Any], payload: dict[str, Any]) -> Path:
    out = repo_dir(project_root, config, "execution_receipts", "repositories/execution_receipts")
    path = out / f"paper_canary_{utc_now().replace(':', '').replace('-', '')}.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def check_common(config: dict[str, Any], ticker: str, allowlist: set[str]) -> str:
    execution = config.get("execution", {}) or {}
    if config.get("allow_real_broker_execution", False):
        raise RuntimeError("allow_real_broker_execution must be false")
    if execution.get("allow_live_broker_orders", False):
        raise RuntimeError("execution.allow_live_broker_orders must be false")
    if ticker.upper() not in allowlist:
        raise RuntimeError(f"Ticker {ticker!r} is not in canary allowlist: {sorted(allowlist)}")
    account_var = str(execution.get("expected_account_id_env_var", "IBKR_PAPER_ACCOUNT_ID"))
    account = os.environ.get(account_var, "").strip()
    if not account:
        raise RuntimeError(f"Missing paper account env var: {account_var}")
    if not account.upper().startswith("DU"):
        raise RuntimeError(f"Refusing canary: {account_var} does not look like a paper account")
    return account


def run_ibkr_what_if(config: dict[str, Any], account: str, ticker: str, quantity: float) -> dict[str, Any]:
    try:
        from ib_insync import IB, MarketOrder, Stock  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install ib_insync on the paper host to run an IBKR canary") from exc

    ibkr = config.get("ibkr", {}) or {}
    host = str(ibkr.get("host", "127.0.0.1"))
    port = int(os.environ.get(str(ibkr.get("port_env_var", "IBKR_TWS_PORT")), "7497"))
    client_id = int(os.environ.get(str(ibkr.get("client_id_env_var", "IBKR_CLIENT_ID")), "88"))

    ib = IB()
    ib.connect(host, port, clientId=client_id, timeout=10)
    try:
        accounts = set(ib.managedAccounts())
        if account not in accounts:
            raise RuntimeError(f"Expected paper account {account} not available in connected IBKR session: {sorted(accounts)}")
        contract = Stock(ticker.upper(), "SMART", "USD")
        ib.qualifyContracts(contract)
        order = MarketOrder("BUY", quantity, account=account)
        order.whatIf = True
        trade = ib.placeOrder(contract, order)
        ib.sleep(2)
        return {
            "status": "what_if_submitted",
            "ibkr_order_id": getattr(trade.order, "orderId", None),
            "order_status": str(getattr(trade.orderStatus, "status", "unknown")),
            "init_margin_change": getattr(trade.orderState, "initMarginChange", None),
            "maint_margin_change": getattr(trade.orderState, "maintMarginChange", None),
            "equity_with_loan_change": getattr(trade.orderState, "equityWithLoanChange", None),
        }
    finally:
        ib.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a safe IBKR paper canary. what_if is the default and does not create an order.")
    parser.add_argument("--config", default="config/falnama_config.yaml")
    parser.add_argument("--mode", choices=["what_if", "tiny_paper_order"], default="what_if")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--quantity", type=float, default=1.0)
    parser.add_argument("--max-notional", type=float, default=250.0, help="Documented cap for manual tiny orders; notional sizing must be implemented before real paper submit.")
    parser.add_argument("--instrument-allowlist", default="config/canary_instruments.yaml")
    parser.add_argument("--confirm-paper-submit", action="store_true", help="Required for tiny_paper_order. what_if does not need this.")
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    config = load_config(project_root / args.config)
    allowlist = load_allowlist(project_root / args.instrument_allowlist)
    account = check_common(config, args.ticker, allowlist)

    payload: dict[str, Any] = {
        "time_utc": utc_now(),
        "broker": "IBKR",
        "execution_mode": "ibkr_paper",
        "paper_account_id": account,
        "mode": args.mode,
        "ticker": args.ticker.upper(),
        "quantity": args.quantity,
        "max_notional": args.max_notional,
        "live_broker_orders_enabled": False,
    }

    if args.mode == "what_if":
        payload.update(run_ibkr_what_if(config, account, args.ticker, args.quantity))
    else:
        # Intentional guard: tiny real paper orders require another implementation pass where
        # notional-to-quantity sizing, current price checks, and cancellation/reconciliation are complete.
        if not args.confirm_paper_submit:
            raise RuntimeError("tiny_paper_order requires --confirm-paper-submit. Prefer --mode what_if until sizing/reconciliation is implemented.")
        raise RuntimeError("tiny_paper_order is intentionally not implemented in this first runtime script. Use what_if canaries first.")

    path = write_receipt(project_root, config, payload)
    payload["receipt_path"] = str(path)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
