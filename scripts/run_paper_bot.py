from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

TRUTHY = {"1", "true", "yes", "y", "on"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def compact_timestamp(value: str) -> str:
    return value.replace("-", "").replace(":", "")


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML object")
    return data


def repo_path(project_root: Path, config: dict[str, Any], key: str, default: str) -> Path:
    path = project_root / config.get("repositories", {}).get(key, default)
    path.mkdir(parents=True, exist_ok=True)
    return path


def latest_file(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def read_latest_recommendations(trades_dir: Path) -> tuple[pd.DataFrame, str | None]:
    xlsx = latest_file(trades_dir, "paper_recommended_trades_*.xlsx")
    if xlsx is not None:
        try:
            return pd.read_excel(xlsx, sheet_name="Recommended Trades"), str(xlsx)
        except ValueError:
            return pd.read_excel(xlsx), str(xlsx)
    csv = latest_file(trades_dir, "paper_recommended_trades_*.csv")
    if csv is not None:
        return pd.read_csv(csv), str(csv)
    return pd.DataFrame(), None


def first_present(row: dict[str, Any], names: list[str], default: Any = None) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and value == value and str(value).strip() != "":
            return value
    return default


def idempotency_key(row: dict[str, Any], run_id: str) -> str:
    pieces = [
        str(first_present(row, ["card_hash"], "")),
        str(first_present(row, ["ticker", "tradable_instrument_name"], "")),
        str(first_present(row, ["paper_trade_action", "expected_direction"], "")),
        str(first_present(row, ["trigger_time_utc", "run_time_utc"], "")),
        run_id,
    ]
    return hashlib.sha256("|".join(pieces).encode("utf-8")).hexdigest()


def build_order_intent(row: dict[str, Any], *, config: dict[str, Any], run_id: str, paper_account_id: str) -> dict[str, Any]:
    action_raw = str(first_present(row, ["paper_trade_action", "expected_direction"], "")).upper()
    if action_raw in {"PAPER_BUY_OR_LONG", "UP", "BUY", "LONG"}:
        action = "BUY"
    elif action_raw in {"PAPER_SELL_OR_SHORT", "DOWN", "SELL", "SHORT"}:
        action = "SELL"
    else:
        raise ValueError(f"Cannot map trade action to order action: {action_raw!r}")

    ticker = first_present(row, ["ticker"], None)
    if not ticker:
        raise ValueError("Recommended trade row has no ticker; executor will not infer instruments from prose.")

    notional = float(first_present(row, ["recommended_notional_usd"], 0.0))
    risk = config.get("risk", {}) or {}
    max_order = float(risk.get("max_order_notional_usd", config.get("max_position_usd", 1000)))
    if notional <= 0:
        raise ValueError("recommended_notional_usd must be positive")
    if notional > max_order:
        raise ValueError(f"recommended_notional_usd {notional} exceeds max_order_notional_usd {max_order}")

    key = idempotency_key(row, run_id)
    now = utc_now()
    return {
        "intent_id": f"intent_{key[:20]}",
        "run_id": run_id,
        "created_time_utc": now,
        "source_card_id": str(first_present(row, ["card_id"], "")),
        "source_card_hash": str(first_present(row, ["card_hash"], "")),
        "market_name": str(first_present(row, ["market_name"], "")),
        "ticker": str(ticker),
        "asset_class": str(first_present(row, ["asset_class"], "other")),
        "action": action,
        "order_type": "MKT",
        "limit_price": None,
        "notional_usd": notional,
        "quantity": None,
        "time_in_force": "DAY",
        "paper_account_id": paper_account_id,
        "idempotency_key": key,
        "risk_approved": True,
        "source_recommended_trade": row,
    }


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")


def load_existing_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = obj.get("idempotency_key") or obj.get("intent", {}).get("idempotency_key")
            if key:
                keys.add(str(key))
    return keys


def submit_to_ibkr_paper(intent: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    # This intentionally supports only the IBKR paper path. It does not support live accounts.
    try:
        from ib_insync import IB, MarketOrder, Stock  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on runtime host
        raise RuntimeError("Install ib_insync on the paper host to submit paper orders") from exc

    ibkr = config.get("ibkr", {}) or {}
    execution = config.get("execution", {}) or {}
    host = str(ibkr.get("host", "127.0.0.1"))
    port = int(os.environ.get(str(ibkr.get("port_env_var", "IBKR_TWS_PORT")), "7497"))
    client_id = int(os.environ.get(str(ibkr.get("client_id_env_var", "IBKR_CLIENT_ID")), "77"))
    expected_account = os.environ.get(str(execution.get("expected_account_id_env_var", "IBKR_PAPER_ACCOUNT_ID")), "").strip()
    if not expected_account or not expected_account.upper().startswith("DU"):
        raise RuntimeError("Refusing to submit: expected IBKR paper account env var is missing or does not start with DU")

    ib = IB()
    ib.connect(host, port, clientId=client_id, timeout=10)
    try:
        accounts = set(ib.managedAccounts())
        if expected_account not in accounts:
            raise RuntimeError(f"Connected IBKR session does not expose expected paper account {expected_account}; got {sorted(accounts)}")
        contract = Stock(intent["ticker"], "SMART", "USD")
        ib.qualifyContracts(contract)
        # Quantity sizing is deliberately conservative. The first runtime implementation does not
        # infer live prices for notional sizing. Set quantity upstream once pricing is implemented.
        quantity = intent.get("quantity")
        if not quantity:
            raise RuntimeError("Order intent has no quantity. Add deterministic notional-to-quantity sizing before paper submission.")
        order = MarketOrder(intent["action"], float(quantity), account=expected_account, tif=intent.get("time_in_force", "DAY"))
        trade = ib.placeOrder(contract, order)
        ib.sleep(2)
        return {
            "receipt_id": f"receipt_{intent['idempotency_key'][:20]}",
            "intent_id": intent["intent_id"],
            "submitted_time_utc": utc_now(),
            "broker": "IBKR",
            "execution_mode": "ibkr_paper",
            "paper_account_id": expected_account,
            "ticker": intent["ticker"],
            "action": intent["action"],
            "order_type": intent["order_type"],
            "requested_notional_usd": intent["notional_usd"],
            "submitted_quantity": quantity,
            "ibkr_order_id": getattr(trade.order, "orderId", None),
            "status": str(getattr(trade.orderStatus, "status", "submitted")),
            "avg_fill_price": getattr(trade.orderStatus, "avgFillPrice", None),
            "filled_quantity": getattr(trade.orderStatus, "filled", None),
            "source_card_id": intent["source_card_id"],
            "source_card_hash": intent["source_card_hash"],
            "idempotency_key": intent["idempotency_key"],
        }
    finally:
        ib.disconnect()


def run_once(project_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    config_path = project_root / args.config
    config = load_config(config_path)
    execution = config.get("execution", {}) or {}
    kill_var = str(execution.get("kill_switch_env_var", "FALNAMA_KILL_SWITCH"))
    if os.environ.get(kill_var, "").strip().lower() in TRUTHY:
        raise RuntimeError(f"Kill switch is active: {kill_var}")

    mode = "live_research" if args.submit_paper_orders else "smoke"
    execution_mode = "ibkr_paper" if args.submit_paper_orders else "dry_run"
    subprocess.run(
        [sys.executable, "scripts/run_pipeline.py", "--config", args.config, "--mode", mode, "--execution", execution_mode],
        cwd=project_root,
        check=True,
    )

    config = load_config(config_path)
    trades_dir = repo_path(project_root, config, "recommended_trades", "repositories/recommended_trades")
    order_dir = repo_path(project_root, config, "order_intents", "repositories/order_intents")
    receipts_dir = repo_path(project_root, config, "execution_receipts", "repositories/execution_receipts")
    run_logs_dir = repo_path(project_root, config, "run_logs", "repositories/run_logs")

    run_time = utc_now()
    run_id = compact_timestamp(run_time)
    paper_account_id = os.environ.get(str(execution.get("expected_account_id_env_var", "IBKR_PAPER_ACCOUNT_ID")), "PAPER_ACCOUNT_NOT_SET")
    df, source = read_latest_recommendations(trades_dir)
    intents: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    intent_path = order_dir / f"order_intents_{run_id}.jsonl"
    all_intents_path = order_dir / "all_order_intents.jsonl"
    receipt_path = receipts_dir / f"execution_receipts_{run_id}.jsonl"
    existing_keys = load_existing_keys(all_intents_path)

    for row in df.to_dict(orient="records"):
        try:
            intent = build_order_intent(row, config=config, run_id=run_id, paper_account_id=paper_account_id)
            if intent["idempotency_key"] in existing_keys:
                raise ValueError("duplicate idempotency key already submitted or staged")
            intents.append(intent)
            existing_keys.add(intent["idempotency_key"])
        except Exception as exc:
            rejected.append({"row": row, "reason": str(exc), "time_utc": utc_now()})

    if intents:
        append_jsonl(intent_path, intents)
        append_jsonl(all_intents_path, intents)

    receipts: list[dict[str, Any]] = []
    if args.submit_paper_orders:
        for intent in intents:
            receipts.append(submit_to_ibkr_paper(intent, config))
        if receipts:
            append_jsonl(receipt_path, receipts)

    heartbeat = {
        "time_utc": utc_now(),
        "status": "ok",
        "execution_mode": execution_mode,
        "submit_paper_orders": args.submit_paper_orders,
        "recommendation_source": source,
        "recommended_rows": int(len(df)),
        "order_intents_created": len(intents),
        "order_intents_rejected": len(rejected),
        "paper_orders_submitted": len(receipts),
        "live_broker_orders_enabled": False,
    }
    (run_logs_dir / "paper_bot_heartbeat.json").write_text(json.dumps(heartbeat, indent=2) + "\n", encoding="utf-8")
    if rejected:
        append_jsonl(run_logs_dir / f"paper_bot_rejected_{run_id}.jsonl", rejected)
    print(json.dumps(heartbeat, indent=2))
    return heartbeat


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Falnama paper bot loop. Default mode is safe dry-run staging only.")
    parser.add_argument("--config", default="config/falnama_config.yaml")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=900)
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--submit-paper-orders", action="store_true", help="Actually submit approved orders to the IBKR paper account. Requires quantities in order intents.")
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    cycles = 0
    while True:
        run_once(project_root, args)
        cycles += 1
        if args.once or (args.max_cycles is not None and cycles >= args.max_cycles):
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
