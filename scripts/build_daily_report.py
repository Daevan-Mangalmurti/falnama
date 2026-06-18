from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_config(path: Path) -> dict[str, Any]:
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict):
            return data
    return {}


def repo_path(project_root: Path, config: dict[str, Any], key: str, default: str) -> Path:
    return project_root / config.get("repositories", {}).get(key, default)


def latest_file(directory: Path, pattern: str) -> Path | None:
    if not directory.exists():
        return None
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def count_xlsx_rows(path: Path | None, sheet_name: str) -> int:
    if path is None or not path.exists():
        return 0
    try:
        return len(pd.read_excel(path, sheet_name=sheet_name))
    except Exception:
        return 0


def count_csv_rows(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    try:
        return len(pd.read_csv(path))
    except Exception:
        return 0


def read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def count_jsonl(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def latest_reconciliation_status(reconciliation_dir: Path) -> tuple[str, str | None]:
    path = latest_file(reconciliation_dir, "reconciliation_*.json")
    data = read_json(path)
    if not data:
        return "missing", None
    return str(data.get("status", "unknown")), str(path)


def build_report(project_root: Path, config: dict[str, Any]) -> str:
    relevant_dir = repo_path(project_root, config, "relevant_markets", "repositories/relevant_markets")
    anomalies_dir = repo_path(project_root, config, "anomalies", "repositories/anomalies")
    cards_dir = repo_path(project_root, config, "index_cards", "repositories/index_cards")
    trades_dir = repo_path(project_root, config, "recommended_trades", "repositories/recommended_trades")
    rejected_dir = repo_path(project_root, config, "rejected_signals", "repositories/rejected_signals")
    order_dir = repo_path(project_root, config, "order_intents", "repositories/order_intents")
    receipts_dir = repo_path(project_root, config, "execution_receipts", "repositories/execution_receipts")
    logs_dir = repo_path(project_root, config, "run_logs", "repositories/run_logs")
    reconciliation_dir = repo_path(project_root, config, "broker_reconciliation", "repositories/broker_reconciliation")

    latest_trades = latest_file(trades_dir, "paper_recommended_trades_*.xlsx")
    latest_rejected = latest_file(rejected_dir, "rejected_signals_*.csv")
    latest_ranked = latest_file(anomalies_dir, "ranked_anomalies_*.csv") or latest_file(relevant_dir, "ranked_anomalies_*.csv")
    latest_strong = latest_file(anomalies_dir, "strong_anomalies_*.csv") or latest_file(relevant_dir, "strong_anomalies_*.csv")
    latest_intents = latest_file(order_dir, "order_intents_*.jsonl")
    latest_receipts = latest_file(receipts_dir, "execution_receipts_*.jsonl")
    heartbeat = read_json(logs_dir / "paper_bot_heartbeat.json")
    recon_status, recon_path = latest_reconciliation_status(reconciliation_dir)

    card_count = len(list(cards_dir.glob("*.json"))) if cards_dir.exists() else 0
    ranked_count = count_csv_rows(latest_ranked)
    strong_count = count_csv_rows(latest_strong)
    recommended_count = count_xlsx_rows(latest_trades, "Recommended Trades")
    rejected_count = count_csv_rows(latest_rejected)
    intent_count = count_jsonl(latest_intents)
    receipt_count = count_jsonl(latest_receipts)

    rejection_reasons: Counter[str] = Counter()
    if latest_rejected and latest_rejected.exists():
        try:
            df = pd.read_csv(latest_rejected)
            for col in ["rejection_reason", "reason", "status_reason"]:
                if col in df.columns:
                    rejection_reasons.update(df[col].dropna().astype(str).tolist())
                    break
        except Exception:
            pass

    lines = [
        f"# Falnama Daily Paper Report",
        "",
        f"Generated UTC: `{utc_now()}`",
        "",
        "## Pipeline summary",
        "",
        f"- Ranked anomalies: **{ranked_count}**",
        f"- Strong anomalies: **{strong_count}**",
        f"- Immutable index cards available: **{card_count}**",
        f"- Recommended paper trades: **{recommended_count}**",
        f"- Rejected signals: **{rejected_count}**",
        f"- Order intents staged: **{intent_count}**",
        f"- Execution receipts: **{receipt_count}**",
        f"- Latest reconciliation status: **{recon_status}**",
        "",
        "## Latest artifacts",
        "",
        f"- Trades workbook: `{latest_trades or 'missing'}`",
        f"- Rejected signals: `{latest_rejected or 'missing'}`",
        f"- Ranked anomalies: `{latest_ranked or 'missing'}`",
        f"- Strong anomalies: `{latest_strong or 'missing'}`",
        f"- Order intents: `{latest_intents or 'missing'}`",
        f"- Execution receipts: `{latest_receipts or 'missing'}`",
        f"- Reconciliation: `{recon_path or 'missing'}`",
        "",
        "## Paper bot heartbeat",
        "",
    ]
    if heartbeat:
        lines.append("```json")
        lines.append(json.dumps(heartbeat, indent=2))
        lines.append("```")
    else:
        lines.append("No heartbeat found.")

    lines.extend(["", "## Top rejection reasons", ""])
    if rejection_reasons:
        for reason, count in rejection_reasons.most_common(10):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("No rejection-reason summary available.")

    lines.extend([
        "",
        "## Guardrail note",
        "",
        "This report is for the IBKR paper environment. It does not indicate live broker execution.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Markdown daily report for Falnama paper trading.")
    parser.add_argument("--config", default="config/falnama_config.yaml")
    parser.add_argument("--source", default="repositories/", help="Kept for workflow compatibility; repository paths come from config.")
    parser.add_argument("--output", default="reports/daily/latest.md")
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    config = load_config(project_root / args.config)
    report = build_report(project_root, config)
    output = project_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report + "\n", encoding="utf-8")
    print(str(output))


if __name__ == "__main__":
    main()
