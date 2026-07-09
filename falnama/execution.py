"""Optional paper-SIMULATION seam (there is no real broker, anywhere).

WHAT:     When execution.mode is 'paper_sim', records each paper recommendation
          as a SIMULATED order, so the end-to-end execution contract can be
          exercised and evaluated. When mode is 'none' (default), does nothing.
CONSUMES: recommendations (Stage 4) + settings.execution
PRODUCES: simulated order records under outputs/recommended_trades/
REVIEWER: anyone studying how recommendations would translate into positions
ROLE:     a clearly-bounded seam. It models execution WITHOUT any broker, money,
          credentials, or network. There is no order routing here and there
          never will be in this repository — the point is that the one place a
          future, separately-reviewed execution layer could attach is explicit.

Honesty note: with no live asset-price feed, this does not invent fill prices.
It records the intended paper order (asset, action, notional) marked 'simulated'
— enough to exercise the contract and evaluate sizing, nothing more.
"""

from __future__ import annotations

from . import io
from .config import Settings
from .io import RunContext


def simulate_fills(recommendations: list[dict], settings: Settings) -> list[dict]:
    """Turn recommendations into simulated order records. Pure and testable."""
    orders = []
    for rec in recommendations:
        if rec.get("action") not in {"buy", "sell"}:
            continue
        orders.append({
            "run_id": rec.get("run_id"),
            "market_name": rec.get("market_name"),
            "card_id": rec.get("card_id"),
            "asset": rec.get("asset"),
            "action": rec.get("action"),
            "notional_usd": rec.get("notional_usd"),
            "status": "simulated",
            "is_paper": True,
            "note": "Simulated paper order — no broker, no fill price modeled.",
        })
    return orders


def run(ctx: RunContext, recommendations: list[dict]) -> list[dict]:
    """No-op unless execution.mode == 'paper_sim'. Otherwise writes simulated
    orders and records the batch in the manifest."""
    if ctx.settings.execution_mode != "paper_sim":
        return []
    orders = simulate_fills(recommendations, ctx.settings)
    io.write_json(ctx.settings.output_dir("recommended_trades") / f"simulated_orders_{ctx.run_id}.json", orders)
    io.update_manifest(ctx, "execution", {"mode": "paper_sim", "simulated_orders": len(orders)})
    return orders
