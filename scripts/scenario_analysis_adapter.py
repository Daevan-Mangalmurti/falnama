from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any


ALLOWED_ASSET_CLASSES = {"equity", "etf", "equity_index", "fx_proxy", "commodity", "other"}
ALLOWED_DIRECTIONS = {"up", "down", "mixed", "unclear"}
TIME_PLAN_KEYS = ["30m", "1h", "2h", "6h", "12h"]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_slug(value: Any, fallback: str = "unknown") -> str:
    text = str(value if value not in (None, "") else fallback).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:100] or fallback


def mock_mode_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("mock_mode", config.get("scenario_analysis_mock_mode", True)))


def canonical_hash(card: dict[str, Any]) -> str:
    """Return the canonical SHA-256 hash for an immutable index card.

    The hash is computed over the JSON card with card_hash blanked. Keep this
    function identical in every validator that recomputes card hashes.
    """
    payload = json.loads(json.dumps(card, default=str))
    payload["card_hash"] = ""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _float_or_default(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def choose_mock_prediction(market_context: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Deterministic mock mapping for CI, smoke tests, and API-contract testing."""
    name = str(market_context.get("market_name", "")).lower()
    if "thai" in name or "thailand" in name:
        asset, ticker, instrument, asset_class = "Thailand equities", "THD", "iShares MSCI Thailand ETF", "etf"
    elif "oil" in name or "iran" in name or "middle east" in name:
        asset, ticker, instrument, asset_class = "Crude oil", "USO", "United States Oil Fund", "commodity"
    elif "china" in name or "taiwan" in name:
        asset, ticker, instrument, asset_class = "China equity risk proxy", "FXI", "iShares China Large-Cap ETF", "etf"
    elif "russia" in name or "ukraine" in name:
        asset, ticker, instrument, asset_class = "Europe equity risk proxy", "VGK", "Vanguard FTSE Europe ETF", "etf"
    else:
        asset, ticker, instrument, asset_class = "Emerging-market risk proxy", "EEM", "iShares MSCI Emerging Markets ETF", "etf"

    min_move = _float_or_default(config.get("minimum_expected_move_bps"), 700.0)
    return {
        "asset": asset,
        "ticker": ticker,
        "tradable_instrument_name": instrument,
        "asset_class": asset_class,
        "expected_direction": "down",
        "expected_return_12h_bps": -max(min_move + 150.0, 850.0),
        "confidence": 0.62,
        "confidence_interval_bps": [-1400.0, -250.0],
        "trade_eligibility": {
            "eligible": True,
            "reason": "Mock output meets configured minimum expected move for pipeline testing.",
            "minimum_expected_move_bps": min_move,
        },
        "reasoning": "MOCK: maps a geopolitical prediction-market signal to a broad liquid market proxy. Replace with Scenario Analysis GPT production reasoning before research use.",
        "time_plan": {
            "30m": "Check whether the anomaly persists and whether public news confirms it.",
            "1h": "Compare the proxy move with related regional risk assets.",
            "2h": "Reassess whether the thesis is stale or contradicted.",
            "6h": "Monitor liquidity, official statements, and correlated assets.",
            "12h": "Close the paper research window and archive the result.",
        },
    }


def _normalize_prediction(raw: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    min_move = _float_or_default(config.get("minimum_expected_move_bps"), 700.0)

    asset = str(raw.get("asset") or raw.get("underlying_asset") or "Unspecified asset").strip()
    ticker = raw.get("ticker")
    ticker = None if ticker in ("", "null", "None") else (str(ticker).strip().upper() if ticker is not None else None)
    instrument = raw.get("tradable_instrument_name") or raw.get("instrument") or raw.get("tradable_instrument")
    instrument = None if instrument in ("", "null", "None") else (str(instrument).strip() if instrument is not None else None)

    asset_class = str(raw.get("asset_class") or "other").strip().lower()
    if asset_class not in ALLOWED_ASSET_CLASSES:
        asset_class = "other"

    direction = str(raw.get("expected_direction") or raw.get("direction") or "unclear").strip().lower()
    if direction in {"long", "buy", "higher", "positive"}:
        direction = "up"
    elif direction in {"short", "sell", "lower", "negative"}:
        direction = "down"
    if direction not in ALLOWED_DIRECTIONS:
        direction = "unclear"

    expected = _float_or_default(raw.get("expected_return_12h_bps", raw.get("expected_return_bps")), 0.0)
    confidence = max(0.0, min(1.0, _float_or_default(raw.get("confidence"), 0.0)))

    ci = raw.get("confidence_interval_bps") or raw.get("expected_return_ci_bps") or [expected, expected]
    if not isinstance(ci, list) or len(ci) != 2:
        ci = [expected, expected]
    ci = [_float_or_default(ci[0], expected), _float_or_default(ci[1], expected)]

    eligibility = raw.get("trade_eligibility") or {}
    if not isinstance(eligibility, dict):
        eligibility = {}
    eligible = bool(eligibility.get("eligible", abs(expected) >= min_move and direction in {"up", "down"}))
    reason = str(eligibility.get("reason") or ("Meets minimum expected move threshold." if eligible else "Does not meet minimum expected move or direction threshold."))

    time_plan = raw.get("time_plan") if isinstance(raw.get("time_plan"), dict) else {}
    normalized_time_plan = {
        key: str(time_plan.get(key) or f"Reassess the scenario at T+{key} and record whether the thesis remains valid.")
        for key in TIME_PLAN_KEYS
    }

    return {
        "asset": asset or "Unspecified asset",
        "ticker": ticker,
        "tradable_instrument_name": instrument,
        "asset_class": asset_class,
        "expected_direction": direction,
        "expected_return_12h_bps": expected,
        "confidence": confidence,
        "confidence_interval_bps": ci,
        "trade_eligibility": {
            "eligible": eligible,
            "reason": reason,
            "minimum_expected_move_bps": min_move,
        },
        "reasoning": str(raw.get("reasoning") or raw.get("rationale") or "No reasoning supplied by Scenario Analysis GPT."),
        "time_plan": normalized_time_plan,
    }


def _call_scenario_analysis_api(market_context: dict[str, Any], config: dict[str, Any], *, generation_mode: str | None = None) -> dict[str, Any]:
    """Generic HTTP boundary for the future Scenario Analysis GPT service.

    Expected response can be either:
      1. a full index-card-shaped object,
      2. {"card": <index-card-shaped object>}, or
      3. {"predictions": [...], "evidence_used": [...], ...}.
    """
    endpoint = config.get("scenario_analysis_api_endpoint")
    if not endpoint or endpoint == "PLACEHOLDER":
        raise NotImplementedError("Scenario Analysis GPT endpoint is not configured.")

    key_env_var = config.get("scenario_analysis_api_key_env_var", "SCENARIO_ANALYSIS_GPT_API_KEY")
    api_key = os.environ.get(str(key_env_var))
    if not api_key:
        raise RuntimeError(f"Missing Scenario Analysis GPT API key in ${key_env_var}.")

    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required for production Scenario Analysis GPT calls.") from exc

    payload = {
        "task": "write_index_card",
        "generation_mode": generation_mode,
        "market_context": market_context,
        "schema_name": "Falnama Immutable Index Card",
        "minimum_expected_move_bps": config.get("minimum_expected_move_bps", 700),
        "prediction_window": "12h",
        "allowed_asset_classes": sorted(ALLOWED_ASSET_CLASSES),
        "required_time_plan_keys": TIME_PLAN_KEYS,
        "guardrails": {
            "paper_trading_only": True,
            "no_broker_execution": True,
            "do_not_revise_existing_cards": True,
            "avoid_ex_post_rationalization": True,
        },
    }
    timeout = int(config.get("scenario_analysis_timeout_seconds", 60))
    response = requests.post(
        str(endpoint),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Scenario Analysis GPT returned non-object JSON.")
    return data


def _build_index_card(
    market_context: dict[str, Any],
    config: dict[str, Any],
    *,
    predictions: list[dict[str, Any]],
    created_time_utc: str,
    mock_mode: bool,
    api_response: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = {
        "market_id": None if market_context.get("market_id") in ("", "nan") else market_context.get("market_id"),
        "market_slug": None if market_context.get("market_slug") in ("", "nan") else market_context.get("market_slug"),
        "market_name": str(market_context.get("market_name") or "Unknown market"),
        "source_market_file": market_context.get("source_market_file"),
        "market_url": market_context.get("market_url"),
    }
    natural_key = source.get("market_id") or source.get("market_slug") or safe_slug(source.get("market_name"), "market")
    response = api_response or {}

    evidence = response.get("evidence_used")
    if not isinstance(evidence, list) or not evidence:
        evidence = [
            "prediction market question and metadata",
            "market-to-asset sensitivity mapping",
            "historical event-study analogues",
            "public information to be checked before execution",
        ]

    uncertainty = response.get("uncertainty_notes")
    if not isinstance(uncertainty, list) or not uncertainty:
        uncertainty = [
            "MOCK MODE output. Do not treat as an investment recommendation."
            if mock_mode
            else "Scenario Analysis GPT output requires downstream validation and public-information checks."
        ]

    justification = response.get("llm_justification") if isinstance(response.get("llm_justification"), dict) else {}
    card = {
        "card_id": f"{safe_slug(natural_key, 'market')}_{created_time_utc}",
        "card_version": "v1-mock" if mock_mode else "v1",
        "card_hash": "",
        "do_not_revise": True,
        "created_time_utc": created_time_utc,
        "mock_mode": mock_mode,
        "source": source,
        "market_name": source["market_name"],
        "realized_outcome": response.get("realized_outcome", "Unknown"),
        "conclusion_time_utc": response.get("conclusion_time_utc"),
        "prediction_window": str(response.get("prediction_window") or "12h"),
        "predictions": predictions,
        "evidence_used": [str(x) for x in evidence],
        "uncertainty_notes": [str(x) for x in uncertainty],
        "llm_justification": {
            "summary": str(justification.get("summary") or ("Mock Scenario Analysis GPT card generated for pipeline testing." if mock_mode else "Scenario Analysis GPT generated a pre-trade market-to-asset scenario card.")),
            "key_assumptions": [str(x) for x in justification.get("key_assumptions", ["The prediction-market signal contains information not fully reflected in public assets.", "The selected instrument is a liquid enough proxy for the event risk."])],
            "failure_modes": [str(x) for x in justification.get("failure_modes", ["The prediction-market move reverses.", "Public information already explains the move.", "The selected proxy asset is weakly exposed to the event."])],
            "why_not_ex_post": str(justification.get("why_not_ex_post") or "The card is timestamped, content-hashed, and never overwritten. Execution must reject cards created after a live trigger time."),
        },
    }
    card["card_hash"] = canonical_hash(card)
    return card


def _coerce_api_response_to_card(
    api_response: dict[str, Any],
    market_context: dict[str, Any],
    config: dict[str, Any],
    *,
    created_time_utc: str,
) -> dict[str, Any]:
    candidate = api_response.get("card") if isinstance(api_response.get("card"), dict) else api_response

    # If the API already returned a full index card, preserve it but enforce local immutability/hash fields.
    if isinstance(candidate, dict) and "predictions" in candidate and "source" in candidate and "market_name" in candidate:
        card = dict(candidate)
        card.setdefault("do_not_revise", True)
        card.setdefault("created_time_utc", created_time_utc)
        card.setdefault("mock_mode", False)
        card.setdefault("card_version", "v1")
        card.setdefault("realized_outcome", "Unknown")
        card.setdefault("conclusion_time_utc", None)
        card.setdefault("prediction_window", "12h")
        card.setdefault("evidence_used", ["Scenario Analysis GPT response"])
        card.setdefault("uncertainty_notes", ["Scenario Analysis GPT output requires downstream audit."])
        card.setdefault("llm_justification", {
            "summary": "Scenario Analysis GPT generated card.",
            "key_assumptions": [],
            "failure_modes": [],
            "why_not_ex_post": "The card is timestamped, content-hashed, and never overwritten.",
        })
        card["predictions"] = [_normalize_prediction(p, config) for p in card.get("predictions", [])]
        if not card["predictions"]:
            raise ValueError("Scenario Analysis GPT returned a card with no predictions.")
        card["card_hash"] = ""
        card["card_hash"] = canonical_hash(card)
        return card

    predictions_raw = api_response.get("predictions")
    if not isinstance(predictions_raw, list) or not predictions_raw:
        raise ValueError("Scenario Analysis GPT response must contain a nonempty predictions array or a full card.")
    predictions = [_normalize_prediction(p, config) for p in predictions_raw]
    return _build_index_card(
        market_context,
        config,
        predictions=predictions,
        created_time_utc=created_time_utc,
        mock_mode=False,
        api_response=api_response,
    )


def generate_index_card(
    market_context: dict[str, Any],
    config: dict[str, Any],
    *,
    created_time_utc: str | None = None,
    generation_mode: str | None = None,
) -> dict[str, Any]:
    """Generate one immutable Scenario Analysis index-card payload.

    This is the single boundary between Falnama and Scenario Analysis GPT.
    It intentionally returns analysis artifacts only. It never creates orders,
    never touches IBKR, and never mutates existing cards.
    """
    created_time = created_time_utc or utc_now()

    endpoint = config.get("scenario_analysis_api_endpoint")
    use_mock = mock_mode_enabled(config) or endpoint in (None, "", "PLACEHOLDER")
    if use_mock:
        prediction = choose_mock_prediction(market_context, config)
        return _build_index_card(
            market_context,
            config,
            predictions=[_normalize_prediction(prediction, config)],
            created_time_utc=created_time,
            mock_mode=True,
            api_response={"mode": "mock"},
        )

    api_response = _call_scenario_analysis_api(market_context, config, generation_mode=generation_mode)
    return _coerce_api_response_to_card(api_response, market_context, config, created_time_utc=created_time)
