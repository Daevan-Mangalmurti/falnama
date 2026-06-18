def generate_index_card(market_context, config, *, created_time_utc=None, generation_mode=None):
    if mock_mode:
        return generate_mock_card(...)

    response = call_scenario_analysis_gpt_api(
        endpoint=config["scenario_analysis_api_endpoint"],
        api_key=os.environ[config["scenario_analysis_api_key_env_var"]],
        timeout=config["scenario_analysis_timeout_seconds"],
        payload={
            "task": "write_index_card",
            "market_context": market_context,
            "schema_name": "Falnama Immutable Index Card",
            "minimum_expected_move_bps": config["minimum_expected_move_bps"],
            "allowed_asset_classes": [
                "equity",
                "etf",
                "equity_index",
                "fx_proxy",
                "commodity",
                "other",
            ],
        },
    )

    card = normalize_response_to_index_card(response)
    card["card_hash"] = canonical_hash(card)
    return card
    