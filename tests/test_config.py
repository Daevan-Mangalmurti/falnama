"""Config loads and validates; bad configs fail loudly."""

from __future__ import annotations

import pytest

from falnama.config import ConfigError, load_config


def test_default_config_loads():
    settings = load_config()  # finds config/falnama.yaml automatically
    assert settings.paper_trading_only is True
    assert settings.card_mode in {"mock", "live"}
    assert settings.output_dir("index_cards").exists()


def test_bad_config_is_rejected(tmp_path):
    cfg = tmp_path / "config" / "falnama.yaml"
    cfg.parent.mkdir()
    cfg.write_text("paper_trading_only: false\ncard_mode: bogus\n")
    with pytest.raises(ConfigError) as exc:
        load_config(cfg)
    # The message should name BOTH problems, not just the first.
    assert "paper_trading_only" in str(exc.value)
    assert "card_mode" in str(exc.value)
