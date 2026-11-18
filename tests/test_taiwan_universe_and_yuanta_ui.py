from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from crypto_trading.feature_engineer import CryptoFeatureEngineer
from taiwan_stock.universe_registry import (
    TaiwanUniverseRegistry,
    build_yuanta_style_quote_panel,
    get_taiwan_tick_size,
)


def test_taiwan_universe_registry_seeds_and_resolves_listed_and_otc(tmp_path: Path) -> None:
    reg = TaiwanUniverseRegistry(db_path=tmp_path / "tw.db")
    all_items = reg.search(limit=200)
    assert len(all_items) >= 45
    assert reg.resolve_symbol("2330") == "2330.TW"
    assert reg.resolve_symbol("台達電") == "2308.TW"
    assert reg.resolve_symbol("6488") == "6488.TWO"
    assert reg.resolve_symbol("00981A") == "00981A.TW"
    assert reg.resolve_symbol("00888") == "00888.TWO"

    cpo = reg.search(industry="光通訊CPO")
    assert any(x["code"] == "6442" for x in cpo)


def test_yuanta_quote_panel_computes_5level_orderbook_and_poc() -> None:
    idx = pd.date_range("2026-06-01", periods=80, freq="1D", tz="UTC")
    close = np.linspace(900.0, 1050.0, len(idx))
    df = pd.DataFrame(
        {
            "Open": close - 5.0,
            "High": close + 10.0,
            "Low": close - 8.0,
            "Close": close,
            "Volume": np.full(len(idx), 25_000_000.0),
        },
        index=idx,
    )
    panel = build_yuanta_style_quote_panel(df, "2330.TW", meta={"name": "台積電", "industry": "半導體"})
    assert panel["available"] is True
    assert len(panel["order_book"]["bids"]) == 5
    assert len(panel["order_book"]["asks"]) == 5
    assert get_taiwan_tick_size(1050.0) == 5.0
    assert panel["volume_profile"]["poc_price"] > 0
    assert "supertrend_bull" in panel["indicators"]


def test_crypto_feature_engineer_includes_advanced_youtube_indicators() -> None:
    idx = pd.date_range("2026-01-01", periods=120, freq="1h", tz="UTC")
    rng = np.random.default_rng(123)
    c = 90000.0 + np.cumsum(rng.normal(0, 150, size=len(idx)))
    df = pd.DataFrame(
        {
            "Open": c - 20.0,
            "High": c + 80.0,
            "Low": c - 70.0,
            "Close": c,
            "Volume": rng.uniform(100, 1000, size=len(idx)),
        },
        index=idx,
    )
    fe = CryptoFeatureEngineer()
    out = fe.compute_features(df, timeframe="1h")
    for col in ("adx_14_norm", "di_spread_14", "supertrend_dist", "squeeze_ratio_20", "ichimoku_tk_spread", "stoch_rsi_14", "cmf_20"):
        assert col in out.columns, f"Missing indicator column: {col}"


def test_brain_alpha_crypto_strategy_translates_vault_alphas_into_crypto_signals(tmp_path: Path) -> None:
    from crypto_trading.strategies.brain_alpha_strategy import BrainAlphaCryptoStrategy, evaluate_fastexpr_on_ohlcv
    from worldquant_brain.alpha_vault import AlphaVault

    vault = AlphaVault(db_path=str(tmp_path / "b.db"))
    vault.save_alpha(
        "BRAIN_Test1",
        "group_neutralize(ts_decay_linear(rank(close - open), 5), subindustry)",
        sharpe=1.85,
        fitness=1.45,
        turnover=0.20,
        status_override="BRAIN_COMPLETE",
        brain_alpha_id="Test1",
        force=True,
    )
    idx = pd.date_range("2026-01-01", periods=160, freq="1h", tz="UTC")
    c = np.linspace(90000.0, 95000.0, len(idx)) + np.sin(np.arange(len(idx))) * 300.0
    df = pd.DataFrame(
        {"Open": c - 50.0, "High": c + 120.0, "Low": c - 100.0, "Close": c, "Volume": np.linspace(500.0, 1500.0, len(idx))},
        index=idx,
    )
    ser = evaluate_fastexpr_on_ohlcv("group_neutralize(ts_decay_linear(rank(close - open), 5), subindustry)", df)
    assert len(ser) == len(df)
    assert ser.abs().max() <= 1.0

    strat = BrainAlphaCryptoStrategy(vault_db_path=str(tmp_path / "b.db"))
    out = strat.generate_signals(df)
    assert "brain_alpha_score" in out.columns
    assert "signal" in out.columns
    assert set(np.unique(out["signal"])).issubset({-1.0, 0.0, 1.0})
