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
    BUILTIN_TAIWAN_UNIVERSE,
    MARKET_INDICES,
    TaiwanUniverseRegistry,
    build_yuanta_style_quote_panel,
    get_taiwan_tick_size,
)


def test_taiwan_universe_registry_seeds_and_resolves_listed_and_otc(tmp_path: Path) -> None:
    assert len(BUILTIN_TAIWAN_UNIVERSE) >= 200
    assert len(MARKET_INDICES) >= 8
    assert any(ix["symbol"] == "^TWII" for ix in MARKET_INDICES)
    assert any(ix["symbol"] == "^SOX" for ix in MARKET_INDICES)

    reg = TaiwanUniverseRegistry(db_path=tmp_path / "tw.db")
    all_items = reg.search(limit=300)
    assert len(all_items) >= 200
    assert all(x.get("market") != "INDEX" for x in all_items)
    # Popularity rank ascending order when volume is 0
    ranks = [x["popularity_rank"] for x in all_items[:20]]
    assert ranks == sorted(ranks)

    # Diverse ETF categories present
    etfs = reg.search(industry="ETF", limit=100)
    assert len(etfs) >= 30
    etf_cats = {x["industry"] for x in etfs}
    assert {"ETF-高股息", "ETF-市值型", "ETF-債券", "ETF-科技半導體"}.issubset(etf_cats)

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


def test_yuanta_quote_panel_evaluates_pe_eps_buy_suitability() -> None:
    idx = pd.date_range("2026-06-01", periods=80, freq="1D", tz="UTC")
    close = np.linspace(100.0, 125.0, len(idx))
    df = pd.DataFrame(
        {
            "Open": close - 1.0,
            "High": close + 2.0,
            "Low": close - 1.5,
            "Close": close,
            "Volume": np.full(len(idx), 12_000_000.0),
        },
        index=idx,
    )
    attractive_info = {
        "trailingPE": 14.5,
        "forwardPE": 12.2,
        "trailingEps": 8.5,
        "forwardEps": 10.2,
        "priceToBook": 1.4,
        "dividendYield": 0.052,
        "returnOnEquity": 0.18,
        "revenueGrowth": 0.22,
    }
    panel_buy = build_yuanta_style_quote_panel(df, "2330.TW", meta={"name": "台積電", "industry": "半導體"}, fundamentals=attractive_info)
    assert panel_buy["fundamentals"]["trailing_pe"] == 14.5
    assert panel_buy["fundamentals"]["trailing_eps"] == 8.5
    assert panel_buy["fundamentals"]["dividend_yield_pct"] == 5.2
    assert panel_buy["suitability"]["verdict_code"] in ("BUY", "ACCUMULATE")
    assert panel_buy["suitability"]["overall_score"] >= 70
    assert len(panel_buy["suitability"]["reasons"]) >= 2

    expensive_info = {
        "trailingPE": 65.0,
        "trailingEps": -1.2,
        "priceToBook": 8.5,
        "returnOnEquity": -0.05,
        "revenueGrowth": -0.25,
    }
    panel_caution = build_yuanta_style_quote_panel(df.iloc[::-1].copy(), "9999.TW", meta={"name": "測試高估股", "industry": "其他"}, fundamentals=expensive_info)
    assert panel_caution["suitability"]["fundamental_score"] < 40
    assert len(panel_caution["suitability"]["risk_notes"]) >= 2


def test_official_twse_tpex_cached_valuation_fallback_in_quote_panel() -> None:
    idx = pd.date_range("2026-06-01", periods=80, freq="1D", tz="UTC")
    close = np.linspace(500.0, 600.0, len(idx))
    df = pd.DataFrame(
        {"Open": close - 2.0, "High": close + 4.0, "Low": close - 3.0, "Close": close, "Volume": np.full(len(idx), 8_000_000.0)},
        index=idx,
    )
    meta = {
        "name": "聯發科",
        "industry": "IC設計",
        "pe_ratio": 15.0,
        "dividend_yield": 4.8,
        "pb_ratio": 2.4,
        "revenue_yoy": 18.5,
    }
    # Even when yfinance .info is empty {}, official TWSE/TPEx cached valuation populates PE, EPS, Yield, PB, ROE, YoY!
    panel = build_yuanta_style_quote_panel(df, "2454.TW", meta=meta, fundamentals={})
    f = panel["fundamentals"]
    assert f["trailing_pe"] == 15.0
    assert f["trailing_eps"] == 40.0  # 600.0 / 15.0
    assert f["dividend_yield_pct"] == 4.8
    assert f["price_to_book"] == 2.4
    assert f["roe_pct"] == 16.0
    assert f["revenue_growth_pct"] == 18.5
    assert panel["suitability"]["verdict_code"] in ("BUY", "ACCUMULATE")


def test_perp_carry_basis_strategy_generates_bounded_low_turnover_signals() -> None:
    from crypto_trading.strategies.perp_carry_basis_strategy import PerpCarryBasisStrategy

    idx = pd.date_range("2026-01-01", periods=180, freq="1h", tz="UTC")
    c = 90000.0 + np.sin(np.linspace(0, 12, len(idx))) * 1500.0
    df = pd.DataFrame(
        {"Open": c - 40.0, "High": c + 120.0, "Low": c - 110.0, "Close": c, "Volume": np.full(len(idx), 800.0)},
        index=idx,
    )
    strat = PerpCarryBasisStrategy()
    out = strat.generate_signals(df)
    assert "signal" in out.columns
    assert "funding_bps_8h" in out.columns
    assert "basis_bps" in out.columns
    assert set(np.unique(out["signal"])).issubset({-1.0, 0.0, 1.0})


def test_popular_watchlist_is_not_full_universe_and_security_types_separated(tmp_path: Path) -> None:
    from datetime import date
    from taiwan_stock.universe_registry import (
        DEFAULT_POPULAR_WATCHLIST,
        POPULAR_WATCHLIST_IS_FULL_UNIVERSE,
        Security,
        parse_twse_html,
        to_yahoo_symbol,
    )

    assert len(DEFAULT_POPULAR_WATCHLIST) >= 200
    assert POPULAR_WATCHLIST_IS_FULL_UNIVERSE is False
    assert to_yahoo_symbol("2330", "TWSE") == "2330.TW"
    assert to_yahoo_symbol("6890", "TIB") == "6890.TW"
    assert to_yahoo_symbol("6488", "TPEX") == "6488.TWO"

    # Parse TWSE HTML distinguishing COMMON_STOCK, TIB, ETF, WARRANT, TDR
    html = (
        "<table>"
        "<tr><td>2330　台積電</td><td>ISIN</td><td>1994/09/05</td><td>上市</td><td>半導體業</td><td>ESVUFR</td></tr>"
        "<tr><td>6890　來億-KY</td><td>ISIN</td><td>2024/06/12</td><td>上市創新板</td><td>運動休閒</td><td>ESVUFR</td></tr>"
        "<tr><td>0050　元大台灣50</td><td>ISIN</td><td>2003/06/30</td><td>上市</td><td>ETF</td><td>CEOGBU</td></tr>"
        "<tr><td>030001　台積電權證</td><td>ISIN</td><td>2026/01/01</td><td>上市</td><td>權證</td><td>RWXXXX</td></tr>"
        "</table>"
    ).encode("big5")
    parsed = parse_twse_html(html, market="TWSE")
    by_sym = {s.symbol: s for s in parsed}
    assert by_sym["2330"].security_type == "COMMON_STOCK" and by_sym["2330"].yahoo_symbol == "2330.TW"
    assert by_sym["6890"].market == "TIB" and by_sym["6890"].board == "TIB"
    assert by_sym["0050"].security_type == "ETF"
    assert by_sym["030001"].security_type == "WARRANT"

    # Point-in-time Day 1 vs Day 2 delisting / survivorship handling (Gate U6 & U7)
    reg = TaiwanUniverseRegistry(db_path=tmp_path / "pit.db")
    day1 = [
        Security("1101", "台泥", "TWSE", "COMMON_STOCK"),
        Security("2330", "台積電", "TWSE", "COMMON_STOCK"),
        Security("6488", "環球晶", "TPEX", "COMMON_STOCK"),
        Security("0050", "元大台灣50", "TWSE", "ETF", "ETF"),
    ]
    day2 = [
        Security("2330", "台積電", "TWSE", "COMMON_STOCK"),
        Security("6488", "環球晶", "TPEX", "COMMON_STOCK"),
        Security("9999", "新掛牌", "TWSE", "COMMON_STOCK"),
    ]
    m1 = reg.write_snapshot(day1, as_of=date(2026, 11, 1))
    m2 = reg.write_snapshot(day2, as_of=date(2026, 11, 2))
    assert m1["active_common_equities"] == 3 and m1["counts"]["ETF"] == 1 and m1["quality_passed"] is True
    assert m2["active_common_equities"] == 3

    hist1 = {r["symbol"] for r in reg.historical_equities("2026-11-01")}
    hist2 = {r["symbol"] for r in reg.historical_equities("2026-11-02")}
    assert "1101" in hist1 and "9999" not in hist1
    assert "1101" not in hist2 and "9999" in hist2

    # Popular watchlist lineage explicitly reports is_full_universe=False
    lin = reg.get_universe_with_lineage(mode="POPULAR_WATCHLIST")
    assert lin["universe_mode"] == "POPULAR_WATCHLIST"
    assert lin["is_full_universe"] is False
    assert len(lin["universe_sha256"]) == 64

    # Common equities filter excludes ETFs (like 00632R)
    eq_only = reg.current_equities(include_etf=False)
    assert all(x["security_type"] == "COMMON_STOCK" for x in eq_only)
    assert all(not str(x["code"]).startswith("00") for x in eq_only)
