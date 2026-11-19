"""Diverse multi-domain WorldQuant BRAIN FASTEXPR template families to avoid 7-skeleton saturation."""
from __future__ import annotations

import random
from typing import Callable


def _grp() -> str:
    return random.choice(["subindustry", "industry", "sector"])


def _w_short() -> int:
    return random.choice([5, 10, 20])


def _w_med() -> int:
    return random.choice([20, 40, 60, 120])


def _w_long() -> int:
    return random.choice([60, 120, 240])


def _hump() -> float:
    return random.choice([0.008, 0.01, 0.015])


def _wrap_hygiene(core: str, grp: str | None = None, h: float | None = None) -> str:
    g = grp or _grp()
    hv = h if h is not None else _hump()
    return f"hump(group_neutralize(winsorize(rank({core}), std=4), {g}), hump={hv})"


_TEMPLATE_FAMILIES: dict[str, list[Callable[[], tuple[str, str]]]] = {
    "FUNDAMENTAL_CF_EV": [
        lambda: (
            "Fund_CF_EV",
            _wrap_hygiene(f"{random.choice(['operating_cashflow', 'operating_income', 'ebitda'])} / (cap + debt + 0.001)"),
        ),
        lambda: (
            "Fund_FCF_Delta",
            _wrap_hygiene(f"ts_delta(operating_cashflow / (assets + 0.001), {_w_med()})"),
        ),
        lambda: (
            "Fund_Sales_Margin",
            _wrap_hygiene(f"ts_zscore(operating_income / (sales + 0.001), {_w_long()})"),
        ),
        lambda: (
            "Fund_Accrual_Quality",
            _wrap_hygiene("(operating_cashflow - operating_income) / (assets + 0.001)"),
        ),
    ],
    "FUNDAMENTAL_ROA_DELEVERAGE": [
        lambda: (
            "Fund_ROA_Delev",
            _wrap_hygiene(f"rank(ebitda / (assets + 0.001)) - rank(ts_delta(debt / (assets + 0.001), {_w_med()}))"),
        ),
        lambda: (
            "Fund_AssetTurn_Z",
            _wrap_hygiene(f"ts_zscore(sales / (assets + 0.001), {_w_long()})"),
        ),
        lambda: (
            "Fund_Equity_Lev",
            _wrap_hygiene(f"-1 * ts_delta(liabilities / (equity + 0.001), {_w_med()})"),
        ),
        lambda: (
            "Fund_ROIC_Trend",
            _wrap_hygiene(f"ts_ir(operating_income / (assets - liabilities + 0.001), {_w_med()})"),
        ),
    ],
    "ANALYST_EPS_DISPERSION": [
        lambda: (
            "Analyst_EPS_Target",
            _wrap_hygiene(f"rank(ts_zscore(eps_consensus_mean, {_w_med()})) - 0.5 * rank(target_price / (close + 0.001) - 1)"),
        ),
        lambda: (
            "Analyst_EPS_Orthog",
            _wrap_hygiene(f"ts_regression(rank(ts_delta(eps_consensus_mean, {_w_short()})), rank(ts_delta(close, {_w_short()})), {_w_med()}, rettype=0)"),
        ),
        lambda: (
            "Analyst_Target_Gap",
            _wrap_hygiene(f"ts_zscore((target_price - close) / (close + 0.001), {_w_med()})"),
        ),
        lambda: (
            "Analyst_EPS_Accel",
            _wrap_hygiene(f"ts_delta(eps_consensus_mean, {_w_short()}) - ts_delay(ts_delta(eps_consensus_mean, {_w_short()}), {_w_short()})"),
        ),
    ],
    "SENTIMENT_BUZZ_MOMENTUM": [
        lambda: (
            "Sent_Buzz_Mom",
            _wrap_hygiene(f"rank(ts_zscore(scl12_buzz, {_w_med()})) * rank(ts_delta(close, {_w_short()}) / (close + 0.001))"),
        ),
        lambda: (
            "Sent_Buzz_Stab",
            _wrap_hygiene(f"-1 * ts_std_dev(ts_backfill(scl12_buzz, 5), {_w_short()})"),
        ),
        lambda: (
            "Sent_Buzz_Resid",
            _wrap_hygiene(f"ts_regression(ts_backfill(scl12_buzz, 5), returns, {_w_med()}, rettype=0)"),
        ),
    ],
    "OPTIONS_IV_SLOPE": [
        lambda: (
            "Opt_IV_Slope",
            _wrap_hygiene(f"rank(ts_zscore(opt6_slopeavg1m, {_w_med()})) * rank(opt6_fcstr2imp)"),
        ),
        lambda: (
            "Opt_IV_MeanRev",
            _wrap_hygiene(f"-1 * ts_zscore(opt6_slopeavg1m, {_w_short()})"),
        ),
        lambda: (
            "Opt_IV_Gated",
            f"trade_when(ts_mean(opt6_fcstr2imp, {_w_med()}) > 0.5, {_wrap_hygiene(f'-1 * ts_zscore(opt6_slopeavg1m, {_w_med()})')}, -1)",
        ),
    ],
    "PV_RESIDUAL_HUMP": [
        lambda: (
            "PV_Resid_Mean",
            _wrap_hygiene(f"-1 * ts_regression(returns, ts_mean(returns, 5), {_w_med()}, rettype=0)"),
        ),
        lambda: (
            "PV_Vwap_Reversion",
            _wrap_hygiene(f"-1 * ts_corr(open, volume, {_w_short()}) * rank((close - vwap) / (vwap + 0.0001))"),
        ),
        lambda: (
            "PV_Intraday_Eff",
            _wrap_hygiene(f"-1 * ts_mean((close - open) / (high - low + 0.001), {_w_short()})"),
        ),
        lambda: (
            "PV_Vol_Price_Div",
            _wrap_hygiene(f"-1 * ts_corr(ts_delta(close, 1), ts_delta(volume, 1), {_w_med()})"),
        ),
        lambda: (
            "PV_Range_Break",
            _wrap_hygiene(f"(close - ts_min(low, {_w_med()})) / (ts_max(high, {_w_med()}) - ts_min(low, {_w_med()}) + 0.001)"),
        ),
    ],
    "VOLATILITY_LIQ_DECAY": [
        lambda: (
            "VolLiq_AvDiff",
            _wrap_hygiene(f"rank(ts_av_diff(close, {_w_med()})) * rank(volume * close)"),
        ),
        lambda: (
            "VolLiq_LowVol_Anomaly",
            _wrap_hygiene(f"-1 * ts_std_dev(returns, {_w_med()}) / (ts_mean(abs(returns), {_w_long()}) + 0.0001)"),
        ),
        lambda: (
            "VolLiq_Amihud_Illiq",
            _wrap_hygiene(f"-1 * ts_mean(abs(returns) / (volume * close + 1.0), {_w_med()})"),
        ),
        lambda: (
            "VolLiq_Skew_Rev",
            _wrap_hygiene(f"-1 * ts_skewness(returns, {_w_med()})"),
        ),
    ],
}


def generate_diverse_domain_candidate(domain: str | None = None) -> dict[str, str]:
    """Generate a structurally diverse FASTEXPR candidate across 25+ sub-families."""
    domains = list(_TEMPLATE_FAMILIES.keys())
    dom = domain if domain in _TEMPLATE_FAMILIES else random.choice(domains)
    builder = random.choice(_TEMPLATE_FAMILIES[dom])
    name, expr = builder()
    return {"name": f"Exp_{name}_{random.randint(100, 999)}", "expression": expr, "domain": dom}


def list_template_family_count() -> int:
    return sum(len(v) for v in _TEMPLATE_FAMILIES.values())
