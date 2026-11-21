"""Diverse multi-domain WorldQuant BRAIN FASTEXPR template families to avoid 7-skeleton saturation."""
from __future__ import annotations

import random
from typing import Callable


def _grp() -> str:
    return random.choice(["subindustry", "industry", "sector"])


def _w_short() -> int:
    return random.choice([3, 5, 7, 8, 10, 11, 13, 15, 17, 19, 20])


def _w_med() -> int:
    return random.choice([20, 23, 29, 30, 37, 40, 43, 53, 60, 63, 73, 89, 90, 107, 120])


def _w_long() -> int:
    return random.choice([60, 67, 89, 90, 113, 120, 149, 180, 199, 240])


def _hump() -> float:
    return random.choice([0.005, 0.008, 0.01, 0.012, 0.015, 0.02])


def _wrap_hygiene(
    core: str,
    grp: str | None = None,
    h: float | None = None,
    transform: str | None = None,
) -> str:
    g = grp or _grp()
    hv = h if h is not None else _hump()
    std_v = random.choice([3, 3.5, 4])
    inner = core
    if transform == "TS_REGRESSION_RESID" and "ts_regression" not in inner:
        inner = f"ts_regression(rank({inner}), returns, {_w_med()}, rettype=0)"
    elif transform == "REGRESSION_NEUT_CAP" and "regression_neut" not in inner:
        inner = f"regression_neut(rank({inner}), log(cap + 1.0))"
    elif transform == "REGRESSION_NEUT_VOL" and "regression_neut" not in inner:
        inner = f"regression_neut(rank({inner}), ts_std_dev(returns, 20))"
    elif transform == "TRADE_WHEN_VOL_EVENT" and "trade_when" not in inner:
        base_sig = f"group_neutralize(winsorize(rank({inner}), std={std_v}), {g})"
        return f"trade_when(volume > ts_mean(volume, 20), {base_sig}, -1)"
    elif transform == "ZSCORE_RANK" and "ts_zscore" not in inner:
        inner = f"ts_zscore({inner}, {_w_med()})"
    elif transform == "SUB_UNIVERSE_LIQ_BOOST":
        inner = f"(0.5 + 0.5 * rank(ts_mean(volume * close, 21))) * rank({inner})"
    elif transform == "NONLINEAR_SIGNED_POWER" and "signed_power" not in inner:
        inner = f"signed_power(ts_zscore({inner}, {_w_short()}), 0.75)"
    return f"hump(group_neutralize(winsorize(rank({inner}), std={std_v}), {g}), hump={hv})"


_TEMPLATE_FAMILIES: dict[str, list[Callable[..., tuple[str, str]]]] = {
    "FUNDAMENTAL_CF_EV": [
        lambda grp=None, transform=None: (
            "Fund_CF_EV",
            _wrap_hygiene(f"{random.choice(['cashflow_op', 'operating_income', 'ebitda'])} / (cap + debt + 0.001)", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "Fund_FCF_Delta",
            _wrap_hygiene(f"ts_delta(cashflow_op / (assets + 0.001), {_w_med()})", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "Fund_Sales_Margin",
            _wrap_hygiene(f"ts_zscore(operating_income / (sales + 0.001), {_w_long()})", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "Fund_Accrual_Quality",
            _wrap_hygiene("(cashflow_op - operating_income) / (assets + 0.001)", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "Fund_EBITDA_EV_Ir",
            _wrap_hygiene(f"ts_mean(ebitda / (cap + debt + 0.001), {_w_med()}) / (ts_std_dev(ebitda / (cap + debt + 0.001), {_w_med()}) + 0.001)", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "Fund_CF_DebtService",
            _wrap_hygiene(f"(cashflow_op - ts_delta(debt, {_w_med()})) / (cap + 0.001)", grp=grp, transform=transform),
        ),
    ],
    "FUNDAMENTAL_ROA_DELEVERAGE": [
        lambda grp=None, transform=None: (
            "Fund_ROA_Delev",
            _wrap_hygiene(f"rank(ebitda / (assets + 0.001)) - rank(ts_delta(debt / (assets + 0.001), {_w_med()}))", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "Fund_AssetTurn_Z",
            _wrap_hygiene(f"ts_zscore(sales / (assets + 0.001), {_w_long()})", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "Fund_Equity_Lev",
            _wrap_hygiene(f"-1 * ts_delta(liabilities / (equity + 0.001), {_w_med()})", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "Fund_ROIC_Trend",
            _wrap_hygiene(f"ts_mean(operating_income / (assets - liabilities + 0.001), {_w_med()}) / (ts_std_dev(operating_income / (assets - liabilities + 0.001), {_w_med()}) + 0.001)", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "Fund_GrossProf_Assets",
            _wrap_hygiene("(operating_income + ebitda) / (2 * assets + 0.001)", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "Fund_CapEff_Cash",
            _wrap_hygiene("rank(cashflow_op / (equity + abs(debt) + 0.001)) * rank(sales / (assets + 0.001))", grp=grp, transform=transform),
        ),
    ],
    "ANALYST_EPS_DISPERSION": [
        lambda grp=None, transform=None: (
            "Analyst_EPS_Target",
            _wrap_hygiene(f"rank(ts_zscore(est_eps, {_w_med()})) - 0.5 * rank(anl4_fs_detail_estimate_1qf_v4_nd_epsr_high / (close + 0.001))", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "Analyst_EPS_Orthog",
            _wrap_hygiene(f"ts_regression(rank(ts_delta(est_eps, {_w_short()})), rank(ts_delta(close, {_w_short()})), {_w_med()}, rettype=0)", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "Analyst_Target_Gap",
            _wrap_hygiene(f"ts_zscore((anl4_fs_detail_estimate_1qf_v4_nd_epsr_high - anl4_fs_detail_estimate_1qf_v4_nd_epsr_mean) / (close + 0.001), {_w_med()})", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "Analyst_EPS_Accel",
            _wrap_hygiene(f"ts_delta(anl4_qf_az_eps, {_w_short()}) - ts_delay(ts_delta(anl4_qf_az_eps, {_w_short()}), {_w_short()})", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "Analyst_EPS_Ir",
            _wrap_hygiene(f"ts_mean(est_eps, {_w_med()}) / (ts_std_dev(est_eps, {_w_med()}) + 0.001)", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "Analyst_Rev_Yield",
            _wrap_hygiene(f"rank(est_eps / (close + 0.001)) + rank(ts_delta(anl4_qf_az_eps, {_w_short()}) / (close + 0.001))", grp=grp, transform=transform),
        ),
    ],
    "SENTIMENT_BUZZ_MOMENTUM": [
        lambda grp=None, transform=None: (
            "Sent_Buzz_Mom",
            _wrap_hygiene(f"rank(ts_zscore(scl12_buzz, {_w_med()})) * rank(ts_delta(close, {_w_short()}) / (close + 0.001))", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "Sent_Buzz_Stab",
            _wrap_hygiene(f"-1 * ts_std_dev(ts_backfill(scl12_buzz, 5), {_w_short()})", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "Sent_Buzz_Resid",
            _wrap_hygiene(f"ts_regression(ts_backfill(scl12_buzz, 5), returns, {_w_med()}, rettype=0)", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "Sent_Buzz_Ir",
            _wrap_hygiene(f"(ts_mean(ts_backfill(scl12_buzz, 5), {_w_short()}) / (ts_std_dev(ts_backfill(scl12_buzz, 5), {_w_short()}) + 0.001)) * rank(volume / (ts_mean(volume, {_w_med()}) + 0.001))", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "Sent_Buzz_PriceConfirm",
            _wrap_hygiene(f"rank(ts_delta(ts_backfill(scl12_buzz, 5), {_w_short()})) - rank(ts_delta(close, {_w_short()}) / (close + 0.001))", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "Sent_Buzz_VolRatio",
            _wrap_hygiene(f"ts_zscore(ts_backfill(scl12_buzz, 5), {_w_med()}) / (ts_std_dev(returns, {_w_short()}) + 0.001)", grp=grp, transform=transform),
        ),
    ],
    "OPTIONS_IV_SLOPE": [
        lambda grp=None, transform=None: (
            "Opt_IV_Slope",
            _wrap_hygiene(f"rank(ts_zscore(implied_volatility_call_60 - implied_volatility_put_60, {_w_med()})) * rank(implied_volatility_mean_60)", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "Opt_IV_MeanRev",
            _wrap_hygiene(f"-1 * ts_zscore(implied_volatility_mean_skew_180, {_w_short()})", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "Opt_IV_Gated",
            f"trade_when(ts_mean(parkinson_volatility_60, {_w_med()}) > 0.1, {_wrap_hygiene(f'-1 * ts_zscore(implied_volatility_mean_skew_180, {_w_med()})', grp=grp, transform=transform)}, -1)",
        ),
        lambda grp=None, transform=None: (
            "Opt_IV_Ir",
            _wrap_hygiene(f"(ts_mean(implied_volatility_mean_skew_180, {_w_med()}) / (ts_std_dev(implied_volatility_mean_skew_180, {_w_med()}) + 0.001)) * rank(implied_volatility_mean_60)", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "Opt_IV_Resid_Ret",
            _wrap_hygiene(f"ts_regression(implied_volatility_call_60 - implied_volatility_put_60, returns, {_w_med()}, rettype=0)", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "Opt_IV_Delta_Conf",
            _wrap_hygiene(f"-1 * ts_delta(implied_volatility_mean_skew_180, {_w_short()}) * rank(ts_mean(historical_volatility_60, {_w_short()}))", grp=grp, transform=transform),
        ),
    ],
    "PV_RESIDUAL_HUMP": [
        lambda grp=None, transform=None: (
            "PV_Resid_Mean",
            _wrap_hygiene(f"-1 * ts_regression(returns, ts_mean(returns, 5), {_w_med()}, rettype=0)", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "PV_Vwap_Reversion",
            _wrap_hygiene(f"-1 * ts_corr(open, volume, {_w_short()}) * rank((close - vwap) / (vwap + 0.0001))", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "PV_Intraday_Eff",
            _wrap_hygiene(f"-1 * ts_mean((close - open) / (high - low + 0.001), {_w_short()})", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "PV_Vol_Price_Div",
            _wrap_hygiene(f"-1 * ts_corr(ts_delta(close, 1), ts_delta(volume, 1), {_w_med()})", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "PV_Range_Break",
            _wrap_hygiene(f"ts_scale(close, {_w_med()})", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "PV_Ir_VolConfirm",
            _wrap_hygiene(f"rank(ts_mean(returns, {_w_short()}) / (ts_std_dev(returns, {_w_short()}) + 0.001)) * rank(volume / (ts_mean(volume, {_w_med()}) + 0.001))", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "PV_CloseVwap_Reg",
            _wrap_hygiene(f"-1 * ts_regression((close - vwap) / (vwap + 0.0001), returns, {_w_short()}, rettype=0)", grp=grp, transform=transform),
        ),
    ],
    "VOLATILITY_LIQ_DECAY": [
        lambda grp=None, transform=None: (
            "VolLiq_AvDiff",
            _wrap_hygiene(f"rank(ts_av_diff(close, {_w_med()})) * rank(volume * close)", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "VolLiq_LowVol_Anomaly",
            _wrap_hygiene(f"-1 * ts_std_dev(returns, {_w_med()}) / (ts_mean(abs(returns), {_w_long()}) + 0.0001)", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "VolLiq_Amihud_Illiq",
            _wrap_hygiene(f"-1 * ts_mean(abs(returns) / (volume * close + 1.0), {_w_med()})", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "VolLiq_Skew_Rev",
            _wrap_hygiene(f"-1 * ts_quantile(returns, {_w_med()})", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "VolLiq_Turn_Stability",
            _wrap_hygiene(f"-1 * ts_std_dev(volume * close, {_w_short()}) / (ts_mean(volume * close, {_w_med()}) + 1.0)", grp=grp, transform=transform),
        ),
        lambda grp=None, transform=None: (
            "VolLiq_UpDown_VolRatio",
            _wrap_hygiene(f"-1 * ts_corr(abs(returns), volume, {_w_short()}) * rank(ts_av_diff(close, {_w_med()}))", grp=grp, transform=transform),
        ),
    ],
    "OPTIONS_ORTHOGONAL": [
        lambda grp=None, transform=None: (
            "Opt_IV_RV_Spread",
            _wrap_hygiene(
                "implied_volatility_call_120 / (parkinson_volatility_120 + 0.001)",
                grp=grp,
                transform=transform or "REGRESSION_NEUT_CAP",
            ),
        ),
        lambda grp=None, transform=None: (
            "Opt_CP_Skew_TradeWhen",
            f"trade_when(pcr_oi_270 < 1, group_neutralize(rank(implied_volatility_call_60 - implied_volatility_put_60), {grp or _grp()}), -1)",
        ),
        lambda grp=None, transform=None: (
            "Opt_IV_Delta_Spread",
            _wrap_hygiene(
                f"ts_delta(implied_volatility_call_60, {_w_short()}) - ts_delta(implied_volatility_put_60, {_w_short()})",
                grp=grp,
                transform=transform or "REGRESSION_NEUT_VOL",
            ),
        ),
    ],
    "FUND_ANALYST_ORTHO": [
        lambda grp=None, transform=None: (
            "Retained_Earnings_PerShare_Delta",
            _wrap_hygiene(
                f"ts_delta(retained_earnings / (sharesout + 0.001), {_w_med()})",
                grp=grp,
                transform=transform or "REGRESSION_NEUT_CAP",
            ),
        ),
        lambda grp=None, transform=None: (
            "OpInc_Over_Close_Ortho",
            _wrap_hygiene(
                f"ts_rank(operating_income / (close + 0.001), {_w_long()})",
                grp=grp,
                transform=transform or "REGRESSION_NEUT_VOL",
            ),
        ),
    ],
}


def generate_diverse_domain_candidate(
    domain: str | None = None,
    *,
    neutralization: str | None = None,
    transform: str | None = None,
) -> dict[str, str]:
    """Generate a structurally diverse FASTEXPR candidate across 40+ sub-families."""
    domains = list(_TEMPLATE_FAMILIES.keys())
    dom = domain if domain in _TEMPLATE_FAMILIES else random.choice(domains)
    builder = random.choice(_TEMPLATE_FAMILIES[dom])
    grp = neutralization.lower() if (neutralization and neutralization.lower() in ("subindustry", "industry", "sector")) else None
    name, expr = builder(grp=grp, transform=transform)
    return {"name": f"Exp_{name}_{random.randint(100, 999)}", "expression": expr, "domain": dom}


def list_template_family_count() -> int:
    return sum(len(v) for v in _TEMPLATE_FAMILIES.values())
