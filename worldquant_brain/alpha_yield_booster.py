"""Alpha Yield Booster: cross-domain orthogonal combos, smart universe/neutralization rotation, and gate-specific near-miss repair."""
from __future__ import annotations

import os
import random
from typing import Any


_HIGH_YIELD_ATOMS: dict[str, list[str]] = {
    "FUND_QUALITY": [
        "ebitda / (assets + 0.001)",
        "cashflow_op / (cap + debt + 0.001)",
        "(cashflow_op - operating_income) / (assets + 0.001)",
        "operating_income / (sales + 0.001)",
        "-1 * ts_delta(debt / (assets + 0.001), 60)",
        "sales / (assets + 0.001)",
    ],
    "ANALYST_REV": [
        "ts_zscore(est_eps, 40)",
        "ts_delta(est_eps, 20) / (abs(est_eps) + 0.01)",
        "est_eps / (close + 0.001)",
        "ts_regression(rank(ts_delta(anl4_qf_az_eps, 10)), rank(ts_delta(close, 10)), 40, rettype=0)",
    ],
    "SENTIMENT_OPT": [
        "ts_zscore(scl12_buzz, 40)",
        "-1 * ts_std_dev(ts_backfill(scl12_buzz, 5), 20)",
        "-1 * ts_zscore(implied_volatility_call_60 - implied_volatility_put_60, 40)",
        "ts_zscore(implied_volatility_mean_60, 40)",
    ],
    "PV_LOW_CORR": [
        "-1 * ts_regression(returns, ts_mean(returns, 5), 40, rettype=0)",
        "-1 * ts_std_dev(returns, 20) / (ts_mean(abs(returns), 120) + 0.0001)",
        "-1 * ts_mean(abs(returns) / (volume * close + 1.0), 20)",
        "-1 * ts_corr(open, volume, 10) * rank((close - vwap) / (vwap + 0.0001))",
        "-1 * ts_quantile(returns, 40)",
    ],
}

_COMBO_PAIRS: tuple[tuple[str, str], ...] = (
    ("FUND_QUALITY", "ANALYST_REV"),
    ("FUND_QUALITY", "PV_LOW_CORR"),
    ("ANALYST_REV", "PV_LOW_CORR"),
    ("FUND_QUALITY", "SENTIMENT_OPT"),
    ("ANALYST_REV", "SENTIMENT_OPT"),
    ("SENTIMENT_OPT", "PV_LOW_CORR"),
)


def generate_smart_settings(domain: str = "") -> dict[str, Any]:
    """Rotate less-crowded universes (TOP3000/TOP1000/TOP500) and domain-matched decay/neutralization."""
    dom = (domain or "").upper()
    if "FUND" in dom or "ANALYST" in dom or "COMBO" in dom:
        decays = [0, 2, 4]
        universes = ["TOP3000", "TOP1000", "TOP500"]
        neuts = ["SUBINDUSTRY", "INDUSTRY", "SECTOR"]
    elif "SENTIMENT" in dom or "OPT" in dom:
        decays = [4, 6, 8]
        universes = ["TOP1000", "TOP3000", "TOP500"]
        neuts = ["INDUSTRY", "SUBINDUSTRY", "SECTOR"]
    else:
        decays = [4, 6, 8, 10]
        universes = ["TOP1000", "TOP500", "TOP3000"]
        neuts = ["SUBINDUSTRY", "INDUSTRY", "MARKET"]

    _delay = random.choice([0, 1]) if os.getenv("IQC_ENABLE_DELAY0", "false").lower() == "true" else 1
    _cfg = {
        "instrumentType": "EQUITY",
        "region": "USA",
        "universe": random.choice(universes),
        "delay": _delay,
        "decay": random.choice(decays),
        "neutralization": random.choice(neuts),
        "truncation": random.choice([0.04, 0.05, 0.06]),
        "pasteurization": "ON",
        "unitHandling": "VERIFY",
        "nanHandling": "OFF",
        "language": "FASTEXPR",
        "visualization": False,
    }
    if os.getenv("IQC_USE_TEST_PERIOD", "false").lower() == "true":
        _cfg["testPeriod"] = os.getenv("IQC_TEST_PERIOD", "P1Y")
    return _cfg


def generate_orthogonal_combo_candidate() -> dict[str, str]:
    """Combine 2 orthogonal alpha domains (e.g. Fundamental Quality + Analyst Revision or Low-Corr PV)."""
    dom_a, dom_b = random.choice(_COMBO_PAIRS)
    atom_a = random.choice(_HIGH_YIELD_ATOMS[dom_a])
    atom_b = random.choice(_HIGH_YIELD_ATOMS[dom_b])
    grp = random.choice(["subindustry", "industry", "sector"])
    h = random.choice([0.008, 0.01, 0.012])
    mode = random.choice(["BLEND", "GATED", "VOL_SCALED", "VOL_SPIKE_EVENT", "GRADUAL_EVENT_DECAY", "ORTHOG_RESID"])

    if mode == "BLEND":
        w_a = random.choice([0.5, 0.6, 0.4])
        w_b = round(1.0 - w_a, 1)
        core = f"{w_a} * winsorize(rank({atom_a}), std=4) + {w_b} * winsorize(rank({atom_b}), std=4)"
        expr = f"hump(group_neutralize({core}, {grp}), hump={h})"
    elif mode == "GATED":
        # Official BRAIN liquidity-weighted dual decay (prevents Sub-Universe Sharpe NaN/cutoff failures)
        sig = f"group_neutralize(0.5 * winsorize(rank(ts_backfill({atom_a}, 5)), std=4) + 0.5 * winsorize(rank(ts_backfill({atom_b}, 5)), std=4), {grp})"
        expr = f"hump(ts_decay_linear({sig}, 5) * rank(volume * close) + ts_decay_linear({sig}, 10) * (1 - rank(volume * close)), hump={h})"
    elif mode == "VOL_SCALED":
        # BRAIN Tip: avoid concentrating weight on volatile stocks; scale by inverse realized volatility
        sig = f"(0.5 * winsorize(rank(ts_backfill({atom_a}, 5)), std=4) + 0.5 * winsorize(rank(ts_backfill({atom_b}, 5)), std=4)) / (ts_std_dev(returns, 20) + 0.005)"
        expr = f"hump(group_neutralize({sig}, {grp}), hump={h})"
    elif mode == "VOL_SPIKE_EVENT":
        # BRAIN Capacity Tip: trade when short-term vol spikes above long-term vol, hold otherwise
        sig = f"group_neutralize(0.5 * winsorize(rank(ts_backfill({atom_a}, 5)), std=4) + 0.5 * winsorize(rank(ts_backfill({atom_b}, 5)), std=4), {grp})"
        expr = f"hump(ts_backfill(trade_when(ts_std_dev(returns, 5) > ts_std_dev(returns, 60), {sig}, -1), 5), hump={h})"
    elif mode == "GRADUAL_EVENT_DECAY":
        # BRAIN Event Tip: blend current signal with last volume-surge event value to decay smoothly without flatlining
        sig = f"group_neutralize(0.5 * winsorize(rank(ts_backfill({atom_a}, 5)), std=4) + 0.5 * winsorize(rank(ts_backfill({atom_b}, 5)), std=4), {grp})"
        evt_held = f"ts_backfill(trade_when(rank(volume / ts_mean(volume, 20)) > 0.5, {sig}, -1), 5)"
        expr = f"hump(0.5 * {sig} + 0.5 * {evt_held}, hump={h})"
    else:
        expr = f"hump(group_neutralize(ts_regression(rank(ts_backfill({atom_a}, 5)), rank(ts_backfill({atom_b}, 5)), 60, rettype=0), {grp}), hump={h})"

    return {
        "name": f"Combo_{dom_a}_{dom_b}_{mode}",
        "expression": expr,
        "domain": f"COMBO_{dom_a}_{dom_b}",
    }


def precision_repair_near_miss(
    expression: str,
    settings: dict[str, Any] | None,
    *,
    sharpe: float,
    fitness: float,
    turnover: float,
    margin: float | None = None,
    failed_checks: list[str] | None = None,
    objective: str = "BALANCED",
    net_returns: float | None = None,
    previous_net_returns: float | None = None,
    previous_turnover: float | None = None,
    drawdown: float = 0.08,
) -> list[dict[str, Any]]:
    """Generate targeted repairs mapped to the exact failed BRAIN IS-check."""
    expr = (expression or "").strip()
    if not expr:
        return []
    failed = {str(c).upper() for c in (failed_checks or [])}
    base = dict(settings or generate_smart_settings("COMBO"))
    repairs: list[dict[str, Any]] = []

    if "LOW_MARGIN" in failed or (margin is not None and margin < 0.0012):
        s = dict(base)
        s["decay"] = min(12, max(8, int(base.get("decay", 4) or 4) + 4))
        s["truncation"] = 0.05
        repairs.append({
            "name": "Repair_LowMargin_LiqDecayHump",
            "expression": f"hump(ts_decay_linear(ts_backfill({expr}, 5), 5) * rank(volume * close) + ts_decay_linear(ts_backfill({expr}, 5), 10) * (1 - rank(volume * close)), hump=0.015)",
            "settings": s,
            "reason": "LOW_MARGIN_LIQ_DECAY_HUMP",
        })

    _obj_actions: set[str] = set()
    try:
        from worldquant_brain.optimization_policy import (
            AlphaPerformance,
            ObjectivePolicy,
            OptimizationObjective,
            recommend_actions,
        )
        _nr = float(net_returns) if net_returns is not None else max(0.0, float(sharpe) * 0.05)
        _perf = AlphaPerformance(
            returns=_nr,
            net_returns=_nr,
            sharpe=float(sharpe),
            fitness=float(fitness),
            turnover=float(turnover),
            margin=float(margin) if margin is not None else 0.001,
            drawdown=float(drawdown),
            previous_net_returns=previous_net_returns,
            previous_turnover=previous_turnover,
        )
        _pol = ObjectivePolicy(OptimizationObjective(str(objective).upper()))
        _obj_actions = {a.action for a in recommend_actions(_perf, _pol)}
    except Exception:
        _obj_actions = set()

    if "PRESERVE_TURNOVER_AND_STRESS_COST" in _obj_actions:
        repairs.append({
            "name": "Preserve_ProductiveTurnover_CostStress",
            "expression": expr,
            "settings": dict(base),
            "reason": "HIGH_TURNOVER_PRODUCTIVE_PRESERVED",
        })
    elif "HIGH_TURNOVER" in failed or "APPLY_HUMP" in _obj_actions or turnover > 0.55:
        s = dict(base)
        s["decay"] = min(10, max(6, int(base.get("decay", 4) or 4) + 2))
        repairs.append({
            "name": "Repair_WastefulTurnover_Hump",
            "expression": expr if expr.startswith("hump(") else f"hump({expr}, hump=0.01)",
            "settings": s,
            "reason": "WASTEFUL_TURNOVER_APPLY_HUMP",
        })

    if "LOWER_DECAY" in _obj_actions:
        s_low = dict(base)
        s_low["decay"] = max(0, int(base.get("decay", 4) or 4) - 2)
        repairs.append({
            "name": "Repair_StaleSignal_LowerDecay",
            "expression": expr,
            "settings": s_low,
            "reason": "LOW_TURNOVER_STALE_LOWER_DECAY",
        })

    if "LOW_SHARPE" in failed or abs(sharpe) < 1.25:
        s = dict(base)
        s["universe"] = "TOP1000" if str(base.get("universe", "TOP3000")) == "TOP3000" else "TOP500"
        s["neutralization"] = "INDUSTRY" if str(base.get("neutralization", "SUBINDUSTRY")) == "SUBINDUSTRY" else "SUBINDUSTRY"
        repairs.append({
            "name": f"Repair_Universe_{s['universe']}",
            "expression": expr,
            "settings": s,
            "reason": f"ROTATE_UNIVERSE_{s['universe']}",
        })

    if any("SUB" in c or "NAN" in c for c in failed):
        s_sub = dict(base)
        s_sub["universe"] = "TOP1000"
        repairs.append({
            "name": "Repair_SubUniverse_LiqDecay",
            "expression": f"ts_decay_linear(ts_backfill({expr}, 5), 5) * rank(volume * close) + ts_decay_linear(ts_backfill({expr}, 5), 10) * (1 - rank(volume * close))",
            "settings": s_sub,
            "reason": "SUB_UNIVERSE_LIQ_DECAY_BACKFILL",
        })

    if "LOW_FITNESS" in failed or (fitness < 1.0 and abs(sharpe) >= 1.10):
        s_fit = dict(base)
        if turnover > 0.20:
            s_fit["decay"] = min(10, max(6, int(base.get("decay", 4) or 4) + 2))
            repairs.append({
                "name": "Repair_Fitness_TargetTurnover125",
                "expression": expr if expr.startswith("ts_decay_linear(") else f"ts_decay_linear({expr}, 6)",
                "settings": s_fit,
                "reason": "LOW_FITNESS_REDUCE_TURNOVER_TO_SWEET_SPOT",
            })
        else:
            # Turnover already near 0.125 floor -> boost Returns via convex tail concentration
            s_fit["decay"] = 2
            s_fit["truncation"] = 0.08
            repairs.append({
                "name": "Repair_Fitness_BoostReturns",
                "expression": f"group_neutralize(winsorize({expr}, std=3.5), subindustry)" if any(k in expr.lower() for k in ("scl12_", "snt_", "buzz", "sentiment")) else f"group_neutralize(signed_power(rank({expr}) - 0.5, 1.5), subindustry)",
                "settings": s_fit,
                "reason": "LOW_FITNESS_BOOST_RETURNS_CONCENTRATION",
            })

    if any("CORR" in c for c in failed):
        atom = random.choice(_HIGH_YIELD_ATOMS["FUND_QUALITY"] + _HIGH_YIELD_ATOMS["ANALYST_REV"])
        repairs.append({
            "name": "Repair_Corr_OrthogCombo",
            "expression": f"hump(group_neutralize(0.5 * winsorize(rank({expr}), std=4) + 0.5 * winsorize(rank({atom}), std=4), industry), hump=0.01)",
            "settings": dict(base),
            "reason": "CORR_ORTHOG_COMBO",
        })
        if "close" in expr:
            repairs.append({
                "name": "Repair_Corr_FieldSubVwap",
                "expression": expr.replace("close", "vwap"),
                "settings": dict(base),
                "reason": "CORR_FIELD_SUBSTITUTION_VWAP",
            })

    return repairs


def maybe_invert_negative_sharpe_candidate(
    expression: str,
    metrics: dict[str, Any],
    *,
    min_abs_sharpe: float = 1.25,
    max_turnover: float = 0.65,
) -> str | None:
    """Flip strongly negative-Sharpe alphas (-1 * expr) to salvage inverted economic signals."""
    sharpe = float(metrics.get("sharpe") or 0.0)
    turnover = float(metrics.get("turnover") or 1.0)
    if sharpe <= -abs(min_abs_sharpe) and turnover <= max_turnover:
        clean = expression.strip()
        if clean.startswith("-1 * (") and clean.endswith(")"):
            return clean[6:-1].strip()
        return f"-1 * ({clean})"
    return None
