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
        "nanHandling": "ON" if any(k in dom for k in ("FUND", "ANALYST", "OPT", "SENTIMENT", "NEWS", "COMBO", "MODEL")) else "OFF",
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

    import re as _re

    if float(sharpe) <= -1.05:
        repairs.append({
            "name": "Repair_NegativeSharpe_SignFlip",
            "expression": expr[5:].strip() if expr.startswith("-1 * ") else f"-1 * ({expr})",
            "settings": dict(base),
            "reason": "NEGATIVE_SHARPE_SIGN_FLIP",
        })

    if any("CONCENTRATED_WEIGHT" in c or "WEIGHT" in c for c in failed):
        s_w = dict(base)
        s_w["truncation"] = min(0.05, float(base.get("truncation", 0.08) or 0.08))
        s_w["nanHandling"] = "ON"
        repairs.append({
            "name": "Repair_ConcentratedWeight_RankBackfill",
            "expression": f"group_neutralize(rank(ts_backfill({expr}, 20)), subindustry)",
            "settings": s_w,
            "reason": "CONCENTRATED_WEIGHT_RANK_BACKFILL_TRUNC05",
        })

    if any("2Y_SHARPE" in c for c in failed):
        s_2y = dict(base)
        s_2y["decay"] = min(10, max(4, int(base.get("decay", 4) or 4) + 2))
        repairs.append({
            "name": "Repair_Low2YSharpe_JumpDecay",
            "expression": f"jump_decay(group_neutralize(winsorize(rank({expr}), std=3.5), subindustry), 10, sensitivity=0.5)",
            "settings": s_2y,
            "reason": "LOW_2Y_SHARPE_JUMP_DECAY_STABILIZE",
        })

    if any("CORR" in c for c in failed):
        repairs.extend(generate_decorrelation_variants(expr, base, max_variants=4))

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


REGION_UNIVERSE_POOL: tuple[tuple[str, str], ...] = (
    ("USA", "TOP3000"),
    ("USA", "TOP1000"),
    ("EUR", "TOP2500"),
    ("ASI", "MINVOL1M"),
    ("GLB", "TOP3000"),
)

ADVANCED_NEUTRALIZATIONS: tuple[str, ...] = (
    "SUBINDUSTRY",
    "INDUSTRY",
    "CROWDING",
    "REVERSION_AND_MOMENTUM",
    "STATISTICAL",
)


def pick_low_crowding_region_settings(enable_multi_region: bool = False) -> dict[str, str]:
    """Rotate across low-crowding regions (when enabled) and advanced risk-model neutralizations."""
    if enable_multi_region or os.getenv("BRAIN_ENABLE_MULTI_REGION", "false").lower() == "true":
        region, universe = random.choice(REGION_UNIVERSE_POOL)
    else:
        region, universe = "USA", random.choice(["TOP3000", "TOP1000", "TOP500"])
    neut = random.choice(ADVANCED_NEUTRALIZATIONS)
    return {"region": region, "universe": universe, "neutralization": neut}


def generate_decorrelation_variants(
    expression: str,
    settings: dict[str, Any] | None = None,
    *,
    max_variants: int = 4,
    unavailable_neutralizations: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Official WorldQuant BRAIN 4-pillar de-correlation rescue for good Alphas hitting SELF/PROD correlation.

    1. Equivalent data field substitution (close -> vwap / open / (high+low)/2).
    2. Operator substitution (rank -> zscore, ts_mean -> ts_median, ts_zscore -> ts_rank).
    3. Meaningful grouping & neutralization rotation (SUBINDUSTRY -> INDUSTRY / SECTOR / CROWDING).
    4. Orthogonal conditioning / cross-domain combo (ts_regression residual or orthogonal fundamental/analyst blend).
    """
    import re as _re

    expr = (expression or "").strip()
    if not expr:
        return []
    base = dict(settings or generate_smart_settings("COMBO"))
    unavail = {str(u).upper() for u in (unavailable_neutralizations or set())}
    out: list[dict[str, Any]] = []
    seen_exprs: set[tuple[str, str]] = {(expr, str(base.get("neutralization") or "").upper())}

    def _add(name: str, new_expr: str, new_sett: dict[str, Any], reason: str) -> None:
        if len(out) >= max_variants:
            return
        key = (new_expr.strip(), str(new_sett.get("neutralization") or "").upper())
        if not new_expr.strip() or key in seen_exprs:
            return
        seen_exprs.add(key)
        out.append({"name": name, "expression": new_expr.strip(), "settings": new_sett, "reason": reason})

    # Pillar 1: Equivalent data field substitution
    if _re.search(r"\bclose\b", expr):
        _add("Decorr_FieldSub_Vwap", _re.sub(r"\bclose\b", "vwap", expr), dict(base), "CORR_FIELD_SUBSTITUTION_VWAP")
        _add("Decorr_FieldSub_Open", _re.sub(r"\bclose\b", "open", expr), dict(base), "CORR_FIELD_SUBSTITUTION_OPEN")

    # Pillar 2: Similar operator substitution (median vs mean, zscore vs rank, ts_rank vs ts_zscore)
    if _re.search(r"\bts_mean\s*\(", expr):
        _add("Decorr_OpSub_TsMedian", _re.sub(r"\bts_mean\s*\(", "ts_median(", expr, count=1), dict(base), "CORR_OP_SUBSTITUTION_TS_MEDIAN")
    elif _re.search(r"\bts_zscore\s*\(", expr):
        _add("Decorr_OpSub_TsRank", _re.sub(r"\bts_zscore\s*\(", "ts_rank(", expr, count=1), dict(base), "CORR_OP_SUBSTITUTION_TS_RANK")
    elif _re.search(r"\brank\s*\(", expr):
        _add("Decorr_OpSub_Zscore", _re.sub(r"\brank\s*\(", "winsorize(zscore(", expr, count=1).replace(")", "), std=4)", 1) if expr.count(")") >= 1 else expr, dict(base), "CORR_OP_SUBSTITUTION_ZSCORE")

    # Pillar 3: Meaningful grouping & neutralization rotation (no arbitrary groups)
    cur_neut = str(base.get("neutralization") or "SUBINDUSTRY").upper()
    for alt_neut in ("INDUSTRY", "SECTOR", "CROWDING", "MARKET"):
        if alt_neut != cur_neut and alt_neut not in unavail:
            s_neut = dict(base)
            s_neut["neutralization"] = alt_neut
            _add(f"Decorr_Neut_{alt_neut}", expr, s_neut, f"CORR_NEUTRALIZATION_{alt_neut}")
            break

    # Pillar 4: Orthogonal conditioning / cross-domain blend
    _add(
        "Decorr_Orthog_RetResid",
        f"group_neutralize(ts_regression(winsorize(rank({expr}), std=4), returns, 60, lag=0, rettype=0), industry)",
        dict(base),
        "CORR_ORTHOG_RETURN_RESIDUAL",
    )
    atom = random.choice(_HIGH_YIELD_ATOMS["FUND_QUALITY"] + _HIGH_YIELD_ATOMS["ANALYST_REV"])
    _add(
        "Decorr_Orthog_Combo",
        f"hump(group_neutralize(0.5 * winsorize(rank({expr}), std=4) + 0.5 * winsorize(rank({atom}), std=4), industry), hump=0.01)",
        dict(base),
        "CORR_ORTHOG_COMBO",
    )
    return out[:max_variants]
