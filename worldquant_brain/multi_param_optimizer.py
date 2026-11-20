"""Multi-Parameter Adaptive Alpha Optimizer & Winner-Only Family Submission Pool.

Integrates:
  1. Universal FastExpr numeric parameter extractor (windows, exponents, std, hump,
     weights, plus simulation settings decay/truncation; skips epsilon guards & signs).
  2. Adaptive multi-method search:
     - Binary Search (monotonic constraints: Turnover, Weight concentration).
     - Ternary / Golden-Section Search (unimodal curves: Sharpe/Fitness vs window/decay).
     - Directional Coordinate Ascent (multi-parameter joint tuning that observes
       before->after metric deltas dSharpe/dParam, dFitness/dParam, dTurnover/dParam
       to choose increasing vs decreasing direction, with bounded trial cap to prevent
       DSR overfitting).
  3. Polish-After-Pass & Winner-Only Submission Pool:
     - Passing alphas can still be polished for higher Sharpe/Fitness/Margin.
     - All dominated / intermediate passing tuned variants are absorbed into
       `PASSED_DO_NOT_SUBMIT` (preserved for parameter-response evidence, never submitted).
     - ONLY the single best variant per `structural_hash` family is promoted to
       `SUBMITTABLE_BEST` (and automatically demotes any prior family champion).
  4. Historical Failed-Case Rescue Scanner:
     - Inspects past `brain_jobs` in SQLite for strong negative-Sharpe gems (<= -1.25)
       and near-miss cases (>= 1.10) not yet rescued or tried.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

from worldquant_brain.canonical_ast import parse_fastexpr


class SearchMethod(StrEnum):
    BINARY = "BINARY"
    TERNARY = "TERNARY"
    COORDINATE_ASCENT = "COORDINATE_ASCENT"


class PoolDisposition(StrEnum):
    PASSED_DO_NOT_SUBMIT = "PASSED_DO_NOT_SUBMIT"
    SUBMITTABLE_BEST = "SUBMITTABLE_BEST"
    FAILED = "FAILED"


@dataclass(frozen=True)
class TunableParam:
    """Represents a single tunable numeric parameter inside an expression or settings."""

    name: str
    location: str  # "EXPRESSION" or "SETTING"
    current: float
    low: float
    high: float
    is_int: bool
    span_start: int = -1
    span_end: int = -1
    role: str = "WINDOW"  # WINDOW | HUMP | STD | EXPONENT | WEIGHT | DECAY | TRUNCATION

    def format_value(self, value: float) -> str:
        if self.is_int:
            return str(int(round(max(self.low, min(self.high, value)))))
        clamped = max(self.low, min(self.high, value))
        return f"{clamped:.4f}".rstrip("0").rstrip(".")


@dataclass(frozen=True)
class ParamDeltaObservation:
    """Records how changing one parameter affected alpha metrics (before -> after)."""

    param_name: str
    before_value: float
    after_value: float
    delta_param: float
    delta_sharpe: float
    delta_fitness: float
    delta_turnover: float
    delta_margin: float
    improved_score: bool

    @property
    def recommended_direction(self) -> str:
        """Return 'UPPER' to keep moving up, 'LOWER' to move down, or 'STOP' if flat."""
        if abs(self.delta_param) < 1e-9:
            return "STOP"
        if self.improved_score:
            return "UPPER" if self.delta_param > 0 else "LOWER"
        return "LOWER" if self.delta_param > 0 else "UPPER"


_EPSILON_LITERALS: frozenset[str] = frozenset({"0.001", "0.0001", "0.00001", "1e-4", "1e-5", "1e-6", "0.0"})


def extract_tunable_parameters(
    expression: str,
    settings: Mapping[str, Any] | None = None,
) -> list[TunableParam]:
    """Extract all tunable numeric parameters from a FastExpr formula and its simulation settings."""
    expr = str(expression or "")
    params: list[TunableParam] = []

    # 1. Named keyword parameters: hump=..., std=...
    for m in re.finditer(r"\b(hump|std)\s*=\s*(\d+(?:\.\d+)?)", expr):
        kw = m.group(1).lower()
        val = float(m.group(2))
        s_idx, e_idx = m.span(2)
        if kw == "hump":
            params.append(TunableParam(f"expr_hump_{s_idx}", "EXPRESSION", val, 0.002, 0.030, False, s_idx, e_idx, "HUMP"))
        else:
            params.append(TunableParam(f"expr_std_{s_idx}", "EXPRESSION", val, 2.0, 5.0, False, s_idx, e_idx, "STD"))

    # Balanced-parenthesis helper to locate top-level arguments inside nested calls
    def _top_level_arg_spans(text: str, open_paren_idx: int) -> list[tuple[int, int]]:
        depth = 0
        arg_start = open_paren_idx + 1
        spans: list[tuple[int, int]] = []
        for idx in range(open_paren_idx, len(text)):
            ch = text[idx]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    spans.append((arg_start, idx))
                    break
            elif ch == "," and depth == 1:
                spans.append((arg_start, idx))
                arg_start = idx + 1
        return spans

    used_spans = {(p.span_start, p.span_end) for p in params}

    # 2. Time-series integer windows (works even with nested sub-expressions)
    for m in re.finditer(r"\b(ts_[a-z_]+)\s*\(", expr):
        op = m.group(1).lower()
        open_idx = m.end() - 1
        for a_s, a_e in _top_level_arg_spans(expr, open_idx)[1:]:
            raw = expr[a_s:a_e]
            nm = re.fullmatch(r"\s*(\d+)\s*", raw)
            if not nm:
                continue
            s_idx = a_s + nm.start(1)
            e_idx = a_s + nm.end(1)
            if (s_idx, e_idx) in used_spans:
                continue
            used_spans.add((s_idx, e_idx))
            val = int(nm.group(1))
            lo = 5.0 if op in ("ts_corr", "ts_covariance", "ts_regression") else 2.0
            hi = float(min(252, max(60, val * 3)))
            params.append(TunableParam(f"expr_{op}_w_{s_idx}", "EXPRESSION", float(val), lo, hi, True, s_idx, e_idx, "WINDOW"))

    # 3. signed_power exponent (works even with nested sub-expressions)
    for m in re.finditer(r"\bsigned_power\s*\(", expr):
        open_idx = m.end() - 1
        args = _top_level_arg_spans(expr, open_idx)
        if len(args) >= 2:
            a_s, a_e = args[1]
            raw = expr[a_s:a_e]
            nm = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", raw)
            if nm:
                s_idx = a_s + nm.start(1)
                e_idx = a_s + nm.end(1)
                if (s_idx, e_idx) not in used_spans:
                    used_spans.add((s_idx, e_idx))
                    val = float(nm.group(1))
                    params.append(TunableParam(f"expr_power_{s_idx}", "EXPRESSION", val, 0.25, 2.5, False, s_idx, e_idx, "EXPONENT"))

    # 4. Simulation settings: decay & truncation
    sett = dict(settings or {})
    if "decay" in sett and sett["decay"] is not None:
        d_val = float(int(sett["decay"]))
        params.append(TunableParam("setting_decay", "SETTING", d_val, 0.0, 20.0, True, role="DECAY"))
    if "truncation" in sett and sett["truncation"] is not None:
        t_val = float(sett["truncation"])
        params.append(TunableParam("setting_truncation", "SETTING", t_val, 0.02, 0.10, False, role="TRUNCATION"))

    return params


def apply_parameter_value(
    expression: str,
    settings: Mapping[str, Any],
    param: TunableParam,
    new_value: float,
) -> tuple[str, dict[str, Any]]:
    """Apply a new numeric value to either the expression span or settings dict."""
    new_sett = dict(settings)
    if param.location == "SETTING":
        key = "decay" if param.role == "DECAY" else "truncation"
        new_sett[key] = int(round(new_value)) if param.is_int else round(float(new_value), 4)
        return expression, new_sett
    formatted = param.format_value(new_value)
    new_expr = expression[: param.span_start] + formatted + expression[param.span_end :]
    return new_expr, new_sett


def compute_alpha_composite_score(metrics: Mapping[str, Any]) -> float:
    """Composite tuning objective combining Sharpe, Fitness, Margin, and Turnover health."""
    sh = float(metrics.get("sharpe") or 0.0)
    fit = float(metrics.get("fitness") or 0.0)
    to = float(metrics.get("turnover") if metrics.get("turnover") is not None else 0.25)
    margin = metrics.get("margin")
    m_bps = (float(margin) * 10000.0) if (margin is not None and abs(float(margin)) < 0.5) else float(margin or 0.0)

    # Penalize extreme turnover (>0.45 or <0.01) softly while rewarding Sharpe/Fitness/Margin
    to_pen = 0.0
    if to > 0.45:
        to_pen = (to - 0.45) * 2.5
    elif to < 0.01:
        to_pen = 0.5
    margin_bonus = min(0.4, max(-0.4, m_bps / 50.0))
    return round(0.55 * sh + 0.35 * fit + margin_bonus - to_pen, 6)


def ternary_search_points(low: float, high: float, *, is_int: bool = False) -> tuple[float, float]:
    """Return the two trisection interior probe points (m1, m2) for unimodal ternary search."""
    m1 = low + (high - low) / 3.0
    m2 = high - (high - low) / 3.0
    if is_int:
        i1, i2 = int(round(m1)), int(round(m2))
        if i1 == i2 and (high - low) >= 2:
            i2 = min(int(high), i1 + 1)
        return float(i1), float(i2)
    return round(m1, 4), round(m2, 4)


def shrink_ternary_bracket(
    low: float,
    high: float,
    m1: float,
    m2: float,
    score_m1: float,
    score_m2: float,
) -> tuple[float, float]:
    """Shrink [low, high] by 1/3 toward the higher-scoring trisection side."""
    if score_m1 < score_m2:
        return m1, high
    return low, m2


def observe_parameter_delta(
    param: TunableParam,
    before_metrics: Mapping[str, Any],
    after_metrics: Mapping[str, Any],
    after_value: float,
) -> ParamDeltaObservation:
    """Measure how changing `param` shifted Sharpe, Fitness, Turnover, Margin, and composite score."""
    b_sc = compute_alpha_composite_score(before_metrics)
    a_sc = compute_alpha_composite_score(after_metrics)
    return ParamDeltaObservation(
        param_name=param.name,
        before_value=param.current,
        after_value=after_value,
        delta_param=after_value - param.current,
        delta_sharpe=float(after_metrics.get("sharpe") or 0.0) - float(before_metrics.get("sharpe") or 0.0),
        delta_fitness=float(after_metrics.get("fitness") or 0.0) - float(before_metrics.get("fitness") or 0.0),
        delta_turnover=float(after_metrics.get("turnover") or 0.0) - float(before_metrics.get("turnover") or 0.0),
        delta_margin=float(after_metrics.get("margin") or 0.0) - float(before_metrics.get("margin") or 0.0),
        improved_score=(a_sc > b_sc + 1e-4),
    )


def propose_multi_param_tuning_candidates(
    expression: str,
    settings: Mapping[str, Any],
    *,
    base_metrics: Mapping[str, Any],
    failed_checks: list[str] | None = None,
    prior_observations: list[ParamDeltaObservation] | None = None,
    max_candidates: int = 6,
) -> list[dict[str, Any]]:
    """Propose multi-parameter tuning candidates using Binary, Ternary, and Directional Coordinate Ascent.

    Works for BOTH:
      - Near-miss / failed alphas (repairing failing checks)
      - Already-PASSING alphas (polishing to find a stronger family champion)
    """
    params = extract_tunable_parameters(expression, settings)
    if not params:
        return []

    failed = [str(c).upper() for c in (failed_checks or []) if c]
    obs_by_param = {o.param_name: o for o in (prior_observations or [])}
    candidates: list[dict[str, Any]] = []
    seen_expr_sett: set[tuple[str, str]] = set()

    def _emit(param: TunableParam, val: float, method: SearchMethod, note: str) -> None:
        new_expr, new_sett = apply_parameter_value(expression, settings, param, val)
        sig = (new_expr, json.dumps(new_sett, sort_keys=True))
        if sig in seen_expr_sett or (new_expr == expression and new_sett == dict(settings)):
            return
        seen_expr_sett.add(sig)
        candidates.append({
            "name": f"Tune_{method.value}_{param.role}_{param.format_value(val)}",
            "expression": new_expr,
            "settings": new_sett,
            "tuned_param": param.name,
            "before_value": param.current,
            "after_value": int(round(val)) if param.is_int else round(float(val), 4),
            "search_method": method.value,
            "reason": note,
            "edit_class": "PARAMETER_VARIANT",
            "structural_novelty": False,
        })

    for p in params:
        if len(candidates) >= max_candidates:
            break
        prior = obs_by_param.get(p.name)

        # 1. If we already observed a delta for this parameter -> Directional Coordinate Ascent
        if prior is not None:
            direction = prior.recommended_direction
            if direction == "UPPER" and p.current < p.high:
                step_val = (p.current + p.high) / 2.0
                _emit(p, step_val, SearchMethod.COORDINATE_ASCENT, f"COORD_ASCENT_UP:{p.name}")
            elif direction == "LOWER" and p.current > p.low:
                step_val = (p.low + p.current) / 2.0
                _emit(p, step_val, SearchMethod.COORDINATE_ASCENT, f"COORD_ASCENT_DOWN:{p.name}")
            continue

        # 2. Monotonic failure constraints -> Binary Search
        if any("TURNOVER" in c for c in failed) and p.role in ("DECAY", "HUMP"):
            to = float(base_metrics.get("turnover") or 0.35)
            target = (p.current + p.high) / 2.0 if to >= 0.30 else (p.low + p.current) / 2.0
            _emit(p, target, SearchMethod.BINARY, f"BINARY_TURNOVER:{p.name}")
            continue

        if any("WEIGHT" in c or "CONCENTRATION" in c for c in failed) and p.role == "TRUNCATION":
            target = (p.low + p.current) / 2.0
            _emit(p, target, SearchMethod.BINARY, f"BINARY_WEIGHT:{p.name}")
            continue

        # 3. Unimodal exploration (or polishing an already-passing alpha) -> Ternary Search (m1, m2)
        m1, m2 = ternary_search_points(p.low, p.high, is_int=p.is_int)
        _emit(p, m1, SearchMethod.TERNARY, f"TERNARY_M1:{p.name}")
        if len(candidates) < max_candidates:
            _emit(p, m2, SearchMethod.TERNARY, f"TERNARY_M2:{p.name}")

    return candidates[:max_candidates]


@dataclass
class TunedAlphaRecord:
    alpha_id: str
    expression: str
    settings: dict[str, Any]
    structural_hash: str
    passed_official: bool
    composite_score: float
    sharpe: float
    fitness: float
    turnover: float
    disposition: PoolDisposition


class WinnerOnlyFamilyPool:
    """Absorbs all passing tuned variants into PASSED_DO_NOT_SUBMIT and promotes ONLY
    the single best variant per `structural_hash` family into SUBMITTABLE_BEST.
    """

    def __init__(self, max_polish_trials_per_family: int = 8) -> None:
        self.max_polish_trials_per_family = max(1, int(max_polish_trials_per_family))
        self.records: dict[str, TunedAlphaRecord] = {}
        self.family_trial_counts: dict[str, int] = {}

    def register_result(
        self,
        *,
        alpha_id: str,
        expression: str,
        settings: Mapping[str, Any],
        metrics: Mapping[str, Any],
        passed_official: bool,
    ) -> TunedAlphaRecord:
        sh = parse_fastexpr(expression).structural_hash
        self.family_trial_counts[sh] = self.family_trial_counts.get(sh, 0) + 1
        score = compute_alpha_composite_score(metrics)

        if not passed_official:
            rec = TunedAlphaRecord(
                alpha_id=alpha_id,
                expression=expression,
                settings=dict(settings),
                structural_hash=sh,
                passed_official=False,
                composite_score=score,
                sharpe=float(metrics.get("sharpe") or 0.0),
                fitness=float(metrics.get("fitness") or 0.0),
                turnover=float(metrics.get("turnover") or 0.0),
                disposition=PoolDisposition.FAILED,
            )
            self.records[alpha_id] = rec
            return rec

        # Find current family champion (if any)
        current_champ = next(
            (r for r in self.records.values() if r.structural_hash == sh and r.disposition == PoolDisposition.SUBMITTABLE_BEST),
            None,
        )
        if current_champ is None or score > current_champ.composite_score:
            if current_champ is not None:
                # Demote previous champion into PASSED_DO_NOT_SUBMIT so only 1 per family is submittable
                current_champ.disposition = PoolDisposition.PASSED_DO_NOT_SUBMIT
            disposition = PoolDisposition.SUBMITTABLE_BEST
        else:
            disposition = PoolDisposition.PASSED_DO_NOT_SUBMIT

        rec = TunedAlphaRecord(
            alpha_id=alpha_id,
            expression=expression,
            settings=dict(settings),
            structural_hash=sh,
            passed_official=True,
            composite_score=score,
            sharpe=float(metrics.get("sharpe") or 0.0),
            fitness=float(metrics.get("fitness") or 0.0),
            turnover=float(metrics.get("turnover") or 0.0),
            disposition=disposition,
        )
        self.records[alpha_id] = rec
        return rec

    def can_polish_family(self, expression: str) -> bool:
        sh = parse_fastexpr(expression).structural_hash
        return self.family_trial_counts.get(sh, 0) < self.max_polish_trials_per_family

    def submittable_best(self) -> list[TunedAlphaRecord]:
        return sorted(
            [r for r in self.records.values() if r.disposition == PoolDisposition.SUBMITTABLE_BEST],
            key=lambda r: r.composite_score,
            reverse=True,
        )

    def passed_do_not_submit(self) -> list[TunedAlphaRecord]:
        return [r for r in self.records.values() if r.disposition == PoolDisposition.PASSED_DO_NOT_SUBMIT]


def scan_rescuable_historical_failures(
    db_path: str | Path,
    *,
    min_abs_sharpe: float = 1.10,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Inspect historical `brain_jobs` for rescuable strong negative-Sharpe or near-miss alphas."""
    from worldquant_brain.alpha_miner import validate_candidate_expression

    path = Path(db_path)
    if not path.exists():
        return []

    conn = sqlite3.connect(str(path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        tried_hashes: set[str] = set()
        try:
            for r in conn.execute("SELECT expression FROM brain_tried_expressions").fetchall():
                tried_hashes.add(parse_fastexpr(str(r[0])).exact_hash)
        except Exception:
            pass

        rows = conn.execute(
            "SELECT job_id, alpha_id, expression, settings_json, metrics_json FROM brain_jobs WHERE status='COMPLETED' AND metrics_json IS NOT NULL ORDER BY rowid DESC"
        ).fetchall()
    finally:
        conn.close()

    rescues: list[dict[str, Any]] = []
    seen_struct: set[str] = set()
    for r in rows:
        expr = str(r["expression"] or "").strip()
        ok, _ = validate_candidate_expression(expr)
        if not ok:
            continue
        try:
            m = json.loads(r["metrics_json"])
            sett = json.loads(r["settings_json"]) if r["settings_json"] else {}
        except Exception:
            continue
        sh = m.get("sharpe")
        if sh is None or abs(float(sh)) < min_abs_sharpe:
            continue
        if m.get("checks_passed") is True and float(sh) >= 1.25:
            continue

        sh_f = float(sh)
        struct_h = parse_fastexpr(expr).structural_hash
        if struct_h in seen_struct:
            continue
        seen_struct.add(struct_h)

        if sh_f <= -min_abs_sharpe:
            flipped = f"-1 * ({expr})" if not expr.startswith("-1 *") else expr[4:].strip()
            if parse_fastexpr(flipped).exact_hash in tried_hashes:
                continue
            rescues.append({
                "source_alpha_id": r["alpha_id"],
                "original_sharpe": sh_f,
                "rescue_type": "SIGN_FLIP",
                "expression": flipped,
                "settings": sett,
                "simulation_only": True,
                "auto_submit": False,
                "submission_allowed": False,
                "operator_approved": False,
                "queue_stage": "LOCAL_REVIEW",
                "tuning_candidates": propose_multi_param_tuning_candidates(
                    flipped, sett, base_metrics=m, failed_checks=m.get("failed_checks") or [], max_candidates=2
                ),
            })
        else:
            cands = propose_multi_param_tuning_candidates(
                expr, sett, base_metrics=m, failed_checks=m.get("failed_checks") or [], max_candidates=3
            )
            cands = [c for c in cands if parse_fastexpr(c["expression"]).exact_hash not in tried_hashes]
            if cands:
                rescues.append({
                    "source_alpha_id": r["alpha_id"],
                    "original_sharpe": sh_f,
                    "rescue_type": "MULTI_PARAM_TUNE",
                    "expression": expr,
                    "settings": sett,
                    "simulation_only": True,
                    "auto_submit": False,
                    "submission_allowed": False,
                    "operator_approved": False,
                    "queue_stage": "LOCAL_REVIEW",
                    "tuning_candidates": cands,
                })
        if len(rescues) >= limit:
            break
    return rescues
