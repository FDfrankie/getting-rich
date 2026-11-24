"""P42 (KB v78 Ch.71 Patrol R1-R3): Pre-simulation FastExpr AST Linter (FE01-FE15).

Zero-quota deterministic static gate executed before submitting FastExpr formulas to
WorldQuant BRAIN. Catches syntax pitfalls, Bronze-locked Genius operators, un-densified
pv13_* groupings, bare vector fields, invalid windows, raw placeholders, and high-turnover
un-decayed delta probes.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from worldquant_brain.multi_sim_client import find_locked_bronze_operators, is_consultant_mode

_BARE_VEC_FIELD_RE = re.compile(r"(?<!vec_avg\()(?<!vec_sum\()(?<!vec_max\()(?<!vec_min\()\b(vec_[a-zA-Z0-9_]+)\b")
_FAST_DELTA_RE = re.compile(r"\bts_delta\s*\([^()]*,\s*([1-5])\s*\)", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(
    r"(\{[a-zA-Z0-9_]+\}|\bPLACEHOLDER\b|\bTODO\b|\bTBD\b|\bP05_FIELD\b|\bALT_0[78]\b|\bFIELD_NAME\b)",
    re.IGNORECASE,
)
_GENIUS_ONLY_OPS: frozenset[str] = frozenset({
    "group_cartesian_product",
    "group_extra",
    "group_max",
    "group_min",
    "ts_kurtosis",
    "ts_skewness",
    "ts_theilsen",
    "ts_vector_neut",
    "ts_vector_proj",
})
_VALID_NEUTRALIZATIONS: frozenset[str] = frozenset({
    "NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY",
    "CROWDING", "FAST", "SLOW", "SLOW_AND_FAST", "STATISTICAL", "COUNTRY",
})
_TS_WINDOW_OPS: tuple[str, ...] = (
    "ts_rank", "ts_scale", "ts_zscore", "ts_delta", "ts_mean", "ts_std_dev",
    "ts_av_diff", "ts_decay_linear", "ts_sum", "ts_min", "ts_max", "ts_ir",
    "ts_corr", "ts_regression",
)
_GROUP_OPS: tuple[str, ...] = (
    "group_neutralize", "group_rank", "group_zscore", "group_scale", "group_mean", "group_backfill",
)


def _extract_func_calls(expr: str, func_name: str) -> list[str]:
    """Extract top-level argument strings inside balanced func_name(...) calls."""
    out: list[str] = []
    pat = re.compile(rf"\b{re.escape(func_name)}\s*\(", re.IGNORECASE)
    for m in pat.finditer(expr):
        i = m.end()
        depth = 1
        while i < len(expr) and depth > 0:
            if expr[i] == "(":
                depth += 1
            elif expr[i] == ")":
                depth -= 1
            i += 1
        if depth == 0:
            out.append(expr[m.end():i - 1])
    return out


def _split_top_args(arg_str: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in arg_str:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def lint_fastexpr(
    expression: str,
    settings: Mapping[str, Any] | None = None,
    *,
    consultant_mode: bool | None = None,
) -> dict[str, Any]:
    """Run FE01-FE15 static checks on a FastExpr formula and simulation settings."""
    expr = str(expression or "").strip()
    sett = dict(settings or {})
    is_cons = is_consultant_mode() if consultant_mode is None else bool(consultant_mode)
    errors: list[str] = []
    warnings: list[str] = []
    rules: list[str] = []

    if not expr:
        return {"ok": False, "errors": ["FE15: empty expression"], "warnings": [], "rules_triggered": ["FE15"]}

    if expr.count("(") != expr.count(")"):
        errors.append("FE10: unbalanced parentheses")
        rules.append("FE10")

    # FE15: Zero raw placeholder tokens
    ph = _PLACEHOLDER_RE.findall(expr)
    if ph:
        errors.append(f"FE15: unresolved placeholder token(s): {sorted(set(ph))}")
        rules.append("FE15")

    # FE01: winsorize() MUST use named parameter std=...
    for arg_str in _extract_func_calls(expr, "winsorize"):
        args = _split_top_args(arg_str)
        if len(args) >= 2 and not any(a.replace(" ", "").lower().startswith("std=") for a in args[1:]):
            errors.append(f"FE01: winsorize() must use named std= parameter (got: winsorize({arg_str}))")
            rules.append("FE01")

    # FE02: group_mean(x, weight, group) MUST pass 3 arguments
    for arg_str in _extract_func_calls(expr, "group_mean"):
        args = _split_top_args(arg_str)
        if len(args) != 3:
            errors.append(f"FE02: group_mean() requires 3 arguments (x, weight, group), got {len(args)}")
            rules.append("FE02")

    # FE03: pv13_* relation grouping fields MUST be wrapped with densify(...)
    for gop in _GROUP_OPS:
        for arg_str in _extract_func_calls(expr, gop):
            args = _split_top_args(arg_str)
            if not args:
                continue
            grp_arg = args[2] if gop == "group_mean" and len(args) >= 3 else (args[1] if len(args) >= 2 else "")
            if re.fullmatch(r"pv13_[a-zA-Z0-9_]+", grp_arg.strip(), re.IGNORECASE):
                errors.append(f"FE03: relation grouping field '{grp_arg.strip()}' must be wrapped with densify({grp_arg.strip()})")
                rules.append("FE03")

    # FE04: Bare vec_* array fields must be reduced via vec_avg()/vec_sum()
    for m in _BARE_VEC_FIELD_RE.finditer(expr):
        tok = m.group(1)
        if tok.lower() not in {"vec_avg", "vec_sum", "vec_max", "vec_min", "vec_stddev", "vec_count", "vec_ir"}:
            errors.append(f"FE04: vector field '{tok}' must be reduced via vec_avg()/vec_sum()")
            rules.append("FE04")

    # FE05: Literal division by zero guard
    if re.search(r"/\s*0(?:\.0+)?(?:\s|$|\))", expr):
        errors.append("FE05: literal division by zero")
        rules.append("FE05")

    # FE08 / FE09: Bronze Base 66-op compatibility & Genius-only operator flag
    if not is_cons:
        locked = find_locked_bronze_operators(expr)
        if locked:
            genius_hits = [op for op in locked if op.lower() in _GENIUS_ONLY_OPS]
            if genius_hits:
                errors.append(f"FE09: Genius-only operator(s) blocked in Bronze/Silver mode: {genius_hits}")
                rules.append("FE09")
            other_locked = [op for op in locked if op.lower() not in _GENIUS_ONLY_OPS]
            if other_locked:
                errors.append(f"FE08: locked non-Bronze operator(s): {other_locked}")
                rules.append("FE08")

    # FE10: Positive integer time-series windows
    for op_name in _TS_WINDOW_OPS:
        win_idx = 2 if op_name in {"ts_corr", "ts_regression"} else 1
        for arg_str in _extract_func_calls(expr, op_name):
            args = _split_top_args(arg_str)
            if len(args) > win_idx:
                raw_w = args[win_idx].strip()
                if re.fullmatch(r"-?\d+(?:\.\d+)?", raw_w):
                    val = float(raw_w)
                    if val <= 0 or val != int(val):
                        errors.append(f"FE10: {op_name}() window must be a positive integer (got {raw_w})")
                        rules.append("FE10")

    # FE11: Neutralization & truncation bounds
    if sett:
        neut = str(sett.get("neutralization") or "").strip().upper()
        if neut and neut not in _VALID_NEUTRALIZATIONS:
            errors.append(f"FE11: invalid neutralization '{neut}'")
            rules.append("FE11")
        trunc = sett.get("truncation")
        if trunc is not None:
            try:
                if float(trunc) > 0.10 or float(trunc) < 0.0:
                    warnings.append(f"FE11: truncation={trunc} outside recommended [0.0, 0.10]")
                    rules.append("FE11")
            except (TypeError, ValueError):
                errors.append(f"FE11: non-numeric truncation '{trunc}'")
                rules.append("FE11")

    # FE12: Turnover control warning for un-decayed short-window ts_delta(..., 1..5)
    if _FAST_DELTA_RE.search(expr) and "ts_decay_linear" not in expr.lower() and "hump(" not in expr.lower():
        decay_val = int(sett.get("decay", 0) or 0) if sett else 0
        if decay_val < 6:
            warnings.append(f"FE12: short-window ts_delta with decay={decay_val}<6 and no ts_decay_linear/hump risks >90% turnover")
            rules.append("FE12")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "rules_triggered": sorted(set(rules)),
    }
