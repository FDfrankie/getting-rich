"""Bounded Single-Failure Parameter Binary Search for WorldQuant BRAIN Near-Misses.

Triggers ONLY when:
  - Exactly 1 check failed (len(failed_checks) == 1)
  - Candidate is genuinely near-miss (abs(sharpe) >= 1.10)
Performs bounded bracket midpoint probing (max 3-4 steps) on the parameter
most directly responsible for the single failing check, tagging outputs as
`PARAMETER_VARIANT` (never counted as Structural Novelty).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class BinarySearchBracket:
    param_name: str
    low: float
    high: float
    current: float
    is_int: bool = False

    def next_midpoints(self, last_direction: str = "BOTH") -> list[float | int]:
        """Return bounded midpoint candidates from [low, current] and/or [current, high]."""
        pts: list[float] = []
        if last_direction in ("LOWER", "BOTH") and self.current > self.low:
            pts.append((self.low + self.current) / 2.0)
        if last_direction in ("UPPER", "BOTH") and self.high > self.current:
            pts.append((self.current + self.high) / 2.0)

        out: list[float | int] = []
        seen: set[float | int] = set()
        for v in pts:
            val: float | int = int(round(v)) if self.is_int else round(v, 4)
            if val != (int(round(self.current)) if self.is_int else round(self.current, 4)) and val not in seen:
                seen.add(val)
                out.append(val)
        return out


def _replace_hump(expr: str, new_hump: float) -> str:
    if re.search(r"hump\s*=\s*\d+(?:\.\d+)?", expr):
        return re.sub(r"(hump\s*=\s*)\d+(?:\.\d+)?", rf"\g<1>{new_hump:.4f}".rstrip("0").rstrip("."), expr, count=1)
    return f"hump({expr}, hump={new_hump:.4f})".replace("0)", ")")


def _replace_first_window(expr: str, old_w: int, new_w: int) -> str:
    return re.sub(rf"(,\s*){old_w}(\s*[,)])", rf"\g<1>{new_w}\g<2>", expr, count=1)


def _extract_primary_window(expr: str) -> int | None:
    m = re.search(r"ts_[a-z_]+\([^()]+,\s*(\d+)\s*(?:,|\))", expr)
    if m:
        return int(m.group(1))
    return None


def generate_single_failure_binary_search(
    expression: str,
    base_settings: Mapping[str, Any],
    *,
    failed_checks: list[str],
    sharpe: float,
    turnover: float | None = None,
    max_candidates: int = 3,
) -> list[dict[str, Any]]:
    """Generate bounded binary-search midpoint variants when exactly ONE check failed."""
    failed = [str(c).upper() for c in failed_checks if c]
    if len(failed) != 1 or abs(float(sharpe)) < 1.10:
        return []

    check = failed[0]
    expr = str(expression or "").strip()
    if not expr:
        return []

    base_decay = int(base_settings.get("decay", 4) or 4)
    base_trunc = float(base_settings.get("truncation", 0.05) or 0.05)
    out: list[dict[str, Any]] = []

    if "TURNOVER" in check or "FITNESS" in check:
        to = float(turnover) if turnover is not None else 0.35
        if to >= 0.30 or "HIGH_TURNOVER" in check:
            # Binary search decay upward in [base_decay, 16] and hump in [0.008, 0.024]
            decay_br = BinarySearchBracket("decay", low=float(base_decay), high=16.0, current=float(base_decay), is_int=True)
            for mid_d in decay_br.next_midpoints("UPPER"):
                s = dict(base_settings)
                s["decay"] = int(mid_d)
                out.append({
                    "name": f"BinSearch_Decay_{mid_d}",
                    "expression": expr,
                    "settings": s,
                    "reason": f"BINARY_SEARCH_DECAY:{base_decay}->{mid_d}",
                    "edit_class": "PARAMETER_VARIANT",
                    "structural_novelty": False,
                })
            h_match = re.search(r"hump\s*=\s*(\d+(?:\.\d+)?)", expr)
            cur_h = float(h_match.group(1)) if h_match else 0.008
            h_br = BinarySearchBracket("hump", low=cur_h, high=0.024, current=cur_h, is_int=False)
            for mid_h in h_br.next_midpoints("UPPER"):
                out.append({
                    "name": f"BinSearch_Hump_{mid_h}",
                    "expression": _replace_hump(expr, float(mid_h)),
                    "settings": dict(base_settings),
                    "reason": f"BINARY_SEARCH_HUMP:{cur_h}->{mid_h}",
                    "edit_class": "PARAMETER_VARIANT",
                    "structural_novelty": False,
                })
        else:
            # LOW_TURNOVER: binary search decay downward in [0, base_decay]
            decay_br = BinarySearchBracket("decay", low=0.0, high=float(max(2, base_decay)), current=float(base_decay), is_int=True)
            for mid_d in decay_br.next_midpoints("LOWER"):
                s = dict(base_settings)
                s["decay"] = int(mid_d)
                out.append({
                    "name": f"BinSearch_DecayDown_{mid_d}",
                    "expression": expr,
                    "settings": s,
                    "reason": f"BINARY_SEARCH_DECAY_DOWN:{base_decay}->{mid_d}",
                    "edit_class": "PARAMETER_VARIANT",
                    "structural_novelty": False,
                })

    elif "WEIGHT" in check or "CONCENTRATION" in check:
        trunc_br = BinarySearchBracket("truncation", low=0.02, high=max(0.06, base_trunc), current=base_trunc, is_int=False)
        for mid_t in trunc_br.next_midpoints("LOWER"):
            s = dict(base_settings)
            s["truncation"] = float(mid_t)
            out.append({
                "name": f"BinSearch_Trunc_{mid_t}",
                "expression": expr,
                "settings": s,
                "reason": f"BINARY_SEARCH_TRUNCATION:{base_trunc}->{mid_t}",
                "edit_class": "PARAMETER_VARIANT",
                "structural_novelty": False,
            })

    elif "SHARPE" in check or "2Y_SHARPE" in check:
        w = _extract_primary_window(expr)
        if w is not None and w >= 4:
            w_br = BinarySearchBracket("window", low=float(max(3, w // 2)), high=float(min(120, w * 2)), current=float(w), is_int=True)
            for mid_w in w_br.next_midpoints("BOTH"):
                cand_expr = _replace_first_window(expr, w, int(mid_w))
                if cand_expr != expr:
                    out.append({
                        "name": f"BinSearch_Win_{w}_to_{mid_w}",
                        "expression": cand_expr,
                        "settings": dict(base_settings),
                        "reason": f"BINARY_SEARCH_WINDOW:{w}->{mid_w}",
                        "edit_class": "PARAMETER_VARIANT",
                        "structural_novelty": False,
                    })

    return out[:max(1, int(max_candidates))]
