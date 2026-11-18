"""Three-layer Alpha Novelty, Public Expression Denylist, and Conservative Self-Correlation Buffer."""
from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class AlphaFingerprint:
    exact_hash: str
    normalized_hash: str
    structural_hash: str
    fields: tuple[str, ...]
    operators: tuple[str, ...]
    family: str


@dataclass(frozen=True, slots=True)
class DuplicateDecision:
    blocked: bool
    reason: str
    similarity: float
    matched_alpha_id: str | None = None


@dataclass(frozen=True, slots=True)
class CorrelationPrecheckDecision:
    decision: str  # PRIORITY | ACCEPT_LOW_PRIORITY | QUARANTINE | REJECT_BEFORE_BRAIN
    allowed_for_brain: bool
    local_corr: float
    effective_threshold_note: str


# Seeded well-known public Alpha101 / textbook expressions (only for denylist & family learning, never re-emitted)
DEFAULT_PUBLIC_FORMULAS: tuple[dict[str, str], ...] = (
    {
        "source_title": "101 Formulaic Alphas #101",
        "source_url": "https://arxiv.org/abs/1601.00991",
        "license": "Academic Reference",
        "expression": "(close - open) / ((high - low) + 0.001)",
    },
    {
        "source_title": "101 Formulaic Alphas #12",
        "source_url": "https://arxiv.org/abs/1601.00991",
        "license": "Academic Reference",
        "expression": "sign(delta(volume, 1)) * (-1 * delta(close, 1))",
    },
    {
        "source_title": "101 Formulaic Alphas #6",
        "source_url": "https://arxiv.org/abs/1601.00991",
        "license": "Academic Reference",
        "expression": "-1 * correlation(open, volume, 10)",
    },
    {
        "source_title": "Basic Price Reversal",
        "source_url": "https://platform.worldquantbrain.com/",
        "license": "Public Example",
        "expression": "rank(ts_delta(close, 20))",
    },
)


def _compact(expr: str) -> str:
    return re.sub(r"\s+", "", (expr or "").strip().lower())


def _operators(expr: str) -> tuple[str, ...]:
    ops = re.findall(r"([a-z_][a-z0-9_]*)\s*\(", (expr or "").lower())
    return tuple(sorted(set(ops)))


def _fields(expr: str) -> tuple[str, ...]:
    tokens = set(re.findall(r"\b([a-z_][a-z0-9_]*)\b", (expr or "").lower()))
    ops = set(_operators(expr))
    keywords = {"std", "hump", "rettype", "driver", "mode", "true", "false", "industry", "subindustry", "sector", "market"}
    return tuple(sorted(tokens - ops - keywords))


def _normalize_numbers(expr: str) -> str:
    s = _compact(expr)
    return re.sub(r"\b\d+(?:\.\d+)?\b", "#", s)


def _strip_outer_decay(expr: str) -> str:
    s = _compact(expr)
    changed = True
    while changed:
        changed = False
        m = re.match(r"^ts_decay_linear\((.+),\d+\)$", s)
        if m:
            s = m.group(1)
            changed = True
    return re.sub(r"\b\d+(?:\.\d+)?\b", "#", s)


def _family_signature(fields: Iterable[str], operators: Iterable[str]) -> str:
    f_str = ",".join(sorted(set(fields)))
    o_str = ",".join(sorted(set(operators)))
    return f"F[{f_str}]|O[{o_str}]"


def fingerprint(expression: str) -> AlphaFingerprint:
    c = _compact(expression)
    norm = _normalize_numbers(expression)
    struct = _strip_outer_decay(expression)
    flds = _fields(expression)
    ops = _operators(expression)
    fam = _family_signature(flds, ops)
    return AlphaFingerprint(
        exact_hash=hashlib.sha256(c.encode("utf-8")).hexdigest(),
        normalized_hash=hashlib.sha256(norm.encode("utf-8")).hexdigest(),
        structural_hash=hashlib.sha256(struct.encode("utf-8")).hexdigest(),
        fields=flds,
        operators=ops,
        family=fam,
    )


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def near_duplicate_score(a: AlphaFingerprint, b: AlphaFingerprint) -> float:
    if a.exact_hash == b.exact_hash:
        return 1.0
    if a.normalized_hash == b.normalized_hash or a.structural_hash == b.structural_hash:
        return 0.96
    f_sim = jaccard(a.fields, b.fields)
    o_sim = jaccard(a.operators, b.operators)
    return round(0.65 * f_sim + 0.35 * o_sim, 4)


def find_duplicate(
    expression: str,
    existing: Iterable[Mapping[str, Any]],
    *,
    public_denylist: Iterable[Mapping[str, Any]] | None = None,
    threshold: float = 0.86,
) -> DuplicateDecision:
    fp = fingerprint(expression)
    for pub in (public_denylist or DEFAULT_PUBLIC_FORMULAS):
        pub_expr = str(pub.get("expression") or "")
        if not pub_expr:
            continue
        pfp = fingerprint(pub_expr)
        if fp.exact_hash == pfp.exact_hash or fp.normalized_hash == pfp.normalized_hash:
            return DuplicateDecision(True, "PUBLIC_EXPRESSION_DENYLIST", 1.0, str(pub.get("source_title") or "PUBLIC"))

    best_sim = 0.0
    best_id: str | None = None
    for item in existing:
        ex_expr = str(item.get("expression") or "")
        if not ex_expr:
            continue
        efp = fingerprint(ex_expr)
        aid = str(item.get("alpha_id") or item.get("id") or "")
        if fp.exact_hash == efp.exact_hash:
            return DuplicateDecision(True, "EXACT_EXPRESSION", 1.0, aid or None)
        if fp.normalized_hash == efp.normalized_hash or fp.structural_hash == efp.structural_hash:
            return DuplicateDecision(True, "PARAMETER_ONLY_VARIANT", 0.96, aid or None)
        sim = near_duplicate_score(fp, efp)
        if sim > best_sim:
            best_sim = sim
            best_id = aid or None
    if best_sim >= threshold:
        return DuplicateDecision(True, "SEMANTIC_NEAR_DUPLICATE", best_sim, best_id)
    return DuplicateDecision(False, "NOVEL", best_sim, best_id)


def evaluate_correlation_buffer(
    local_corr: float,
    *,
    shares_raw_fields: bool = False,
) -> CorrelationPrecheckDecision:
    """Conservative self-correlation buffer (< 0.45 PRIORITY, 0.45-0.50 ACCEPT_LOW, 0.50-0.60 QUARANTINE, >= 0.60 REJECT_BEFORE_BRAIN).
    Applies an extra 0.05 safety margin when sharing raw fields with an existing alpha.
    """
    c = abs(float(local_corr))
    margin = 0.05 if shares_raw_fields else 0.0
    eff = c + margin
    note = "shared_raw_fields_margin=+0.05" if shares_raw_fields else "standard_buffer"
    if eff >= 0.60:
        return CorrelationPrecheckDecision("REJECT_BEFORE_BRAIN", False, c, note)
    if eff >= 0.50:
        return CorrelationPrecheckDecision("QUARANTINE", True, c, note)
    if eff >= 0.45:
        return CorrelationPrecheckDecision("ACCEPT_LOW_PRIORITY", True, c, note)
    return CorrelationPrecheckDecision("PRIORITY", True, c, note)


def seed_public_denylist(conn: sqlite3.Connection) -> None:
    now = datetime.now(timezone.utc).isoformat()
    for pub in DEFAULT_PUBLIC_FORMULAS:
        fp = fingerprint(pub["expression"])
        conn.execute(
            """
            INSERT OR IGNORE INTO public_expression_denylist (
                exact_hash, normalized_hash, structural_hash, expression,
                source_title, source_url, license, fields_json, operators_json, family, imported_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fp.exact_hash,
                fp.normalized_hash,
                fp.structural_hash,
                pub["expression"],
                pub["source_title"],
                pub["source_url"],
                pub["license"],
                str(list(fp.fields)),
                str(list(fp.operators)),
                fp.family,
                now,
            ),
        )
