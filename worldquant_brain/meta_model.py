"""BRAIN_ALPHA_META_MODEL (Phase 8): leakage-free AST-family grouped meta-predictor.

Strictly prohibited from generating direct Crypto orders.
Enforces chronological split by completed_at and groups AST families into the same split.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from worldquant_brain.alpha_repository import compute_ast_family_hash

MODEL_DESIGNATION = "BRAIN_ALPHA_META_MODEL"


def extract_expression_features(record: Mapping[str, Any]) -> dict[str, Any]:
    expr = str(record.get("expression") or "").strip()
    ops = re.findall(r"([a-z_][a-z0-9_]*)\s*\(", expr.lower())
    depth = 0
    max_depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth = max(0, depth - 1)
    return {
        "ast_family_hash": str(record.get("ast_family_hash") or compute_ast_family_hash(expr)),
        "operator_count": len(ops),
        "ast_depth": max_depth,
        "expression_complexity": len(expr),
        "has_group_neutralize": int("group_neutralize" in ops),
        "has_decay": int("ts_decay_linear" in ops or "hump" in ops),
        "decay": float(record.get("decay") or 4.0),
        "truncation": float(record.get("truncation") or 0.08),
        "completed_at": str(record.get("completed_at") or record.get("created_at") or "1970-01-01T00:00:00Z"),
    }


def split_family_chronological(
    records: Iterable[Mapping[str, Any]],
    *,
    train_frac: float = 0.75,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split chronologically by completed_at while forcing same AST family into the same split."""
    rows = [dict(r) | extract_expression_features(r) for r in records]
    rows.sort(key=lambda r: r["completed_at"])
    if not rows:
        return [], []
    cutoff_idx = max(1, int(len(rows) * train_frac))
    train_families: set[str] = set()
    for r in rows[:cutoff_idx]:
        train_families.add(r["ast_family_hash"])

    train_split: list[dict[str, Any]] = []
    test_split: list[dict[str, Any]] = []
    for r in rows:
        if r["ast_family_hash"] in train_families:
            train_split.append(r)
        else:
            test_split.append(r)
    return train_split, test_split
