"""Tri-State Data Leakage & Future-Tampering Audit Gate.

Implements the 'Future-Tampering Experiment' and Tri-State verification
(PASS / FAIL / UNVERIFIED) so unverified strategies or factors are NEVER
treated as PASS, and subtle look-ahead leakage in preprocessing or signals
is caught automatically before live/BRAIN promotion.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable

import numpy as np
import pandas as pd


class VerificationState(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNVERIFIED = "UNVERIFIED"


@dataclass(frozen=True)
class LeakageAuditReport:
    state: VerificationState
    cutoff_index: int
    max_abs_diff_before_cutoff: float | None
    changed_rows: int
    reason: str


def run_future_tampering_audit(
    df: pd.DataFrame,
    signal_fn: Callable[[pd.DataFrame], pd.Series | np.ndarray],
    *,
    cutoff_ratio: float = 0.70,
    shock_multiplier: float = 3.0,
    atol: float = 1e-10,
) -> LeakageAuditReport:
    """Perturb all rows strictly after `cutoff` and verify past outputs [0:cutoff] remain identical."""
    n = len(df)
    if n < 20:
        return LeakageAuditReport(
            state=VerificationState.UNVERIFIED,
            cutoff_index=0,
            max_abs_diff_before_cutoff=None,
            changed_rows=0,
            reason=f"INSUFFICIENT_BARS(n={n}<20)",
        )

    cutoff = max(10, min(n - 5, int(n * cutoff_ratio)))
    df_tampered = df.copy()
    for col in ("Open", "High", "Low", "Close", "Volume", "open", "high", "low", "close", "volume"):
        if col in df_tampered.columns:
            df_tampered.loc[df_tampered.index[cutoff:], col] = (
                pd.to_numeric(df_tampered.loc[df_tampered.index[cutoff:], col], errors="coerce") * shock_multiplier
            )

    try:
        base_out = np.asarray(signal_fn(df), dtype=np.float64)
        tamp_out = np.asarray(signal_fn(df_tampered), dtype=np.float64)
    except Exception as exc:
        return LeakageAuditReport(
            state=VerificationState.UNVERIFIED,
            cutoff_index=cutoff,
            max_abs_diff_before_cutoff=None,
            changed_rows=0,
            reason=f"AUDIT_EXECUTION_ERROR:{exc}",
        )

    if len(base_out) != n or len(tamp_out) != n:
        return LeakageAuditReport(
            state=VerificationState.UNVERIFIED,
            cutoff_index=cutoff,
            max_abs_diff_before_cutoff=None,
            changed_rows=0,
            reason="OUTPUT_LENGTH_MISMATCH",
        )

    b_pre = np.nan_to_num(base_out[:cutoff], nan=0.0)
    t_pre = np.nan_to_num(tamp_out[:cutoff], nan=0.0)
    diffs = np.abs(b_pre - t_pre)
    max_diff = float(np.max(diffs)) if len(diffs) > 0 else 0.0
    changed = int(np.sum(diffs > atol))

    if changed > 0:
        return LeakageAuditReport(
            state=VerificationState.FAIL,
            cutoff_index=cutoff,
            max_abs_diff_before_cutoff=max_diff,
            changed_rows=changed,
            reason=f"FUTURE_LEAKAGE_DETECTED(changed_rows={changed}, max_diff={max_diff:.6g})",
        )

    return LeakageAuditReport(
        state=VerificationState.PASS,
        cutoff_index=cutoff,
        max_abs_diff_before_cutoff=max_diff,
        changed_rows=0,
        reason="STRICT_CAUSALITY_VERIFIED",
    )
