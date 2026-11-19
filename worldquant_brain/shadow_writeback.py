"""Shadow-mode / Record-only writeback bridge for ParetoArchive & ForbiddenEditMotif.

All functions in this module are strictly non-blocking and degrade gracefully:
they never raise exceptions into callers and never override official BRAIN gates.
"""
from __future__ import annotations

import contextlib
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from worldquant_brain.canonical_ast import parse_fastexpr
from worldquant_brain.forbidden_edit_motif import (
    ForbiddenEditMotifShadowStore,
    MotifContext,
    build_edit_motif_from_expressions,
)
from worldquant_brain.pareto_archive import (
    ArchiveScope,
    ObjectiveLane,
    ObjectiveValue,
    ParetoArchive,
    ParetoArchiveShadowStore,
    ParetoCandidate,
    ParetoContext,
    ParetoObjectiveVector,
)

_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


def _ensure_shadow_tables(db_path: str | Path) -> None:
    path_str = str(db_path)
    if not path_str or path_str == ":memory:":
        return
    conn = sqlite3.connect(path_str, timeout=10.0)
    try:
        existing = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "research_edit_motifs" not in existing:
            m010 = _MIGRATIONS_DIR / "010_forbidden_edit_motifs.sql"
            if m010.exists():
                conn.executescript(m010.read_text(encoding="utf-8"))
        if "pareto_archive_entries" not in existing:
            m011 = _MIGRATIONS_DIR / "011_pareto_archive.sql"
            if m011.exists():
                conn.executescript(m011.read_text(encoding="utf-8"))
    finally:
        conn.close()


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def record_simulation_shadow(
    db_path: str | Path | None,
    *,
    expression: str,
    settings: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    alpha_id: str | None = None,
    parent_expression: str | None = None,
    dataset_id: str = "",
    hypothesis_family: str = "general",
) -> dict[str, Any]:
    """Record a simulation outcome into ParetoArchive & ForbiddenEditMotif in shadow mode.

    Never raises; returns status dict with `recorded_pareto`, `recorded_motif`, and `degraded`.
    """
    out: dict[str, Any] = {
        "recorded_pareto": False,
        "recorded_motif": False,
        "degraded": False,
    }
    if not db_path or str(db_path) == ":memory:" or not str(expression or "").strip():
        return out

    try:
        _ensure_shadow_tables(db_path)
        now = datetime.now(timezone.utc).isoformat()
        sett = dict(settings or {})
        met = dict(metrics or {})

        region = str(sett.get("region") or "USA").upper()
        delay = int(sett.get("delay") if sett.get("delay") is not None else 1)
        if delay not in (0, 1):
            delay = 1
        universe = str(sett.get("universe") or "TOP3000").upper()
        data_domain = str(dataset_id or sett.get("dataset_domain") or "price_volume")
        raw_lane = str(sett.get("objective_lane") or "BALANCED").upper()
        try:
            lane = ObjectiveLane(raw_lane)
        except ValueError:
            lane = ObjectiveLane.BALANCED

        parsed = parse_fastexpr(expression)
        expr_hash = parsed.exact_hash or _stable_hash(expression)

        sharpe = met.get("sharpe")
        fitness = met.get("fitness")
        turnover = met.get("turnover")
        margin = met.get("margin")
        checks_passed = bool(met.get("checks_passed", False))
        failed_checks = [str(c).upper() for c in (met.get("failed_checks") or []) if c]

        # Build ParetoObjectiveVector (shadow-only; missing metrics remain UNKNOWN)
        if sharpe is not None and fitness is not None:
            oq_score = max(0.0, min(1.0, (float(sharpe) / 3.0) * 0.6 + (float(fitness) / 2.0) * 0.4))
            oq = ObjectiveValue.known(oq_score, evidence_source="WORLDQUANT_BRAIN", calculated_at=now)
        else:
            oq = ObjectiveValue.unknown()

        if turnover is not None:
            to_val = float(turnover)
            ce_score = max(0.0, min(1.0, 1.0 - abs(to_val - 0.125) / 0.70))
            ce = ObjectiveValue.known(ce_score, evidence_source="WORLDQUANT_BRAIN", calculated_at=now)
        else:
            ce = ObjectiveValue.unknown()

        ctx = ParetoContext(
            region=region,
            delay=delay,
            universe=universe,
            data_domain=data_domain,
            blueprint_family=hypothesis_family or "general",
            objective_lane=lane,
        )
        unk = ObjectiveValue.unknown()
        vec = ParetoObjectiveVector(
            official_quality=oq,
            os_stability=unk,
            perturbation_robustness=unk,
            pnl_uniqueness=unk,
            merged_portfolio_utility=unk,
            cost_efficiency=ce,
            capacity=unk,
            information_gain=unk,
            correlation_risk=unk,
            drawdown_risk=unk,
            overfitting_risk=unk,
            data_quality_risk=unk,
            complexity_cost=unk,
            simulation_minutes=unk,
        )
        cand = ParetoCandidate(
            trial_id=f"trial_{expr_hash[:12]}",
            artifact_id=str(alpha_id or f"art_{expr_hash[:12]}"),
            candidate_id=f"cand_{expr_hash[:12]}",
            policy_version="shadow-v1",
            objective_schema_version="v1",
            normalization_version="ast-v1",
            context=ctx,
            objectives=vec,
            input_snapshot_hash=expr_hash,
            admitted_at=now,
        )
        arch = ParetoArchive(scope=ArchiveScope.COHORT, scope_key=ctx.cohort_key())
        arch.admit(cand)
        p_store = ParetoArchiveShadowStore(db_path, shadow_mode=True)
        written = p_store.persist_archive(arch, alpha_pool_hash=expr_hash)
        out["recorded_pareto"] = written > 0

        if parent_expression and str(parent_expression).strip() and str(parent_expression).strip() != str(expression).strip():
            motif = build_edit_motif_from_expressions(
                parent_expression,
                expression,
            )
            failure_gate = failed_checks[0] if failed_checks else ("PASSED" if checks_passed else "UNSPECIFIED")
            m_ctx = MotifContext(
                hypothesis_family=hypothesis_family or "general",
                data_domain=data_domain,
                failure_gate=failure_gate,
                region=region,
                universe=universe,
                delay=delay,
                objective_lane=lane,
                parent_turnover_band=None,
                parent_correlation_band="UNKNOWN",
                parent_coverage_band=None,
                parent_ast_signature=parse_fastexpr(parent_expression).structural_expression[:120],
                field_catalog_version="brain-fields-v1",
                operator_schema_version="brain-operators-v1",
            )
            m_store = ForbiddenEditMotifShadowStore(db_path, shadow_mode=True)
            m_store.record_motif_and_context(motif, m_ctx)
            out["recorded_motif"] = True

    except Exception as exc:
        out["degraded"] = True
        out["error"] = str(exc)

    return out
