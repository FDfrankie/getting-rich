"""Unit tests for Shadow-mode Pareto Research Archive (worldquant_brain/pareto_archive.py)."""
from __future__ import annotations

import math
import sqlite3
import tempfile
import unittest
from pathlib import Path

from worldquant_brain.pareto_archive import (
    ArchiveCapacityPolicy,
    ArchiveScope,
    ArchiveStatus,
    ObjectiveLane,
    ObjectiveState,
    ObjectiveValue,
    ParetoArchive,
    ParetoArchiveShadowStore,
    ParetoCandidate,
    ParetoContext,
    ParetoObjectiveVector,
    dominates,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TS = "2026-11-20T00:00:00Z"


def _ov(val: float | None, state: ObjectiveState = ObjectiveState.KNOWN) -> ObjectiveValue:
    if state == ObjectiveState.UNKNOWN:
        return ObjectiveValue.unknown(evidence_source="NONE", calculated_at=TS)
    if state == ObjectiveState.NOT_APPLICABLE:
        return ObjectiveValue.not_applicable(evidence_source="NONE", calculated_at=TS)
    assert val is not None
    return ObjectiveValue.known(val, evidence_source="WORLDQUANT_BRAIN", calculated_at=TS)


def _make_vec(
    *,
    official_quality: float | None = 0.80,
    os_stability: float | None = 0.75,
    perturbation_robustness: float | None = 0.70,
    pnl_uniqueness: float | None = 0.80,
    merged_portfolio_utility: float | None = 0.65,
    cost_efficiency: float | None = 0.70,
    capacity: float | None = 0.65,
    information_gain: float | None = 0.60,
    correlation_risk: float | None = 0.20,
    drawdown_risk: float | None = 0.25,
    overfitting_risk: float | None = 0.20,
    data_quality_risk: float | None = 0.15,
    complexity_cost: float | None = 0.20,
    simulation_minutes: float | None = 0.20,
) -> ParetoObjectiveVector:
    def wrap(v: float | None) -> ObjectiveValue:
        return _ov(None, ObjectiveState.UNKNOWN) if v is None else _ov(v)

    return ParetoObjectiveVector(
        official_quality=wrap(official_quality),
        os_stability=wrap(os_stability),
        perturbation_robustness=wrap(perturbation_robustness),
        pnl_uniqueness=wrap(pnl_uniqueness),
        merged_portfolio_utility=wrap(merged_portfolio_utility),
        cost_efficiency=wrap(cost_efficiency),
        capacity=wrap(capacity),
        information_gain=wrap(information_gain),
        correlation_risk=wrap(correlation_risk),
        drawdown_risk=wrap(drawdown_risk),
        overfitting_risk=wrap(overfitting_risk),
        data_quality_risk=wrap(data_quality_risk),
        complexity_cost=wrap(complexity_cost),
        simulation_minutes=wrap(simulation_minutes),
    )


def _make_candidate(
    cid: str,
    vec: ParetoObjectiveVector,
    *,
    universe: str = "TOP3000",
    lane: ObjectiveLane = ObjectiveLane.BALANCED,
    domain: str = "FUNDAMENTAL",
) -> ParetoCandidate:
    return ParetoCandidate(
        trial_id=f"trial_{cid}",
        artifact_id=f"art_{cid}",
        candidate_id=cid,
        policy_version="shadow-v1",
        objective_schema_version="pareto-v1",
        normalization_version="norm-v1",
        context=ParetoContext(
            region="USA",
            delay=1,
            universe=universe,
            data_domain=domain,
            blueprint_family="CF_EV",
            objective_lane=lane,
        ),
        objectives=vec,
        input_snapshot_hash=f"snap_{cid}",
        admitted_at=TS,
    )


class TestParetoArchive(unittest.TestCase):
    def test_dominating_candidate_enters_first_front(self) -> None:
        arch = ParetoArchive(scope=ArchiveScope.COHORT, scope_key="USA:D1:TOP3000")
        c_strong = _make_candidate("strong", _make_vec(official_quality=0.90, capacity=0.80, drawdown_risk=0.10))
        c_weak = _make_candidate("weak", _make_vec(official_quality=0.60, capacity=0.50, drawdown_risk=0.40))
        arch.admit(c_strong)
        arch.admit(c_weak)
        front1 = arch.first_front()
        self.assertEqual([e.candidate.candidate_id for e in front1], ["strong"])
        self.assertTrue(dominates(c_strong, c_weak))

    def test_tradeoff_candidates_both_survive(self) -> None:
        arch = ParetoArchive(scope=ArchiveScope.COHORT, scope_key="USA:D1:TOP3000")
        c_a = _make_candidate("high_sharpe", _make_vec(official_quality=0.95, capacity=0.40))
        c_b = _make_candidate("high_cap", _make_vec(official_quality=0.70, capacity=0.95))
        arch.admit(c_a)
        arch.admit(c_b)
        front1_ids = {e.candidate.candidate_id for e in arch.first_front()}
        self.assertEqual(front1_ids, {"high_sharpe", "high_cap"})

    def test_unknown_objective_does_not_become_best(self) -> None:
        c_known = _make_candidate("known", _make_vec(official_quality=0.80, os_stability=0.80))
        c_unknown = _make_candidate("unknown", _make_vec(official_quality=0.80, os_stability=None))
        self.assertFalse(dominates(c_unknown, c_known))

    def test_missing_required_objective_is_not_promotion_ready(self) -> None:
        vec = _make_vec(official_quality=None)
        self.assertFalse(vec.promotion_ready())

    def test_archive_admission_is_idempotent(self) -> None:
        arch = ParetoArchive(scope=ArchiveScope.COHORT, scope_key="USA:D1:TOP3000")
        c = _make_candidate("dup", _make_vec())
        e1 = arch.admit(c)
        e2 = arch.admit(c)
        self.assertEqual(e1.archive_entry_id, e2.archive_entry_id)
        self.assertEqual(len(arch.entries), 1)

    def test_same_key_different_objective_vector_conflicts(self) -> None:
        arch = ParetoArchive(scope=ArchiveScope.COHORT, scope_key="USA:D1:TOP3000")
        c1 = _make_candidate("conflict", _make_vec(official_quality=0.80))
        c2 = _make_candidate("conflict", _make_vec(official_quality=0.95))
        arch.admit(c1)
        with self.assertRaises(ValueError):
            arch.admit(c2)

    def test_capacity_eviction_preserves_extremes(self) -> None:
        policy = ArchiveCapacityPolicy(global_capacity=10, objective_capacity=10, cohort_capacity=2, family_capacity=10)
        arch = ParetoArchive(scope=ArchiveScope.COHORT, scope_key="USA:D1:TOP3000", capacity_policy=policy)
        arch.admit(_make_candidate("ext_a", _make_vec(official_quality=0.99, capacity=0.20)))
        arch.admit(_make_candidate("mid", _make_vec(official_quality=0.60, capacity=0.60)))
        arch.admit(_make_candidate("ext_b", _make_vec(official_quality=0.20, capacity=0.99)))
        active_ids = {e.candidate.candidate_id for e in arch.active_entries()}
        self.assertEqual(active_ids, {"ext_a", "ext_b"})

    def test_crowding_distance_preserves_diversity(self) -> None:
        arch = ParetoArchive(scope=ArchiveScope.COHORT, scope_key="USA:D1:TOP3000")
        arch.admit(_make_candidate("e1", _make_vec(official_quality=0.95, capacity=0.10)))
        arch.admit(_make_candidate("e2", _make_vec(official_quality=0.50, capacity=0.50)))
        arch.admit(_make_candidate("e3", _make_vec(official_quality=0.10, capacity=0.95)))
        front1 = arch.first_front()
        by_id = {e.candidate.candidate_id: e.crowding_distance for e in front1}
        self.assertTrue(math.isinf(by_id["e1"]))
        self.assertTrue(math.isinf(by_id["e3"]))
        self.assertGreater(by_id["e2"], 0.0)

    def test_different_cohorts_do_not_dominate_each_other(self) -> None:
        c_3000 = _make_candidate("c3000", _make_vec(official_quality=0.99, capacity=0.99), universe="TOP3000")
        c_1000 = _make_candidate("c1000", _make_vec(official_quality=0.20, capacity=0.20), universe="TOP1000")
        self.assertFalse(dominates(c_3000, c_1000, require_same_cohort=True))

    def test_non_finite_objective_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ObjectiveValue.known(float("nan"), evidence_source="WORLDQUANT_BRAIN", calculated_at=TS)

    def test_shadow_sqlite_persistence_roundtrip(self) -> None:
        m011 = (REPO_ROOT / "migrations" / "011_pareto_archive.sql").read_text(encoding="utf-8")
        arch = ParetoArchive(scope=ArchiveScope.COHORT, scope_key="USA:D1:TOP3000")
        arch.admit(_make_candidate("s1", _make_vec(official_quality=0.90, capacity=0.40)))
        arch.admit(_make_candidate("s2", _make_vec(official_quality=0.50, capacity=0.90)))
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db_path = Path(td) / "test.db"
            conn = sqlite3.connect(db_path)
            conn.executescript(m011)
            conn.close()
            store = ParetoArchiveShadowStore(db_path, shadow_mode=True)
            written = store.persist_archive(arch)
            self.assertEqual(written, 2)
            conn = sqlite3.connect(db_path)
            cnt = conn.execute("SELECT COUNT(*) FROM pareto_archive_entries").fetchone()[0]
            snaps = conn.execute("SELECT COUNT(*) FROM pareto_archive_snapshots").fetchone()[0]
            conn.close()
            self.assertEqual(cnt, 2)
            self.assertEqual(snaps, 1)


if __name__ == "__main__":
    unittest.main()
