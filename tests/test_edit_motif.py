"""Unit tests for Forbidden Edit Motif schema, hashing, AST bridge, and gate-specific outcomes."""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from worldquant_brain.forbidden_edit_motif import (
    EditMotif,
    EditMotifType,
    ForbiddenEditMotifShadowStore,
    MotifContext,
    VetoLevel,
    build_edit_motif_from_expressions,
    determine_veto_level,
    motif_repair_succeeded,
)
from worldquant_brain.pareto_archive import ObjectiveLane

REPO_ROOT = Path(__file__).resolve().parent.parent


def _buzz_motif() -> EditMotif:
    return EditMotif(
        motif_type=EditMotifType.ADD_RANK,
        target_ast_path="$.children[0].children[0].children[0]",
        before_subtree_hash="hash:scl12_buzz",
        after_subtree_hash="hash:rank(scl12_buzz)",
        added_operators=("rank",),
        removed_operators=(),
        changed_fields=(),
        changed_literals=(),
        setting_changes={},
        structural_change=True,
        parameter_only=False,
        ast_schema_version="fastexpr-ast-v1",
    )


def _buzz_context(universe: str = "TOP3000", failure_gate: str = "SUB_UNIVERSE_FAILURE") -> MotifContext:
    return MotifContext(
        hypothesis_family="SOCIAL_BUZZ_VOLATILITY",
        data_domain="SENTIMENT",
        failure_gate=failure_gate,
        region="USA",
        universe=universe,
        delay=1,
        objective_lane=ObjectiveLane.BALANCED,
        parent_turnover_band="PREFERRED",
        parent_correlation_band="UNKNOWN",
        parent_coverage_band="MEDIUM",
        parent_ast_signature="group_neutralize(neg(ts_std_dev(ts_backfill(FIELD,WINDOW),WINDOW)),GROUP)",
        field_catalog_version="brain-fields-v1",
        operator_schema_version="brain-operators-v1",
    )


class TestEditMotif(unittest.TestCase):
    def test_motif_hash_is_stable(self) -> None:
        m1 = _buzz_motif()
        m2 = _buzz_motif()
        self.assertEqual(m1.motif_hash(), m2.motif_hash())
        self.assertEqual(m1.motif_id(), m2.motif_id())

    def test_parameter_only_and_structural_cannot_both_be_true(self) -> None:
        bad = EditMotif(
            motif_type=EditMotifType.ADD_RANK,
            target_ast_path="$",
            before_subtree_hash="a",
            after_subtree_hash="b",
            added_operators=("rank",),
            removed_operators=(),
            changed_fields=(),
            changed_literals=(),
            setting_changes={},
            structural_change=True,
            parameter_only=True,
            ast_schema_version="fastexpr-ast-v1",
        )
        with self.assertRaises(ValueError):
            bad.validate()

    def test_context_hash_changes_across_universe(self) -> None:
        c1 = _buzz_context(universe="TOP3000")
        c2 = _buzz_context(universe="TOP1000")
        self.assertNotEqual(c1.context_hash(), c2.context_hash())

    def test_context_hash_changes_across_failure_gate(self) -> None:
        c1 = _buzz_context(failure_gate="SUB_UNIVERSE_FAILURE")
        c2 = _buzz_context(failure_gate="HIGH_TURNOVER")
        self.assertNotEqual(c1.context_hash(), c2.context_hash())

    def test_rank_failure_is_not_global_rank_veto(self) -> None:
        m = _buzz_motif()
        c_sentiment = _buzz_context(universe="TOP3000", failure_gate="SUB_UNIVERSE_FAILURE")
        c_fundamental = MotifContext(
            hypothesis_family="CF_EV",
            data_domain="FUNDAMENTAL",
            failure_gate="LOW_SHARPE",
            region="USA",
            universe="TOP3000",
            delay=1,
            objective_lane=ObjectiveLane.BALANCED,
            parent_turnover_band="PREFERRED",
            parent_correlation_band="LOW",
            parent_coverage_band="HIGH",
            parent_ast_signature="group_neutralize(FIELD,GROUP)",
            field_catalog_version="brain-fields-v1",
            operator_schema_version="brain-operators-v1",
        )
        self.assertNotEqual(c_sentiment.motif_context_id(m), c_fundamental.motif_context_id(m))

    def test_permanent_veto_only_for_allowlisted_reason(self) -> None:
        lvl, mult, exp = determine_veto_level(
            attempts=1,
            failures=1,
            distinct_parent_count=1,
            distinct_batch_count=1,
            permanent_invalid_reason="LOOK_AHEAD",
        )
        self.assertEqual(lvl, VetoLevel.PERMANENT)
        self.assertEqual(mult, 0.0)
        self.assertEqual(exp, 0.0)

        lvl_non, _, _ = determine_veto_level(
            attempts=1,
            failures=1,
            distinct_parent_count=1,
            distinct_batch_count=1,
            permanent_invalid_reason="ARBITRARY_REASON",
        )
        self.assertNotEqual(lvl_non, VetoLevel.PERMANENT)

    def test_sub_universe_success_is_gate_specific(self) -> None:
        # Exact Buzz failure case from empirical evidence
        buzz_ok = motif_repair_succeeded(
            failure_gate="SUB_UNIVERSE_FAILURE",
            before_sharpe=1.74,
            after_sharpe=0.03,
            before_fitness=1.36,
            after_fitness=0.0,
            before_turnover=0.2744,
            after_turnover=None,
            before_correlation=None,
            after_correlation=None,
            before_sub_universe_sharpe=0.53,
            after_sub_universe_sharpe=-0.36,
            after_sub_universe_passed=False,
            before_margin=None,
            after_margin=None,
        )
        self.assertFalse(buzz_ok)

    def test_turnover_success_requires_margin_retention(self) -> None:
        # Turnover dropped, but margin collapsed -> must NOT be counted as success
        collapsed = motif_repair_succeeded(
            failure_gate="HIGH_TURNOVER",
            before_sharpe=1.60,
            after_sharpe=1.55,
            before_fitness=1.20,
            after_fitness=1.18,
            before_turnover=0.65,
            after_turnover=0.30,
            before_correlation=0.30,
            after_correlation=0.30,
            before_sub_universe_sharpe=1.0,
            after_sub_universe_sharpe=1.0,
            after_sub_universe_passed=True,
            before_margin=0.0020,
            after_margin=0.0005,
        )
        self.assertFalse(collapsed)

    def test_ast_bridge_and_shadow_sqlite_persistence(self) -> None:
        m = build_edit_motif_from_expressions(
            "group_neutralize(-1 * ts_std_dev(ts_backfill(scl12_buzz, 5), 7), subindustry)",
            "group_neutralize(-1 * ts_std_dev(ts_backfill(rank(scl12_buzz), 5), 7), subindustry)",
        )
        self.assertEqual(m.motif_type, EditMotifType.ADD_RANK)
        c = _buzz_context()
        m010 = (REPO_ROOT / "migrations" / "010_forbidden_edit_motifs.sql").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            db_path = Path(td) / "test.db"
            conn = sqlite3.connect(db_path)
            conn.executescript(m010)
            conn.close()
            store = ForbiddenEditMotifShadowStore(db_path, shadow_mode=True)
            mid, cid = store.record_motif_and_context(m, c)
            self.assertTrue(mid)
            self.assertTrue(cid)
            # Idempotent 2nd write
            mid2, cid2 = store.record_motif_and_context(m, c)
            self.assertEqual((mid, cid), (mid2, cid2))


if __name__ == "__main__":
    unittest.main()
