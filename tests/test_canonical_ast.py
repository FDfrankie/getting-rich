"""Phase 0 Convergence & Migration Tests: Canonical AST, Miner Convergence, Sub-universe Classifier, and 010/011 SQL Migrations."""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from worldquant_brain.alpha_experiment_store import expression_hash, normalize_expression
from worldquant_brain.alpha_miner import ContinuousAlphaMiner, extract_expression_skeleton
from worldquant_brain.alpha_novelty import fingerprint
from worldquant_brain.alpha_repository import compute_ast_family_hash
from worldquant_brain.canonical_ast import diff_ast_edit, parse_fastexpr
from worldquant_brain.sub_universe_classifier import SubUniverseRootCause, classify_sub_universe_failure
from worldquant_brain.two_stage_funnel import _structural_skeleton


REPO_ROOT = Path(__file__).resolve().parent.parent


class TestCanonicalASTConvergence(unittest.TestCase):
    def test_single_ast_source_across_all_modules(self) -> None:
        expr_a = "group_neutralize( -1 * ts_std_dev( ts_backfill(scl12_buzz, 5), 7 ), subindustry )"
        expr_b = "group_neutralize(-1*ts_std_dev(ts_backfill(scl12_buzz, 10), 20), subindustry)"

        ast_a = parse_fastexpr(expr_a)
        ast_b = parse_fastexpr(expr_b)

        self.assertTrue(ast_a.parse_ok)
        self.assertTrue(ast_b.parse_ok)
        self.assertEqual(ast_a.normalized_hash, ast_b.normalized_hash)
        self.assertEqual(ast_a.ast_family_hash, ast_b.ast_family_hash)
        self.assertEqual(ast_a.ast_family_hash, compute_ast_family_hash(expr_a))
        self.assertEqual(compute_ast_family_hash(expr_a), compute_ast_family_hash(expr_b))

        fp_a = fingerprint(expr_a)
        fp_b = fingerprint(expr_b)
        self.assertEqual(fp_a.exact_hash, expression_hash(expr_a))
        self.assertEqual(fp_a.normalized_hash, fp_b.normalized_hash)
        self.assertEqual(extract_expression_skeleton(expr_a), extract_expression_skeleton(expr_b))
        self.assertEqual(_structural_skeleton(expr_a), _structural_skeleton(expr_b))
        self.assertEqual(normalize_expression(expr_a), ast_a.canonical_expression)

    def test_ast_diff_detects_buzz_add_rank_motif(self) -> None:
        parent = "group_neutralize(-1 * ts_std_dev(ts_backfill(scl12_buzz, 5), 7), subindustry)"
        child = "group_neutralize(-1 * ts_std_dev(ts_backfill(rank(scl12_buzz), 5), 7), subindustry)"
        diff = diff_ast_edit(parent, child)
        self.assertEqual(diff["motif_type"], "ADD_RANK")
        self.assertTrue(diff["structural_change"])
        self.assertFalse(diff["parameter_only"])
        self.assertEqual(diff["added_operators"], ("rank",))

    def test_ast_diff_marks_prime_window_as_parameter_only(self) -> None:
        parent = "ts_mean(close, 20)"
        child = "ts_mean(close, 23)"
        diff = diff_ast_edit(parent, child)
        self.assertTrue(diff["parameter_only"])
        self.assertFalse(diff["structural_change"])
        self.assertEqual(diff["motif_type"], "WINDOW_RESIZE")

    def test_miner_settings_bounded_exploration_and_audit(self) -> None:
        miner = ContinuousAlphaMiner()
        miner.has_tried_expression = lambda _e: False
        miner.mark_expression_tried = lambda _e: None
        s_det = miner.generate_candidate_settings(domain="FUNDAMENTAL_CF_EV", exploration_probability=0.0, random_seed=42)
        self.assertEqual(s_det["decay"], 2)
        self.assertFalse(s_det["_selection_audit"]["used_exploration"])
        self.assertEqual(s_det["_selection_audit"]["why_exploration_was_used"], "evidence_weighted_domain_default")

        explore_hits = sum(
            1
            for seed in range(200)
            if miner.generate_candidate_settings(domain="PV_REVERSAL", exploration_probability=0.10, random_seed=seed)["_selection_audit"]["used_exploration"]
        )
        self.assertLessEqual(explore_hits, 35)  # ~10% exploration bounded well below 20%

    def test_sub_universe_classifier_and_miner_avoid_blind_rank_on_sentiment(self) -> None:
        diag = classify_sub_universe_failure(
            main_sharpe=1.74,
            sub_universe_sharpe=0.53,
            universe="TOP3000",
            dataset_domain="SENTIMENT",
        )
        self.assertEqual(diag.root_cause, SubUniverseRootCause.DATASET_UNIVERSE_MISMATCH)
        self.assertIn("ADD_RANK", diag.disallowed_blind_edits)

        miner = ContinuousAlphaMiner()
        miner.has_tried_expression = lambda _e: False
        miner.mark_expression_tried = lambda _e: None
        refinements = miner.generate_near_miss_refinements(
            "group_neutralize(-1 * ts_std_dev(ts_backfill(scl12_buzz, 5), 7), subindustry)",
            {"universe": "TOP3000", "neutralization": "SUBINDUSTRY", "decay": 4, "truncation": 0.05, "sub_universe_sharpe": 0.53, "dataset_domain": "SENTIMENT"},
            sharpe=1.74,
            fitness=1.36,
            turnover=0.2744,
            failed_checks=["SUB_UNIVERSE"],
        )
        top = refinements[0]
        self.assertEqual(top["sub_universe_diagnosis"], "DATASET_UNIVERSE_MISMATCH")
        self.assertNotIn("rank(group_neutralize", top["expression"])

    def test_prime_window_correlation_variant_marked_parameter_only(self) -> None:
        miner = ContinuousAlphaMiner()
        miner.has_tried_expression = lambda _e: False
        miner.mark_expression_tried = lambda _e: None
        refinements = miner.generate_near_miss_refinements(
            "ts_mean(close, 20)",
            {"universe": "TOP3000", "neutralization": "SUBINDUSTRY", "decay": 4, "truncation": 0.05},
            sharpe=1.60,
            fitness=1.20,
            turnover=0.25,
            failed_checks=["SELF_CORRELATION"],
            max_variants=10,
        )
        prime_items = [c for c in refinements if c.get("name") == "NearMiss_ParamVariantPrimeWindow"]
        self.assertEqual(len(prime_items), 1)
        self.assertEqual(prime_items[0]["edit_class"], "PARAMETER_VARIANT")
        self.assertFalse(prime_items[0]["structural_novelty"])

    def test_migrations_010_011_idempotent_and_rollback_reapply(self) -> None:
        m010 = (REPO_ROOT / "migrations" / "010_forbidden_edit_motifs.sql").read_text(encoding="utf-8")
        m010_rb = (REPO_ROOT / "migrations" / "010_forbidden_edit_motifs_rollback.sql").read_text(encoding="utf-8")
        m011 = (REPO_ROOT / "migrations" / "011_pareto_archive.sql").read_text(encoding="utf-8")
        m011_rb = (REPO_ROOT / "migrations" / "011_pareto_archive_rollback.sql").read_text(encoding="utf-8")

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            conn = sqlite3.connect(db_path)
            # Apply twice (idempotency)
            conn.executescript(m010)
            conn.executescript(m010)
            conn.executescript(m011)
            conn.executescript(m011)
            # Rollback cleanly
            conn.executescript(m011_rb)
            conn.executescript(m010_rb)
            remaining = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            self.assertEqual(remaining, [])
            # Reapply cleanly
            conn.executescript(m010)
            conn.executescript(m011)
            reapplied = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            self.assertIn("pareto_archive_entries", reapplied)
            self.assertIn("research_edit_motifs", reapplied)
            conn.close()


if __name__ == "__main__":
    unittest.main()
