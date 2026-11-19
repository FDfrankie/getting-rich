from __future__ import annotations

import pathlib
import sqlite3
import tempfile
import unittest

from worldquant_brain.alpha_novelty import (
    evaluate_correlation_buffer,
    find_duplicate,
)
from worldquant_brain.brain_client import BrainClient
from worldquant_brain.data_survey import (
    FieldProfile,
    rank_fields,
    should_probe_field,
    validate_vector_field_reducer,
)
from worldquant_brain.simulation_budget import (
    Candidate,
    ThreeMinuteSimulationBudget,
)


class TestNoveltySurveyAndBudget12Gates(unittest.TestCase):
    # 1. Exact duplicate rejected
    def test_01_exact_duplicate_rejected(self):
        existing = [{"alpha_id": "A1", "expression": "rank(operating_cashflow / cap)"}]
        dec = find_duplicate("rank(operating_cashflow / cap)", existing)
        self.assertTrue(dec.blocked)
        self.assertEqual(dec.reason, "EXACT_EXPRESSION")

    # 2. Parameter-only variant rejected
    def test_02_parameter_only_variant_rejected(self):
        existing = [{"alpha_id": "A1", "expression": "ts_rank(operating_cashflow, 20)"}]
        dec = find_duplicate("ts_rank(operating_cashflow, 21)", existing)
        self.assertTrue(dec.blocked)
        self.assertEqual(dec.reason, "PARAMETER_ONLY_VARIANT")

    # 3. Cross-domain novel formula not falsely blocked
    def test_03_cross_domain_candidate_not_blocked(self):
        existing = [{"alpha_id": "A1", "expression": "rank(ts_delta(close, 20))"}]
        dec = find_duplicate("group_neutralize(ts_zscore(eps_consensus_mean, 40), subindustry)", existing)
        self.assertFalse(dec.blocked)
        self.assertEqual(dec.reason, "NOVEL")

    # 4. Queue 175s throttling works when configured
    def test_04_queue_175s_throttle(self):
        budget = ThreeMinuteSimulationBudget(minimum_interval_seconds=175.0, max_pending=3)
        c1 = Candidate("C1", "rank(operating_cashflow)", "FAM_A", "FUNDAMENTAL", local_corr=0.2)
        c2 = Candidate("C2", "rank(eps_consensus_mean)", "FAM_B", "ANALYST", local_corr=0.2)
        self.assertTrue(budget.admit(c1, now_ts=1000.0).admitted)
        d2 = budget.admit(c2, now_ts=1050.0)
        self.assertFalse(d2.admitted)
        self.assertEqual(d2.reason, "THROTTLED_MIN_INTERVAL")
        self.assertTrue(budget.admit(c2, now_ts=1200.0).admitted)

    # 5. Same candidate cannot enter queue twice
    def test_05_same_candidate_not_requeued(self):
        budget = ThreeMinuteSimulationBudget(minimum_interval_seconds=0.0, max_pending=3)
        c1 = Candidate("C1", "rank(operating_cashflow)", "FAM_A", "FUNDAMENTAL", local_corr=0.2)
        self.assertTrue(budget.admit(c1).admitted)
        d2 = budget.admit(c1)
        self.assertFalse(d2.admitted)
        self.assertEqual(d2.reason, "DUPLICATE_CANDIDATE_IN_QUEUE")

    # 6. Same blueprint family only runs 1 baseline at a time
    def test_06_single_baseline_per_family(self):
        budget = ThreeMinuteSimulationBudget(minimum_interval_seconds=0.0, max_pending=3, allow_multiple_per_family=False)
        c1 = Candidate("C1", "rank(operating_cashflow)", "ANALYST_REV", "ANALYST", local_corr=0.2)
        c2 = Candidate("C2", "rank(eps_consensus_mean)", "ANALYST_REV", "ANALYST", local_corr=0.2)
        self.assertTrue(budget.admit(c1).admitted)
        d2 = budget.admit(c2)
        self.assertFalse(d2.admitted)
        self.assertEqual(d2.reason, "FAMILY_BASELINE_ALREADY_ACTIVE")

    # 7. Field Survey preserves category quotas (PV cannot monopolize)
    def test_07_field_survey_preserves_category_quotas(self):
        fields = [FieldProfile(f"pv_{i}", "PV1", "PRICE_VOLUME", coverage=0.99) for i in range(60)]
        fields += [FieldProfile(f"anl_{i}", "ANL1", "ANALYST", coverage=0.70) for i in range(10)]
        ranked = rank_fields(fields, seen_fields=set(), quotas={"PRICE_VOLUME": 5, "ANALYST": 10}, total=15)
        cats = [r.field.category for r in ranked]
        self.assertEqual(cats.count("PRICE_VOLUME"), 5)
        self.assertEqual(cats.count("ANALYST"), 10)

    # 8. Unchanged metadata_hash skips re-probing
    def test_08_unchanged_metadata_skips_probe(self):
        fp = FieldProfile("eps_consensus_mean", "ANL1", "ANALYST", coverage=0.65)
        h = fp.metadata_hash()
        self.assertFalse(should_probe_field(fp, cached_metadata_hash=h))
        self.assertTrue(should_probe_field(fp, cached_metadata_hash="old_hash"))

    # 9. Vector field without reducer is rejected
    def test_09_vector_field_without_reducer_rejected(self):
        vf = FieldProfile("news_vec", "NEWS1", "NEWS", field_type="VECTOR", vector_reducer=None)
        ok, msg = validate_vector_field_reducer(vf)
        self.assertFalse(ok)
        self.assertIn("VECTOR_FIELD_MISSING_REDUCER", msg)

    # 10. Local corr >= 0.60 is rejected before BRAIN (and buffer tiers work)
    def test_10_local_corr_ge_060_rejected_before_brain(self):
        self.assertEqual(evaluate_correlation_buffer(0.30).decision, "PRIORITY")
        self.assertEqual(evaluate_correlation_buffer(0.47).decision, "ACCEPT_LOW_PRIORITY")
        self.assertEqual(evaluate_correlation_buffer(0.55).decision, "QUARANTINE")
        rej = evaluate_correlation_buffer(0.62)
        self.assertEqual(rej.decision, "REJECT_BEFORE_BRAIN")
        self.assertFalse(rej.allowed_for_brain)

        budget = ThreeMinuteSimulationBudget(minimum_interval_seconds=0.0, max_pending=3)
        c_high_corr = Candidate("C_CORR", "rank(eps_consensus_mean)", "FAM_C", "ANALYST", local_corr=0.62)
        d = budget.admit(c_high_corr)
        self.assertFalse(d.admitted)
        self.assertEqual(d.reason, "REJECT_BEFORE_BRAIN")

    # 11. Immediate queuing after thinking finishes when cooldown=0 and pending < max_pending
    def test_11_immediate_queuing_without_forced_180s_wait(self):
        budget = ThreeMinuteSimulationBudget(minimum_interval_seconds=0.0, max_pending=3)
        for idx, fam in enumerate(["F1", "F2", "F3"]):
            c = Candidate(f"C{idx}", f"rank(field_{idx})", fam, "FUNDAMENTAL", local_corr=0.2)
            self.assertTrue(budget.admit(c, now_ts=1000.0).admitted)
        # 4th is blocked only by max_pending=3, not by 180s timer
        c4 = Candidate("C4", "rank(field_4)", "F4", "ANALYST", local_corr=0.2)
        d4 = budget.admit(c4, now_ts=1000.0)
        self.assertFalse(d4.admitted)
        self.assertEqual(d4.reason, "MAX_PENDING_REACHED")

    # 12. Public formula cannot enter Candidate Pool & 6 idempotent tables exist
    def test_12_public_formula_blocked_from_candidate_pool_and_tables_created(self):
        db_file = str(pathlib.Path(tempfile.mkdtemp()) / "public_block.db")
        bc = BrainClient(email="test@example.com", password="secret")
        bc.quota_db_path = db_file
        bc.stage_candidate_for_daily_selection(
            "PUB_101",
            "(close - open) / ((high - low) + 0.001)",
            1.80, 1.30, 0.25, 0.002, day_str="2026-11-19",
        )
        conn = sqlite3.connect(db_file)
        pool_cnt = conn.execute("SELECT COUNT(*) FROM brain_candidate_pool").fetchone()[0]
        denylist_cnt = conn.execute("SELECT COUNT(*) FROM public_expression_denylist").fetchone()[0]
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        conn.close()
        self.assertEqual(pool_cnt, 0)
        self.assertGreaterEqual(denylist_cnt, 4)
        for req_tbl in (
            "public_expression_denylist",
            "alpha_fingerprints",
            "field_catalog",
            "field_probe_cache",
            "simulation_queue_decisions",
            "correlation_prechecks",
        ):
            self.assertIn(req_tbl, tables)


if __name__ == "__main__":
    unittest.main()
