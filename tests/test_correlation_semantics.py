"""Offline tests for PnL-based Correlation Semantics, D0/D1 pairs, Parent/Child repair, and Evidence DB."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from worldquant_brain.correlation_evidence import (
    AlphaCorrelationEvidence,
    CorrelationEvidenceRepository,
    CorrelationScope,
    CorrelationSource,
    CorrelationType,
    DelayPairMetrics,
    evaluate_delay_pair,
    evaluate_parent_child_repair,
    official_self_correlation_passed,
    resolve_correlation_scope,
)


def _ev(**overrides) -> AlphaCorrelationEvidence:
    d = dict(
        candidate_alpha_id="cand-1",
        compared_alpha_id="pool-1",
        correlation_type=CorrelationType.SELF,
        correlation_source=CorrelationSource.BRAIN_SELF_CORRELATION,
        correlation_value=0.45,
        candidate_delay=1,
        compared_delay=1,
        pnl_period_start="2014-01-01",
        pnl_period_end="2024-01-01",
        observations=2520,
        official=True,
        checked_at="2026-11-19T06:00:00Z",
    )
    d.update(overrides)
    return AlphaCorrelationEvidence(**d)


class TestCorrelationSemantics(unittest.TestCase):
    def test_delay_pair_unverified_and_thresholds(self) -> None:
        base = dict(
            expression_hash="h1",
            d0_alpha_id="a_d0",
            d1_alpha_id="a_d1",
            d0_sharpe=2.2,
            d1_sharpe=1.9,
            d0_fitness=1.7,
            d1_fitness=1.6,
            d0_portfolio_utility=0.80,
            d1_portfolio_utility=0.65,
        )
        # Same expression D0/D1 without official PnL correlation -> CORRELATION_NOT_VERIFIED
        d_unv = evaluate_delay_pair(DelayPairMetrics(**base, pnl_correlation=None, pnl_correlation_official=False))
        self.assertEqual(d_unv.decision, "CORRELATION_NOT_VERIFIED")

        # < 0.50 -> keep both
        d_low = evaluate_delay_pair(DelayPairMetrics(**base, pnl_correlation=0.42, pnl_correlation_official=True))
        self.assertEqual(d_low.decision, "KEEP_BOTH_FOR_PORTFOLIO_REVIEW")

        # 0.50 <= corr < 0.70 -> keep preferred
        d_mid = evaluate_delay_pair(DelayPairMetrics(**base, pnl_correlation=0.62, pnl_correlation_official=True))
        self.assertEqual(d_mid.decision, "KEEP_PREFERRED_HOLD_OTHER")
        self.assertEqual(d_mid.preferred_alpha_id, "a_d0")

        # >= 0.70 -> keep one
        d_hi = evaluate_delay_pair(DelayPairMetrics(**base, pnl_correlation=0.70, pnl_correlation_official=True))
        self.assertEqual(d_hi.decision, "HIGHLY_CORRELATED_KEEP_ONE")
        self.assertEqual(d_hi.preferred_alpha_id, "a_d0")

    def test_only_non_expired_official_self_correlation_passes_gate(self) -> None:
        self.assertTrue(official_self_correlation_passed(_ev(correlation_value=0.6999)))
        self.assertFalse(official_self_correlation_passed(_ev(correlation_value=0.7000)))
        self.assertFalse(official_self_correlation_passed(_ev(expired=True)))
        # Proxy / Inner / Prod / Fingerprint must NEVER pass official self-correlation gate
        self.assertFalse(
            official_self_correlation_passed(
                _ev(
                    correlation_type=CorrelationType.LOCAL_PNL_PROXY,
                    correlation_source=CorrelationSource.LOCAL_PNL,
                    official=False,
                    correlation_value=0.10,
                )
            )
        )
        self.assertFalse(
            official_self_correlation_passed(
                _ev(
                    correlation_type=CorrelationType.STRUCTURAL_PROXY,
                    correlation_source=CorrelationSource.EXPRESSION_FINGERPRINT,
                    official=False,
                    correlation_value=0.10,
                )
            )
        )
        self.assertFalse(
            official_self_correlation_passed(
                _ev(
                    correlation_type=CorrelationType.INNER,
                    correlation_source=CorrelationSource.BRAIN_ALPHA_LIST,
                )
            )
        )

    def test_scope_resolution(self) -> None:
        self.assertEqual(resolve_correlation_scope(is_consultant=False), CorrelationScope.OWN_ALPHA_POOL)
        self.assertEqual(resolve_correlation_scope(is_consultant=True), CorrelationScope.BRAIN_ALPHA_POOL)

    def test_parent_child_repair_validation(self) -> None:
        ok, reason = evaluate_parent_child_repair(
            parent_child_pnl_corr=0.45,
            official_corr=True,
            target_gate_resolved=True,
            parent_sharpe=1.50,
            child_sharpe=1.65,
            performance_improvement_ratio=0.10,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "REPAIR_SUCCEEDED")

        # High corr >= 0.70 without 10% improvement -> fail
        ok2, reason2 = evaluate_parent_child_repair(
            parent_child_pnl_corr=0.75,
            official_corr=True,
            target_gate_resolved=True,
            parent_sharpe=1.50,
            child_sharpe=1.55,
            performance_improvement_ratio=0.05,
        )
        self.assertFalse(ok2)
        self.assertIn("10PCT", reason2)

        # Severe Sharpe degradation -> fail
        ok3, reason3 = evaluate_parent_child_repair(
            parent_child_pnl_corr=0.30,
            official_corr=True,
            target_gate_resolved=True,
            parent_sharpe=1.80,
            child_sharpe=1.10,
            performance_improvement_ratio=0.0,
        )
        self.assertFalse(ok3)
        self.assertEqual(reason3, "SHARPE_SEVERELY_DEGRADED")

    def test_repository_preserves_distinct_periods_and_top5(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "corr.db"
            repo = CorrelationEvidenceRepository(db_path)
            repo.migrate()
            repo.migrate()  # idempotent
            for i in range(6):
                repo.save_evidence(
                    _ev(
                        compared_alpha_id=f"pool-{i}",
                        correlation_value=0.10 + i * 0.10,
                        correlation_source=CorrelationSource.BRAIN_GENERATE_SELF_CORRELATION,
                    )
                )
            # Different PnL period does NOT overwrite
            repo.save_evidence(
                _ev(
                    compared_alpha_id="pool-0",
                    correlation_value=0.55,
                    pnl_period_start="2020-01-01",
                    pnl_period_end="2024-01-01",
                )
            )
            top5 = repo.list_for_candidate("cand-1", correlation_type=CorrelationType.SELF, limit=5)
            self.assertEqual(len(top5), 5)
            self.assertGreaterEqual(top5[0]["correlation_value"], top5[1]["correlation_value"])


if __name__ == "__main__":
    unittest.main()
