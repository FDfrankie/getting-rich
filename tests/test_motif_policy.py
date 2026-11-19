"""Unit tests for Forbidden Edit Motif statistical policy (Wilson LB, Beta-Binomial, Veto decisions)."""
from __future__ import annotations

import unittest

from worldquant_brain.forbidden_edit_motif import (
    MotifDecisionAction,
    MotifPolicy,
    VetoLevel,
    decide_motif_action,
    determine_veto_level,
    posterior_failure_probability,
    wilson_lower_bound,
)


def _policy(
    *,
    attempts: int,
    failures: int,
    parents: int,
    batches: int,
    veto: VetoLevel,
    mult: float,
    exp: float,
    cooldown: str | None = None,
) -> MotifPolicy:
    return MotifPolicy(
        motif_id="m_1",
        motif_context_id="mc_1",
        policy_version="shadow-v1",
        attempts=attempts,
        successes=attempts - failures,
        failures=failures,
        distinct_parent_count=parents,
        distinct_batch_count=batches,
        posterior_failure_probability=posterior_failure_probability(failures=failures, attempts=attempts),
        wilson_lower_bound=wilson_lower_bound(failures, attempts),
        veto_level=veto,
        prior_multiplier=mult,
        exploration_probability=exp,
        cooldown_until=cooldown,
    )


class TestMotifPolicy(unittest.TestCase):
    def test_no_evidence_allows_motif(self) -> None:
        dec = decide_motif_action(None, base_prior=0.50)
        self.assertEqual(dec.action, MotifDecisionAction.ALLOW)
        self.assertAlmostEqual(dec.prior_after, 0.50)

    def test_three_failures_trigger_cooldown_only(self) -> None:
        lvl, mult, exp = determine_veto_level(
            attempts=3,
            failures=3,
            distinct_parent_count=2,
            distinct_batch_count=1,
            permanent_invalid_reason=None,
        )
        self.assertEqual(lvl, VetoLevel.COOLDOWN)
        self.assertGreater(exp, 0.0)

    def test_soft_veto_requires_support_and_batches(self) -> None:
        # 5/5 failures in 1 batch only -> NOT soft veto yet (stays cooldown)
        lvl_1b, _, _ = determine_veto_level(
            attempts=5,
            failures=5,
            distinct_parent_count=3,
            distinct_batch_count=1,
            permanent_invalid_reason=None,
        )
        self.assertNotEqual(lvl_1b, VetoLevel.SOFT)

        # 6/6 failures across >=2 batches and >=2 parents -> SOFT veto (Wilson LB > 0.60)
        lvl_2b, mult, exp = determine_veto_level(
            attempts=6,
            failures=6,
            distinct_parent_count=3,
            distinct_batch_count=2,
            permanent_invalid_reason=None,
        )
        self.assertEqual(lvl_2b, VetoLevel.SOFT)
        self.assertLess(mult, 1.0)
        self.assertGreater(exp, 0.0)

    def test_strong_veto_keeps_small_exploration_probability(self) -> None:
        lvl, mult, exp = determine_veto_level(
            attempts=12,
            failures=12,
            distinct_parent_count=5,
            distinct_batch_count=3,
            permanent_invalid_reason=None,
        )
        self.assertEqual(lvl, VetoLevel.STRONG)
        self.assertGreater(exp, 0.0)
        self.assertLessEqual(exp, 0.10)

        pol = _policy(attempts=12, failures=12, parents=5, batches=3, veto=lvl, mult=mult, exp=exp)
        dec_explore = decide_motif_action(pol, base_prior=0.80, exploration_draw=0.01)
        self.assertEqual(dec_explore.action, MotifDecisionAction.ALLOW_EXPLORATION)

    def test_permanent_veto_has_zero_exploration(self) -> None:
        pol = _policy(attempts=1, failures=1, parents=1, batches=1, veto=VetoLevel.PERMANENT, mult=0.0, exp=0.0)
        dec = decide_motif_action(pol, base_prior=0.90, exploration_draw=0.0)
        self.assertEqual(dec.action, MotifDecisionAction.REJECT)
        self.assertEqual(dec.prior_after, 0.0)

    def test_wilson_lower_bound_is_monotonic(self) -> None:
        lb_small = wilson_lower_bound(3, 3)
        lb_mid = wilson_lower_bound(6, 6)
        lb_large = wilson_lower_bound(15, 15)
        self.assertLess(lb_small, lb_mid)
        self.assertLess(lb_mid, lb_large)

    def test_posterior_failure_probability_uses_smoothing(self) -> None:
        p = posterior_failure_probability(failures=1, attempts=1)
        self.assertGreater(p, 0.50)
        self.assertLess(p, 1.0)


if __name__ == "__main__":
    unittest.main()
