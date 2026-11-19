"""Comprehensive unittest suite for Objective-Aware Alpha Architecture integration & migrations."""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from worldquant_brain.alpha_yield_booster import precision_repair_near_miss
from worldquant_brain.evidence_objective_bridge import build_evidence_context
from worldquant_brain.objective_aware_scheduler import (
    LaneCandidate,
    ObjectiveLaneBudget,
    allocate_lane_slots,
    expected_yield_per_minute,
    select_candidates,
)
from worldquant_brain.optimization_policy import (
    AlphaPerformance,
    ObjectivePolicy,
    OptimizationObjective,
    recommend_actions,
    turnover_efficiency,
)


def _perf(**kwargs) -> AlphaPerformance:
    data = dict(
        returns=0.10,
        net_returns=0.09,
        sharpe=1.3,
        fitness=1.1,
        turnover=0.3,
        margin=0.001,
        drawdown=0.08,
        volatility=0.12,
        previous_net_returns=0.08,
        previous_turnover=0.2,
    )
    data.update(kwargs)
    return AlphaPerformance(**data)


class TestObjectiveAwareIntegration(unittest.TestCase):
    def test_productive_turnover_preserved_in_policy_and_booster(self) -> None:
        m = _perf(net_returns=0.18, turnover=0.60, previous_net_returns=0.08, previous_turnover=0.20)
        self.assertAlmostEqual(turnover_efficiency(m), 0.25)
        actions = {a.action for a in recommend_actions(m, ObjectivePolicy(OptimizationObjective.RETURN))}
        self.assertIn("PRESERVE_TURNOVER_AND_STRESS_COST", actions)
        self.assertNotIn("APPLY_HUMP", actions)

        repairs = precision_repair_near_miss(
            "rank(close)",
            {"decay": 2, "universe": "TOP1000"},
            sharpe=1.15,
            fitness=0.95,
            turnover=0.60,
            objective="RETURN",
            net_returns=0.18,
            previous_net_returns=0.08,
            previous_turnover=0.20,
        )
        self.assertTrue(any(r["name"] == "Preserve_ProductiveTurnover_CostStress" for r in repairs))

    def test_wasteful_turnover_recommends_hump_and_trade_when(self) -> None:
        m = _perf(net_returns=0.085, turnover=0.60, previous_net_returns=0.08, previous_turnover=0.20)
        actions = {a.action for a in recommend_actions(m, ObjectivePolicy(OptimizationObjective.BALANCED))}
        self.assertIn("APPLY_HUMP", actions)
        self.assertIn("ADD_TRADE_WHEN", actions)

    def test_stale_signal_recommends_lower_decay(self) -> None:
        m = _perf(net_returns=0.03, returns=0.04, sharpe=0.7, turnover=0.04, previous_net_returns=0.03, previous_turnover=0.04)
        actions = {a.action for a in recommend_actions(m, ObjectivePolicy(OptimizationObjective.RETURN))}
        self.assertIn("LOWER_DECAY", actions)

    def test_return_vs_capacity_objectives_differ(self) -> None:
        m = _perf(net_returns=0.12, turnover=0.35)
        ret_actions = {a.action for a in recommend_actions(m, ObjectivePolicy(OptimizationObjective.RETURN))}
        cap_actions = {a.action for a in recommend_actions(m, ObjectivePolicy(OptimizationObjective.CAPACITY))}
        self.assertIn("TEST_LOWER_DECAY_CONTROLLED", ret_actions)
        self.assertIn("REDUCE_WASTEFUL_TURNOVER", cap_actions)
        self.assertNotEqual(ret_actions, cap_actions)

    def test_lane_budget_slots_and_yield_per_minute(self) -> None:
        budget = ObjectiveLaneBudget()
        budget.validate()
        slots = allocate_lane_slots(20, budget)
        self.assertEqual(slots[OptimizationObjective.RETURN], 6)
        self.assertEqual(slots[OptimizationObjective.SHARPE], 6)
        self.assertEqual(slots[OptimizationObjective.CAPACITY], 3)
        self.assertEqual(slots[OptimizationObjective.BALANCED], 5)

        cands = [
            LaneCandidate("fast_good", OptimizationObjective.RETURN, 0.8, 0.8, 0.8, 0.7, 0.8, 0.7, 0.7, 0.8, 0.1, 0.1, 2.0),
            LaneCandidate("slow_corr", OptimizationObjective.RETURN, 0.5, 0.5, 0.5, 0.5, 0.3, 0.4, 0.4, 0.4, 0.7, 0.4, 4.0),
        ]
        self.assertGreater(expected_yield_per_minute(cands[0]), expected_yield_per_minute(cands[1]))
        sel = select_candidates(cands, total_slots=2)
        self.assertEqual(sel[0].candidate_id, "fast_good")

    def test_evidence_objective_bridge_and_007_migration_idempotent_rollback(self) -> None:
        before = _perf(net_returns=0.08, turnover=0.20)
        after = _perf(net_returns=0.18, turnover=0.60, previous_net_returns=0.08, previous_turnover=0.20)
        ctx = build_evidence_context(
            before=before,
            after=after,
            policy=ObjectivePolicy(OptimizationObjective.RETURN),
            action="PRESERVE_TURNOVER_AND_STRESS_COST",
        )
        self.assertEqual(ctx["optimization_objective"], "RETURN")
        self.assertAlmostEqual(ctx["turnover_efficiency"], 0.25)

        root = Path(__file__).resolve().parents[1]
        up_sql = (root / "migrations" / "007_objective_aware_optimization.sql").read_text(encoding="utf-8")
        down_sql = (root / "migrations" / "007_objective_aware_optimization_rollback.sql").read_text(encoding="utf-8")

        with tempfile.TemporaryDirectory() as td:
            conn = sqlite3.connect(str(Path(td) / "obj.db"))
            # Apply twice (idempotency), rollback, and reapply
            conn.executescript(up_sql)
            conn.executescript(up_sql)
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertTrue({
                "alpha_optimization_policies",
                "alpha_optimization_decisions",
                "alpha_objective_lane_runs",
                "alpha_execution_shaping_trials",
            }.issubset(tables))
            conn.executescript(down_sql)
            tables_after_down = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertNotIn("alpha_optimization_policies", tables_after_down)
            conn.executescript(up_sql)
            tables_reapplied = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertIn("alpha_optimization_policies", tables_reapplied)
            conn.close()


if __name__ == "__main__":
    unittest.main()
