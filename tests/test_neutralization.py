"""Offline unit tests for Neutralization-Aware Architecture & Multi-Objective Ablation."""
from __future__ import annotations

import unittest

from worldquant_brain.neutralization import (
    NeutralizationConfiguration,
    NeutralizationDiagnosis,
    NeutralizationLayer,
    NeutralizationMetrics,
    NeutralizationMode,
    NeutralizationTrial,
    diagnose_neutralization,
    evaluate_neutralization_ablation,
    expression_setting_equivalent,
    has_effective_final_neutralization,
    neutralization_succeeded,
    neutralized_vector,
    score_neutralization_trial,
)


def config(**kw) -> NeutralizationConfiguration:
    d = dict(
        simulation_mode=NeutralizationMode.INDUSTRY,
        layer=NeutralizationLayer.SIMULATION_SETTING,
        long_count=100,
        short_count=100,
    )
    d.update(kw)
    return NeutralizationConfiguration(**d)


def metrics(**kw) -> NeutralizationMetrics:
    d = dict(
        sharpe=1.5,
        returns=0.12,
        drawdown=0.09,
        fitness=1.3,
        turnover=0.25,
        margin=0.001,
        self_correlation=0.4,
        sub_universe_passed=True,
        os_stability=0.75,
        market_beta=0.35,
        long_count=100,
        short_count=100,
    )
    d.update(kw)
    return NeutralizationMetrics(**d)


def trial(tid: str, mode: NeutralizationMode, **m) -> NeutralizationTrial:
    return NeutralizationTrial(tid, config(simulation_mode=mode), metrics(**m))


class TestNeutralization(unittest.TestCase):
    def test_none_without_expression_neutralization_missing(self) -> None:
        c = config(simulation_mode=NeutralizationMode.NONE, expression_group_neutralize_count=0)
        self.assertFalse(has_effective_final_neutralization(c))
        self.assertIn(
            NeutralizationDiagnosis.MISSING_FINAL_NEUTRALIZATION,
            diagnose_neutralization(c, metrics(), metrics()),
        )

    def test_none_with_outermost_group_is_effective(self) -> None:
        c = config(
            simulation_mode=NeutralizationMode.NONE,
            expression_group_neutralize_count=1,
            expression_groups=("industry",),
            outermost_group_neutralize=True,
        )
        self.assertTrue(has_effective_final_neutralization(c))
        self.assertTrue(expression_setting_equivalent(c))

    def test_post_transform_breaks_equivalence(self) -> None:
        c = config(
            simulation_mode=NeutralizationMode.NONE,
            expression_group_neutralize_count=1,
            expression_groups=("industry",),
            outermost_group_neutralize=True,
            post_neutralization_weight_transform=True,
        )
        self.assertFalse(has_effective_final_neutralization(c))
        self.assertFalse(expression_setting_equivalent(c))

    def test_decay_or_truncation_breaks_equivalence(self) -> None:
        self.assertFalse(
            expression_setting_equivalent(
                config(
                    simulation_mode=NeutralizationMode.NONE,
                    expression_group_neutralize_count=1,
                    expression_groups=("industry",),
                    outermost_group_neutralize=True,
                    decay=4,
                )
            )
        )
        self.assertFalse(
            expression_setting_equivalent(
                config(
                    simulation_mode=NeutralizationMode.NONE,
                    expression_group_neutralize_count=1,
                    expression_groups=("industry",),
                    outermost_group_neutralize=True,
                    truncation=0.08,
                )
            )
        )

    def test_double_neutralization_is_warning_not_automatic_failure(self) -> None:
        c = config(
            simulation_mode=NeutralizationMode.SUBINDUSTRY,
            expression_group_neutralize_count=1,
            expression_groups=("industry",),
            outermost_group_neutralize=True,
        )
        result = score_neutralization_trial(
            NeutralizationTrial("x", c, metrics(market_beta=0.1)),
            NeutralizationTrial("b", config(simulation_mode=NeutralizationMode.NONE), metrics()),
        )
        self.assertIn(NeutralizationDiagnosis.DOUBLE_NEUTRALIZATION, result.diagnoses)
        self.assertIn("DOUBLE_NEUTRALIZATION_REQUIRES_ABLATION", result.warnings)

    def test_long_short_imbalance_boundary(self) -> None:
        for long_c, short_c, triggered in [(60, 40, False), (61, 39, True), (40, 60, False), (39, 61, True)]:
            c = config(long_count=long_c, short_count=short_c)
            d = diagnose_neutralization(c, metrics(), metrics(long_count=long_c, short_count=short_c))
            self.assertIs(NeutralizationDiagnosis.LONG_SHORT_IMBALANCE in d, triggered)

    def test_market_beta_boundary(self) -> None:
        for beta, triggered in [(0.3, False), (0.300001, True), (-0.3, False), (-0.300001, True)]:
            d = diagnose_neutralization(config(), metrics(), metrics(market_beta=beta))
            self.assertIs(NeutralizationDiagnosis.MARKET_BETA_EXPOSURE in d, triggered)

    def test_return_retention_below_25_is_dependency(self) -> None:
        d = diagnose_neutralization(config(), metrics(returns=0.12), metrics(returns=0.029))
        self.assertIn(NeutralizationDiagnosis.RISK_FACTOR_DEPENDENCY, d)

    def test_risk_down_but_sharpe_collapse_not_success(self) -> None:
        self.assertFalse(
            neutralization_succeeded(
                metrics(sharpe=1.5, returns=0.12, drawdown=0.12),
                metrics(sharpe=0.8, returns=0.10, drawdown=0.04, market_beta=0.1),
            )
        )

    def test_success_requires_return_retention_and_os(self) -> None:
        self.assertTrue(
            neutralization_succeeded(
                metrics(sharpe=1.5, returns=0.12, drawdown=0.12),
                metrics(sharpe=1.45, returns=0.10, drawdown=0.07, market_beta=0.1, os_stability=0.7),
            )
        )

    def test_constant_signal_collapses_and_group_means_zero(self) -> None:
        self.assertEqual(neutralized_vector([1, 1, 1], ["a", "a", "a"]), [0, 0, 0])
        d = diagnose_neutralization(config(), metrics(), metrics(), constant_expression=True)
        self.assertIn(NeutralizationDiagnosis.CONSTANT_SIGNAL_COLLAPSE, d)
        self.assertEqual(neutralized_vector([1, 3, 2, 6], ["a", "a", "b", "b"]), [-1, 1, -2, 2])

    def test_ablation_selects_multiobjective_best_not_highest_sharpe_only(self) -> None:
        baseline = trial("base", NeutralizationMode.NONE, sharpe=1.4, returns=0.15, drawdown=0.16, market_beta=0.6, os_stability=0.65)
        high_sharpe = trial("market", NeutralizationMode.MARKET, sharpe=1.8, returns=0.04, drawdown=0.10, market_beta=0.05, os_stability=0.7)
        balanced = trial("industry", NeutralizationMode.INDUSTRY, sharpe=1.6, returns=0.12, drawdown=0.08, market_beta=0.1, os_stability=0.8)
        report = evaluate_neutralization_ablation(baseline, [high_sharpe, balanced])
        self.assertEqual(report.selected_trial_id, "industry")

    def test_no_successful_trials_returns_none(self) -> None:
        baseline = trial("base", NeutralizationMode.NONE, sharpe=1.4, returns=0.15, drawdown=0.12, market_beta=0.6)
        bad = trial("bad", NeutralizationMode.MARKET, sharpe=0.5, returns=0.01, drawdown=0.2, market_beta=0.5, os_stability=0.2)
        report = evaluate_neutralization_ablation(baseline, [bad])
        self.assertIsNone(report.selected_trial_id)


if __name__ == "__main__":
    unittest.main()
