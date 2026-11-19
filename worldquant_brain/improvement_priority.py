"""Alpha Improvement Priority Matrix (P0-P6), Yield-per-Minute Scoring, and Per-Parent Repair Budgets."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Mapping

from worldquant_brain.optimization_policy import (
    AlphaDiagnosis,
    AlphaPerformance,
    ObjectivePolicy,
    OptimizationObjective,
    diagnose_alpha,
    recommend_actions,
    turnover_efficiency,
)


class PriorityTier(StrEnum):
    P0 = "P0"  # Submission Blocker
    P1 = "P1"  # High-Value Near Miss
    P2 = "P2"  # Correlation / Diversity Repair
    P3 = "P3"  # Return / Sharpe Research
    P4 = "P4"  # Execution Shaping
    P5 = "P5"  # Robustness
    P6 = "P6"  # Low-Value / Archive


class AlphaDisposition(StrEnum):
    PROMOTE = "PROMOTE"
    ROBUSTNESS = "ROBUSTNESS"
    TARGETED_REPAIR = "TARGETED_REPAIR"
    STRUCTURAL_RESEARCH = "STRUCTURAL_RESEARCH"
    CORRELATION_REPAIR = "CORRELATION_REPAIR"
    ARCHIVE = "ARCHIVE"
    REJECT = "REJECT"


DEFAULT_REPAIR_BUDGETS: dict[PriorityTier, int] = {
    PriorityTier.P0: 0,
    PriorityTier.P1: 3,
    PriorityTier.P2: 2,
    PriorityTier.P3: 2,
    PriorityTier.P4: 3,
    PriorityTier.P5: 4,
    PriorityTier.P6: 0,
}


@dataclass(frozen=True, slots=True)
class PriorityScoreInputs:
    prob_submit_ready: float = 0.5
    expected_net_return_gain: float = 0.1
    expected_sharpe_gain: float = 0.1
    expected_fitness_gain: float = 0.1
    correlation_pass_prob: float = 0.7
    robustness_prob: float = 0.5
    expected_portfolio_utility: float = 0.1
    information_gain: float = 0.2
    domain_novelty: float = 0.5
    simulation_cost: float = 0.1
    correlation_risk: float = 0.2
    overfitting_risk: float = 0.1
    complexity_penalty: float = 0.1
    data_quality_risk: float = 0.05
    expected_minutes: float = 3.0


@dataclass(frozen=True, slots=True)
class AlphaImprovementPlan:
    alpha_id: str
    priority: PriorityTier
    disposition: AlphaDisposition
    objective: OptimizationObjective
    primary_failure_gate: str
    recommended_action: str
    alternatives: tuple[str, ...]
    remaining_budget: int
    expected_yield_per_minute: float
    required_validations: tuple[str, ...]
    allow_simulation_queue: bool
    allow_robustness_promotion: bool
    allow_submission: bool
    forbidden_actions: tuple[str, ...] = ()
    archive_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["priority"] = self.priority.value
        d["disposition"] = self.disposition.value
        d["objective"] = self.objective.value
        return d


def compute_improvement_priority_score(inp: PriorityScoreInputs) -> tuple[float, float]:
    score = (
        0.22 * inp.prob_submit_ready
        + 0.18 * inp.expected_net_return_gain
        + 0.16 * inp.expected_sharpe_gain
        + 0.12 * inp.expected_fitness_gain
        + 0.10 * inp.correlation_pass_prob
        + 0.08 * inp.robustness_prob
        + 0.06 * inp.expected_portfolio_utility
        + 0.04 * inp.information_gain
        + 0.04 * inp.domain_novelty
        - 0.12 * inp.simulation_cost
        - 0.12 * inp.correlation_risk
        - 0.08 * inp.overfitting_risk
        - 0.06 * inp.complexity_penalty
        - 0.06 * inp.data_quality_risk
    )
    yield_per_min = score / max(float(inp.expected_minutes), 0.1)
    return round(score, 4), round(yield_per_min, 4)


def evaluate_improvement_priority(
    record: Mapping[str, Any],
    *,
    objective: OptimizationObjective = OptimizationObjective.BALANCED,
    repairs_used: int = 0,
    consecutive_no_gain: int = 0,
    score_inputs: PriorityScoreInputs | None = None,
) -> AlphaImprovementPlan:
    """Evaluate official BRAIN result in strict 11-step diagnostic order."""
    alpha_id = str(record.get("brain_alpha_id") or record.get("alpha_id") or "UNKNOWN")
    sharpe = float(record.get("sharpe") or 0.0)
    fitness = float(record.get("fitness") or 0.0)
    turnover = float(record.get("turnover") or 0.0)
    returns = float(record.get("returns") or 0.0)
    net_returns = float(record.get("net_returns") if record.get("net_returns") is not None else returns)
    margin = float(record.get("margin") or 0.0)
    drawdown = float(record.get("drawdown") or 0.08)
    self_corr = record.get("self_correlation")
    novelty = float(record.get("novelty") if record.get("novelty") is not None else 0.7)
    metrics_source = str(record.get("metrics_source") or "UNKNOWN")
    sim_id = record.get("official_simulation_id") or record.get("simulation_id")

    s_inp = score_inputs or PriorityScoreInputs(
        expected_minutes=float(record.get("expected_minutes") or 3.0),
        domain_novelty=novelty,
        correlation_risk=float(self_corr) if self_corr is not None else 0.5,
    )
    _, ypm = compute_improvement_priority_score(s_inp)

    # Step 1: Official provenance
    if metrics_source != "WORLDQUANT_BRAIN" or not sim_id:
        return AlphaImprovementPlan(
            alpha_id, PriorityTier.P0, AlphaDisposition.REJECT, objective,
            "OFFICIAL_PROVENANCE_MISSING", "FETCH_OFFICIAL_METRICS", (),
            0, ypm, ("WORLDQUANT_BRAIN_PROVENANCE",), False, False, False,
            forbidden_actions=("PARAMETER_TUNING", "SUBMIT"),
        )

    # Step 2: Bias / Data chronology (Terminal P0)
    if record.get("bias_test_passed") is False:
        return AlphaImprovementPlan(
            alpha_id, PriorityTier.P0, AlphaDisposition.ARCHIVE, objective,
            "BIAS_TEST_FAILED", "AUDIT_DATA_CHRONOLOGY_AND_LAG", (),
            0, ypm, ("OFFICIAL_BIAS_TEST",), False, False, False,
            forbidden_actions=("PARAMETER_VARIANT", "DECAY_TUNING", "WINDOW_TUNING", "NEUTRALIZATION_TUNING"),
            archive_reason="BIAS_TERMINAL_P0",
        )

    # Step 3: Duplicate / Public formula
    if record.get("duplicate_expression") or record.get("duplicate"):
        return AlphaImprovementPlan(
            alpha_id, PriorityTier.P0, AlphaDisposition.REJECT, objective,
            "DUPLICATE_OR_PUBLIC_FORMULA", "GENERATE_NOVEL_HYPOTHESIS", (),
            0, ypm, ("NOVELTY_CHECK",), False, False, False,
        )

    # Step 4: Self-correlation (0.7000 BLOCK, 0.6999 PASS)
    if self_corr is None:
        return AlphaImprovementPlan(
            alpha_id, PriorityTier.P0, AlphaDisposition.CORRELATION_REPAIR, objective,
            "SELF_CORRELATION_MISSING", "FETCH_OFFICIAL_SELF_CORRELATION", (),
            0, ypm, ("SELF_CORRELATION",), False, False, False,
        )
    if float(self_corr) >= 0.70:
        budget = max(0, DEFAULT_REPAIR_BUDGETS[PriorityTier.P2] - repairs_used)
        return AlphaImprovementPlan(
            alpha_id, PriorityTier.P0, AlphaDisposition.CORRELATION_REPAIR, objective,
            "SELF_CORRELATION_BLOCKED", "STRUCTURAL_ORTHOGONALIZATION", ("CHANGE_DATA_DOMAIN", "RESIDUALIZE_SIGNAL"),
            budget, ypm, ("SELF_CORRELATION",), budget > 0, False, False,
            forbidden_actions=("WINDOW_ONLY_TWEAK", "DECAY_ONLY_TWEAK", "TRUNCATION_ONLY_TWEAK"),
        )

    # Step 5: Weight & Sub-universe (before Sharpe/Fitness)
    if record.get("weight_test_passed") is False:
        return AlphaImprovementPlan(
            alpha_id, PriorityTier.P0, AlphaDisposition.TARGETED_REPAIR, objective,
            "WEIGHT_TEST_FAILED", "APPLY_RANK_WINSORIZE_TRUNCATION", ("GROUP_NORMALIZATION",),
            max(0, 2 - repairs_used), ypm, ("WEIGHT_TEST",), True, False, False,
        )
    if record.get("sub_universe_passed") is False:
        return AlphaImprovementPlan(
            alpha_id, PriorityTier.P0, AlphaDisposition.TARGETED_REPAIR, objective,
            "SUB_UNIVERSE_TEST_FAILED", "APPLY_LIQUIDITY_DUAL_DECAY", ("TEST_SMALLER_LIQUID_UNIVERSE",),
            max(0, 2 - repairs_used), ypm, ("SUB_UNIVERSE_TEST",), True, False, False,
        )

    # Step 6: Stop if 2 consecutive repairs yielded no meaningful gain -> P6 Archive
    if consecutive_no_gain >= 2:
        return AlphaImprovementPlan(
            alpha_id, PriorityTier.P6, AlphaDisposition.ARCHIVE, objective,
            "INEFFECTIVE_REPAIR_BUDGET_EXHAUSTED", "ARCHIVE_TO_RESEARCH_MEMORY", (),
            0, ypm, (), False, False, False, archive_reason="NO_IMPROVEMENT_AFTER_2_REPAIRS",
        )

    # Step 7: P6 Low-value cutoff
    if sharpe < 0.80 and fitness < 0.60:
        return AlphaImprovementPlan(
            alpha_id, PriorityTier.P6, AlphaDisposition.ARCHIVE, objective,
            "LOW_VALUE_METRICS", "ARCHIVE_TO_RESEARCH_MEMORY", (),
            0, ypm, (), False, False, False, archive_reason="SHARPE_AND_FITNESS_BELOW_FLOOR",
        )

    # Step 8: Turnover Efficiency & Objective-aware diagnosis
    perf = AlphaPerformance(
        returns=returns,
        net_returns=net_returns,
        sharpe=sharpe,
        fitness=fitness,
        turnover=turnover,
        margin=margin,
        drawdown=drawdown,
        volatility=record.get("volatility"),
        coverage=record.get("coverage"),
        previous_net_returns=record.get("previous_net_returns"),
        previous_turnover=record.get("previous_turnover"),
    )
    pol = ObjectivePolicy(objective)
    diags = diagnose_alpha(perf, pol)
    actions = recommend_actions(perf, pol)

    if AlphaDiagnosis.HIGH_TURNOVER_PRODUCTIVE in diags:
        rem = max(0, DEFAULT_REPAIR_BUDGETS[PriorityTier.P1] - repairs_used)
        return AlphaImprovementPlan(
            alpha_id, PriorityTier.P1, AlphaDisposition.ROBUSTNESS, objective,
            "HIGH_TURNOVER_PRODUCTIVE", "PRESERVE_TURNOVER_AND_STRESS_COST", ("DRAWDOWN_STRESS",),
            rem, ypm, ("COST_1X", "COST_1_5X", "COST_2X", "DRAWDOWN"), rem > 0, True, False,
            forbidden_actions=("MECHANICAL_INCREASE_DECAY", "APPLY_HUMP"),
        )

    if AlphaDiagnosis.HIGH_TURNOVER_LOW_RETURN in diags:
        rem = max(0, DEFAULT_REPAIR_BUDGETS[PriorityTier.P4] - repairs_used)
        return AlphaImprovementPlan(
            alpha_id, PriorityTier.P4, AlphaDisposition.TARGETED_REPAIR, objective,
            "HIGH_TURNOVER_LOW_RETURN", "APPLY_HUMP", ("ADD_TRADE_WHEN", "INCREASE_DECAY"),
            rem, ypm, ("COST_STRESS", "FITNESS", "MARGIN"), rem > 0, False, False,
        )

    # Step 9: Sharpe / Fitness Near-Miss (P1)
    if (1.05 <= sharpe < 1.25) or (0.80 <= fitness < 1.00):
        rem = max(0, DEFAULT_REPAIR_BUDGETS[PriorityTier.P1] - repairs_used)
        rec = actions[0].action if actions else "TEST_NEUTRALIZATION"
        alts = tuple(a.action for a in actions[1:3])
        return AlphaImprovementPlan(
            alpha_id, PriorityTier.P1, AlphaDisposition.TARGETED_REPAIR, objective,
            "SHARPE_OR_FITNESS_NEAR_MISS", rec, alts,
            rem, ypm, ("SHARPE", "FITNESS", "SELF_CORRELATION"), rem > 0, False, False,
        )

    # Step 10: Return/Vol decomposition (P3) when Sharpe < 1.25 or Fitness < 1.0
    if sharpe < 1.25 or fitness < 1.00:
        rem = max(0, DEFAULT_REPAIR_BUDGETS[PriorityTier.P3] - repairs_used)
        vol = record.get("volatility")
        if vol is not None and float(vol) > 0.18 and net_returns >= 0.08:
            rec = "TEST_NEUTRALIZATION"
            alts = ("APPLY_VOLATILITY_SCALING",)
            fg = "HIGH_VOLATILITY"
        else:
            rec = "ADD_ANALYST_NEWS_OR_FUNDAMENTAL"
            alts = ("CHANGE_ECONOMIC_HYPOTHESIS",)
            fg = "LOW_RETURN_LOW_INFORMATION"
        return AlphaImprovementPlan(
            alpha_id, PriorityTier.P3, AlphaDisposition.STRUCTURAL_RESEARCH, objective,
            fg, rec, alts, rem, ypm, ("RETURN", "SHARPE", "SUB_UNIVERSE"), rem > 0, False, False,
        )

    # Step 11: BRAIN_GOOD -> check novelty (Low novelty -> Portfolio Utility first; High novelty -> Robustness P5)
    if float(self_corr) >= 0.50 or novelty < 0.40:
        return AlphaImprovementPlan(
            alpha_id, PriorityTier.P2, AlphaDisposition.CORRELATION_REPAIR, objective,
            "GOOD_METRICS_LOW_NOVELTY", "EVALUATE_PORTFOLIO_MARGINAL_UTILITY", ("STRUCTURAL_ORTHOGONALIZATION",),
            DEFAULT_REPAIR_BUDGETS[PriorityTier.P2], ypm, ("PORTFOLIO_UTILITY", "SELF_CORRELATION"), True, False, False,
        )

    return AlphaImprovementPlan(
        alpha_id, PriorityTier.P5, AlphaDisposition.ROBUSTNESS, objective,
        "NONE", "RUN_ROBUSTNESS_SUITE", ("EVALUATE_PORTFOLIO_UTILITY",),
        DEFAULT_REPAIR_BUDGETS[PriorityTier.P5], ypm, ("TEMPORAL", "PARAMETER", "UNIVERSE", "COST_STRESS", "DSR"),
        True, True, bool(record.get("submit_ready", False)),
    )
