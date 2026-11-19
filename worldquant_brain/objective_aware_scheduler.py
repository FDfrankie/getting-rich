from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from worldquant_brain.optimization_policy import OptimizationObjective


@dataclass(frozen=True, slots=True)
class ObjectiveLaneBudget:
    return_lane: float = 0.30
    sharpe_lane: float = 0.30
    capacity_lane: float = 0.15
    balanced_lane: float = 0.25

    def validate(self) -> None:
        values = (self.return_lane, self.sharpe_lane, self.capacity_lane, self.balanced_lane)
        if any(value < 0 for value in values):
            raise ValueError("Lane budgets cannot be negative")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("Lane budgets must sum to 1.0")


@dataclass(frozen=True, slots=True)
class LaneCandidate:
    candidate_id: str
    objective: OptimizationObjective
    expected_net_return: float
    expected_sharpe: float
    expected_fitness: float
    expected_capacity: float
    novelty: float
    robustness_probability: float
    portfolio_utility: float
    information_gain: float
    correlation_risk: float
    complexity_penalty: float
    expected_minutes: float = 3.0


def expected_yield_per_minute(candidate: LaneCandidate) -> float:
    gross = (
        0.24 * candidate.expected_net_return
        + 0.22 * candidate.expected_sharpe
        + 0.18 * candidate.expected_fitness
        + 0.10 * candidate.robustness_probability
        + 0.08 * candidate.novelty
        + 0.08 * candidate.portfolio_utility
        + 0.06 * candidate.expected_capacity
        + 0.04 * candidate.information_gain
        - 0.15 * candidate.correlation_risk
        - 0.05 * candidate.complexity_penalty
    )
    return gross / max(candidate.expected_minutes, 0.1)


def allocate_lane_slots(total_slots: int, budget: ObjectiveLaneBudget) -> dict[OptimizationObjective, int]:
    budget.validate()
    if total_slots <= 0:
        return {key: 0 for key in (OptimizationObjective.RETURN, OptimizationObjective.SHARPE, OptimizationObjective.CAPACITY, OptimizationObjective.BALANCED)}
    raw = {
        OptimizationObjective.RETURN: total_slots * budget.return_lane,
        OptimizationObjective.SHARPE: total_slots * budget.sharpe_lane,
        OptimizationObjective.CAPACITY: total_slots * budget.capacity_lane,
        OptimizationObjective.BALANCED: total_slots * budget.balanced_lane,
    }
    slots = {key: int(value) for key, value in raw.items()}
    remainder = total_slots - sum(slots.values())
    for key, _ in sorted(raw.items(), key=lambda item: item[1] - int(item[1]), reverse=True)[:remainder]:
        slots[key] += 1
    return slots


def select_candidates(candidates: Sequence[LaneCandidate], *, total_slots: int, budget: ObjectiveLaneBudget | None = None) -> list[LaneCandidate]:
    budget = budget or ObjectiveLaneBudget()
    slots = allocate_lane_slots(total_slots, budget)
    selected: list[LaneCandidate] = []
    for objective, count in slots.items():
        lane = [item for item in candidates if item.objective == objective]
        lane.sort(key=expected_yield_per_minute, reverse=True)
        selected.extend(lane[:count])
    if len(selected) < total_slots:
        selected_ids = {item.candidate_id for item in selected}
        overflow = [item for item in candidates if item.candidate_id not in selected_ids]
        overflow.sort(key=expected_yield_per_minute, reverse=True)
        selected.extend(overflow[: total_slots - len(selected)])
    return sorted(selected, key=expected_yield_per_minute, reverse=True)
