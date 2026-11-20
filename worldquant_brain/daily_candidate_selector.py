"""Daily Top-K Utility selector with rolling 80/20 exploit/explore policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from worldquant_brain.candidate_utility import UtilityResult, evaluate_candidate


@dataclass(frozen=True)
class SelectedCandidate:
    alpha_id: str
    selection_type: str
    utility: float
    utility_result: UtilityResult
    candidate: Mapping[str, Any]


@dataclass(frozen=True)
class SelectionBatch:
    selected: tuple[SelectedCandidate, ...]
    retained_pending_ids: tuple[str, ...]
    rejected_ids: tuple[str, ...]


def should_use_exploration_slot(formal_submission_count: int) -> bool:
    """Every 5th formal submission (2 out of 10 = 20%) is allocated to exploration."""
    next_submission_number = formal_submission_count + 1
    return (next_submission_number % 5) == 0


def _is_pool_safe_for_reserve(candidate: Mapping[str, Any], result: UtilityResult) -> bool:
    """Ensure reserve_high_value=True never picks pool-degrading alphas."""
    if result.submission_utility <= 0.0:
        return False
    corr = candidate.get("self_correlation")
    if corr is not None:
        try:
            if float(corr) >= 0.50:
                return False
        except (TypeError, ValueError):
            pass
    return True


def select_daily_candidates(
    candidates: Iterable[Mapping[str, Any]],
    *,
    top_k: int = 2,
    formal_submission_count: int = 0,
    reserve_high_value: bool = False,
) -> SelectionBatch:
    if top_k < 0:
        raise ValueError("top_k must be >= 0")

    evaluated: list[tuple[Mapping[str, Any], UtilityResult]] = []
    rejected_ids: list[str] = []

    for candidate in candidates:
        alpha_id = str(
            candidate.get("alpha_id")
            or candidate.get("brain_alpha_id")
            or candidate.get("id")
            or ""
        )
        result = evaluate_candidate(candidate)
        if not result.eligible:
            rejected_ids.append(alpha_id)
            continue
        evaluated.append((candidate, result))

    # When reserve_high_value=True, prioritize positive-utility, low-correlation (< 0.50)
    # qualifying alphas in ascending utility order so top gems are saved for monetization
    # without ever intentionally submitting pool-degrading alphas.
    if reserve_high_value:
        pool_safe = [item for item in evaluated if _is_pool_safe_for_reserve(item[0], item[1])]
        pool_unsafe = [item for item in evaluated if not _is_pool_safe_for_reserve(item[0], item[1])]
        pool_safe_sorted = sorted(
            pool_safe,
            key=lambda item: (
                item[1].submission_utility,
                -item[1].novelty_score,
                -item[1].stability_score,
                str(item[0].get("created_at") or ""),
            ),
            reverse=False,
        )
        # Non-pool-safe items are pushed strictly to the back in descending utility order
        pool_unsafe_sorted = sorted(
            pool_unsafe,
            key=lambda item: (
                item[1].submission_utility,
                item[1].novelty_score,
                item[1].stability_score,
                str(item[0].get("created_at") or ""),
            ),
            reverse=True,
        )
        exploitation = pool_safe_sorted + pool_unsafe_sorted
    else:
        exploitation = sorted(
            evaluated,
            key=lambda item: (
                item[1].submission_utility,
                item[1].novelty_score,
                item[1].stability_score,
                str(item[0].get("created_at") or ""),
            ),
            reverse=True,
        )

    exploration = sorted(
        evaluated,
        key=lambda item: (
            item[1].exploration_score,
            item[1].novelty_score,
            item[1].submission_utility,
        ),
        reverse=True,
    )

    chosen_ids: set[str] = set()
    selected: list[SelectedCandidate] = []

    for slot in range(top_k):
        use_exploration = should_use_exploration_slot(formal_submission_count + slot)
        source = exploration if use_exploration else exploitation
        selection_type = "EXPLORE" if use_exploration else ("CHALLENGE_LOWEST_QUALIFYING" if reserve_high_value else "EXPLOIT")
        selected_item = None

        for candidate, utility_result in source:
            alpha_id = str(
                candidate.get("alpha_id")
                or candidate.get("brain_alpha_id")
                or candidate.get("id")
                or ""
            )
            if alpha_id in chosen_ids:
                continue
            if use_exploration:
                has_official_sim = bool(
                    candidate.get("brain_simulation_id")
                    or candidate.get("simulation_id")
                    or candidate.get("status") in {"BRAIN_COMPLETE", "SUBMISSION_CANDIDATE", "IS_CHECK_PASSED", "CORRELATION_PASSED"}
                )
                if utility_result.exploration_score < 0.50 or not has_official_sim:
                    continue
            selected_item = (candidate, utility_result, alpha_id)
            break

        if selected_item is None and use_exploration:
            for candidate, utility_result in exploitation:
                alpha_id = str(
                    candidate.get("alpha_id")
                    or candidate.get("brain_alpha_id")
                    or candidate.get("id")
                    or ""
                )
                if alpha_id in chosen_ids:
                    continue
                selected_item = (candidate, utility_result, alpha_id)
                selection_type = "EXPLOIT_FALLBACK"
                break

        if selected_item is None:
            break

        candidate, utility_result, alpha_id = selected_item
        chosen_ids.add(alpha_id)
        selected.append(
            SelectedCandidate(
                alpha_id=alpha_id,
                selection_type=selection_type,
                utility=utility_result.submission_utility,
                utility_result=utility_result,
                candidate=candidate,
            )
        )

    retained_pending = [
        str(c.get("alpha_id") or c.get("brain_alpha_id") or c.get("id") or "")
        for c, _ in evaluated
        if str(c.get("alpha_id") or c.get("brain_alpha_id") or c.get("id") or "") not in chosen_ids
    ]

    return SelectionBatch(
        selected=tuple(selected),
        retained_pending_ids=tuple(retained_pending),
        rejected_ids=tuple(rejected_ids),
    )
