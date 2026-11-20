"""Pareto Research Archive (Shadow-first Multi-Objective Frontier Engine).

Implements vector rewards, non-dominated sorting, crowding distance,
cohort isolation, capacity trimming, and Shadow-mode SQLite persistence.
"""
from __future__ import annotations

import hashlib
import json
import math

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Iterable, Sequence


class ObjectiveDirection(StrEnum):
    MAXIMIZE = "MAXIMIZE"
    MINIMIZE = "MINIMIZE"


class ObjectiveState(StrEnum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ArchiveScope(StrEnum):
    GLOBAL = "GLOBAL"
    OBJECTIVE = "OBJECTIVE"
    COHORT = "COHORT"
    FAMILY = "FAMILY"


class ArchiveStatus(StrEnum):
    PROVISIONAL = "PROVISIONAL"
    NON_DOMINATED = "NON_DOMINATED"
    DOMINATED = "DOMINATED"
    EVICTED = "EVICTED"
    PROMOTED = "PROMOTED"
    INVALIDATED = "INVALIDATED"


class ObjectiveLane(StrEnum):
    RETURN = "RETURN"
    SHARPE = "SHARPE"
    FITNESS = "FITNESS"
    CAPACITY = "CAPACITY"
    BALANCED = "BALANCED"


@dataclass(frozen=True, slots=True)
class ObjectiveDefinition:
    name: str
    direction: ObjectiveDirection
    epsilon: float
    required_for_promotion: bool


OBJECTIVE_DEFINITIONS: tuple[
    ObjectiveDefinition,
    ...,
] = (
    ObjectiveDefinition(
        name="official_quality",
        direction=ObjectiveDirection.MAXIMIZE,
        epsilon=0.01,
        required_for_promotion=True,
    ),
    ObjectiveDefinition(
        name="os_stability",
        direction=ObjectiveDirection.MAXIMIZE,
        epsilon=0.02,
        required_for_promotion=True,
    ),
    ObjectiveDefinition(
        name="perturbation_robustness",
        direction=ObjectiveDirection.MAXIMIZE,
        epsilon=0.02,
        required_for_promotion=False,
    ),
    ObjectiveDefinition(
        name="pnl_uniqueness",
        direction=ObjectiveDirection.MAXIMIZE,
        epsilon=0.02,
        required_for_promotion=True,
    ),
    ObjectiveDefinition(
        name="merged_portfolio_utility",
        direction=ObjectiveDirection.MAXIMIZE,
        epsilon=0.01,
        required_for_promotion=True,
    ),
    ObjectiveDefinition(
        name="cost_efficiency",
        direction=ObjectiveDirection.MAXIMIZE,
        epsilon=0.02,
        required_for_promotion=True,
    ),
    ObjectiveDefinition(
        name="capacity",
        direction=ObjectiveDirection.MAXIMIZE,
        epsilon=0.02,
        required_for_promotion=False,
    ),
    ObjectiveDefinition(
        name="information_gain",
        direction=ObjectiveDirection.MAXIMIZE,
        epsilon=0.02,
        required_for_promotion=False,
    ),
    ObjectiveDefinition(
        name="correlation_risk",
        direction=ObjectiveDirection.MINIMIZE,
        epsilon=0.02,
        required_for_promotion=True,
    ),
    ObjectiveDefinition(
        name="drawdown_risk",
        direction=ObjectiveDirection.MINIMIZE,
        epsilon=0.02,
        required_for_promotion=True,
    ),
    ObjectiveDefinition(
        name="overfitting_risk",
        direction=ObjectiveDirection.MINIMIZE,
        epsilon=0.02,
        required_for_promotion=True,
    ),
    ObjectiveDefinition(
        name="data_quality_risk",
        direction=ObjectiveDirection.MINIMIZE,
        epsilon=0.02,
        required_for_promotion=True,
    ),
    ObjectiveDefinition(
        name="complexity_cost",
        direction=ObjectiveDirection.MINIMIZE,
        epsilon=0.02,
        required_for_promotion=False,
    ),
    ObjectiveDefinition(
        name="simulation_minutes",
        direction=ObjectiveDirection.MINIMIZE,
        epsilon=0.10,
        required_for_promotion=False,
    ),
)

OBJECTIVE_BY_NAME = {
    definition.name: definition
    for definition in OBJECTIVE_DEFINITIONS
}

# 2. Objective Value
@dataclass(frozen=True, slots=True)
class ObjectiveValue:
    value: float | None
    state: ObjectiveState
    evidence_source: str
    calculated_at: str

    @classmethod
    def known(
        cls,
        value: float,
        *,
        evidence_source: str,
        calculated_at: str | None = None,
    ) -> "ObjectiveValue":
        number = float(value)

        if not math.isfinite(number):
            raise ValueError(
                "Known Pareto objective must be finite"
            )

        return cls(
            value=number,
            state=ObjectiveState.KNOWN,
            evidence_source=evidence_source,
            calculated_at=(
                calculated_at
                or datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        )

    @classmethod
    def unknown(
        cls,
        *,
        evidence_source: str = "UNKNOWN",
        calculated_at: str | None = None,
    ) -> "ObjectiveValue":
        return cls(
            value=None,
            state=ObjectiveState.UNKNOWN,
            evidence_source=evidence_source,
            calculated_at=(
                calculated_at
                or datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        )

    @classmethod
    def not_applicable(
        cls,
        *,
        calculated_at: str | None = None,
    ) -> "ObjectiveValue":
        return cls(
            value=None,
            state=ObjectiveState.NOT_APPLICABLE,
            evidence_source="NOT_APPLICABLE",
            calculated_at=(
                calculated_at
                or datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        )

# 3. Objective Vector
@dataclass(frozen=True, slots=True)
class ParetoObjectiveVector:
    official_quality: ObjectiveValue
    os_stability: ObjectiveValue
    perturbation_robustness: ObjectiveValue
    pnl_uniqueness: ObjectiveValue
    merged_portfolio_utility: ObjectiveValue
    cost_efficiency: ObjectiveValue
    capacity: ObjectiveValue
    information_gain: ObjectiveValue

    correlation_risk: ObjectiveValue
    drawdown_risk: ObjectiveValue
    overfitting_risk: ObjectiveValue
    data_quality_risk: ObjectiveValue
    complexity_cost: ObjectiveValue
    simulation_minutes: ObjectiveValue

    def values(
        self,
    ) -> dict[str, ObjectiveValue]:
        return {
            definition.name: getattr(
                self,
                definition.name,
            )
            for definition in OBJECTIVE_DEFINITIONS
        }

    def known_mask(self) -> int:
        mask = 0

        for index, definition in enumerate(
            OBJECTIVE_DEFINITIONS
        ):
            objective = getattr(
                self,
                definition.name,
            )

            if objective.state == ObjectiveState.KNOWN:
                mask |= 1 << index

        return mask

    def promotion_ready(self) -> bool:
        for definition in OBJECTIVE_DEFINITIONS:
            if not definition.required_for_promotion:
                continue

            objective = getattr(
                self,
                definition.name,
            )

            if objective.state != ObjectiveState.KNOWN:
                return False

        return True

    def validate(self) -> tuple[str, ...]:
        errors: list[str] = []

        for definition in OBJECTIVE_DEFINITIONS:
            objective = getattr(
                self,
                definition.name,
            )

            if (
                objective.state == ObjectiveState.KNOWN
                and objective.value is None
            ):
                errors.append(
                    f"{definition.name}:KNOWN_WITHOUT_VALUE"
                )

            if (
                objective.value is not None
                and not math.isfinite(objective.value)
            ):
                errors.append(
                    f"{definition.name}:NON_FINITE"
                )

        return tuple(errors)

    def canonical_payload(
        self,
    ) -> dict[str, Any]:
        return {
            name: {
                "value": objective.value,
                "state": objective.state.value,
                "evidence_source": (
                    objective.evidence_source
                ),
                "calculated_at": (
                    objective.calculated_at
                ),
            }
            for (
                name,
                objective,
            ) in self.values().items()
        }

    def sha256(self) -> str:
        payload = json.dumps(
            self.canonical_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

        return hashlib.sha256(
            payload
        ).hexdigest()

# 4. Context 與 Candidate
@dataclass(frozen=True, slots=True)
class ParetoContext:
    region: str
    delay: int
    universe: str
    data_domain: str
    blueprint_family: str
    objective_lane: ObjectiveLane

    def validate(self) -> None:
        if self.delay not in (0, 1):
            raise ValueError(
                f"Unsupported delay: {self.delay}"
            )

    def cohort_key(self) -> str:
        self.validate()

        return ":".join(
            (
                self.region,
                f"D{self.delay}",
                self.universe,
                self.data_domain,
            )
        )

    def family_key(self) -> str:
        return ":".join(
            (
                self.cohort_key(),
                self.blueprint_family,
            )
        )


@dataclass(frozen=True, slots=True)
class ParetoCandidate:
    trial_id: str
    artifact_id: str
    candidate_id: str

    policy_version: str
    objective_schema_version: str
    normalization_version: str

    context: ParetoContext
    objectives: ParetoObjectiveVector

    input_snapshot_hash: str
    admitted_at: str

    def validate(self) -> tuple[str, ...]:
        errors = list(
            self.objectives.validate()
        )

        try:
            self.context.validate()
        except ValueError as exc:
            errors.append(str(exc))

        if not self.trial_id:
            errors.append("MISSING_TRIAL_ID")

        if not self.artifact_id:
            errors.append(
                "MISSING_ARTIFACT_ID"
            )

        if not self.input_snapshot_hash:
            errors.append(
                "MISSING_INPUT_SNAPSHOT_HASH"
            )

        return tuple(errors)

# 5. Archive Entry
@dataclass(slots=True)
class ParetoArchiveEntry:
    archive_entry_id: str
    candidate: ParetoCandidate
    archive_scope: ArchiveScope
    archive_scope_key: str

    archive_status: ArchiveStatus

    pareto_rank: int | None = None
    crowding_distance: float = 0.0

    dominated_by_entry_ids: list[str] = field(
        default_factory=list
    )

    dominates_entry_ids: list[str] = field(
        default_factory=list
    )

    last_recomputed_at: str | None = None

    @property
    def promotion_ready(self) -> bool:
        return (
            self.candidate
            .objectives
            .promotion_ready()
        )

    @property
    def objective_vector_hash(self) -> str:
        return (
            self.candidate
            .objectives
            .sha256()
        )

# 1. Stable ID
def stable_id(
    prefix: str,
    *parts: Any,
) -> str:
    payload = json.dumps(
        parts,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    digest = hashlib.sha256(
        payload
    ).hexdigest()[:32]

    return f"{prefix}_{digest}"

# 2. Epsilon Dominance
def _known_pair(
    left: ObjectiveValue,
    right: ObjectiveValue,
) -> tuple[float, float] | None:
    if (
        left.state != ObjectiveState.KNOWN
        or right.state != ObjectiveState.KNOWN
        or left.value is None
        or right.value is None
    ):
        return None

    return left.value, right.value


def dominates(
    candidate_a: ParetoCandidate,
    candidate_b: ParetoCandidate,
    *,
    require_same_cohort: bool = True,
) -> bool:
    if require_same_cohort:
        if (
            candidate_a.context.cohort_key()
            != candidate_b.context.cohort_key()
        ):
            return False

    no_worse = True
    strictly_better = False
    comparable_count = 0

    for definition in OBJECTIVE_DEFINITIONS:
        left = getattr(
            candidate_a.objectives,
            definition.name,
        )

        right = getattr(
            candidate_b.objectives,
            definition.name,
        )

        pair = _known_pair(
            left,
            right,
        )

        if pair is None:
            continue

        comparable_count += 1

        left_value, right_value = pair
        epsilon = definition.epsilon

        if (
            definition.direction
            == ObjectiveDirection.MAXIMIZE
        ):
            if (
                left_value
                < right_value - epsilon
            ):
                no_worse = False
                break

            if (
                left_value
                > right_value + epsilon
            ):
                strictly_better = True

        else:
            if (
                left_value
                > right_value + epsilon
            ):
                no_worse = False
                break

            if (
                left_value
                < right_value - epsilon
            ):
                strictly_better = True

    if comparable_count == 0:
        return False

    return (
        no_worse
        and strictly_better
    )

# 3. Non-dominated Sorting
def non_dominated_sort(
    entries: Sequence[
        ParetoArchiveEntry
    ],
) -> list[list[ParetoArchiveEntry]]:
    if not entries:
        return []

    domination_counts: dict[str, int] = {
        entry.archive_entry_id: 0
        for entry in entries
    }

    dominated_sets: dict[
        str,
        list[ParetoArchiveEntry],
    ] = {
        entry.archive_entry_id: []
        for entry in entries
    }

    for left in entries:
        left.dominated_by_entry_ids.clear()
        left.dominates_entry_ids.clear()

        for right in entries:
            if (
                left.archive_entry_id
                == right.archive_entry_id
            ):
                continue

            if dominates(
                left.candidate,
                right.candidate,
            ):
                dominated_sets[
                    left.archive_entry_id
                ].append(right)

                left.dominates_entry_ids.append(
                    right.archive_entry_id
                )

            elif dominates(
                right.candidate,
                left.candidate,
            ):
                domination_counts[
                    left.archive_entry_id
                ] += 1

                left.dominated_by_entry_ids.append(
                    right.archive_entry_id
                )

    first_front = [
        entry
        for entry in entries
        if domination_counts[
            entry.archive_entry_id
        ] == 0
    ]

    fronts: list[
        list[ParetoArchiveEntry]
    ] = []

    if first_front:
        fronts.append(first_front)

    front_index = 0

    while front_index < len(fronts):
        next_front: list[
            ParetoArchiveEntry
        ] = []

        for entry in fronts[front_index]:
            for dominated in dominated_sets[
                entry.archive_entry_id
            ]:
                domination_counts[
                    dominated.archive_entry_id
                ] -= 1

                if (
                    domination_counts[
                        dominated.archive_entry_id
                    ]
                    == 0
                ):
                    next_front.append(
                        dominated
                    )

        if next_front:
            unique = {
                entry.archive_entry_id: entry
                for entry in next_front
            }

            fronts.append(
                list(unique.values())
            )

        front_index += 1

    now = datetime.now(
        timezone.utc
    ).isoformat()

    for rank, front in enumerate(
        fronts,
        start=1,
    ):
        for entry in front:
            entry.pareto_rank = rank
            entry.archive_status = (
                ArchiveStatus.NON_DOMINATED
                if rank == 1
                else ArchiveStatus.DOMINATED
            )

            entry.last_recomputed_at = now

    return fronts

# 4. Crowding Distance
def assign_crowding_distance(
    front: Sequence[
        ParetoArchiveEntry
    ],
) -> None:
    if not front:
        return

    for entry in front:
        entry.crowding_distance = 0.0

    if len(front) <= 2:
        for entry in front:
            entry.crowding_distance = (
                float("inf")
            )

        return

    for definition in OBJECTIVE_DEFINITIONS:
        known_entries = [
            entry
            for entry in front
            if (
                getattr(
                    entry.candidate.objectives,
                    definition.name,
                ).state
                == ObjectiveState.KNOWN
            )
        ]

        if len(known_entries) <= 2:
            for entry in known_entries:
                entry.crowding_distance = (
                    float("inf")
                )

            continue

        reverse = (
            definition.direction
            == ObjectiveDirection.MAXIMIZE
        )

        known_entries.sort(
            key=lambda entry: (
                getattr(
                    entry.candidate.objectives,
                    definition.name,
                ).value
            ),
            reverse=reverse,
        )

        first = known_entries[0]
        last = known_entries[-1]

        first.crowding_distance = (
            float("inf")
        )

        last.crowding_distance = (
            float("inf")
        )

        values = [
            getattr(
                entry.candidate.objectives,
                definition.name,
            ).value
            for entry in known_entries
        ]

        finite_values = [
            float(value)
            for value in values
            if value is not None
        ]

        objective_min = min(
            finite_values
        )

        objective_max = max(
            finite_values
        )

        objective_range = (
            objective_max
            - objective_min
        )

        if objective_range <= 1e-12:
            continue

        for index in range(
            1,
            len(known_entries) - 1,
        ):
            if math.isinf(
                known_entries[
                    index
                ].crowding_distance
            ):
                continue

            previous_value = getattr(
                known_entries[
                    index - 1
                ].candidate.objectives,
                definition.name,
            ).value

            next_value = getattr(
                known_entries[
                    index + 1
                ].candidate.objectives,
                definition.name,
            ).value

            if (
                previous_value is None
                or next_value is None
            ):
                continue

            distance = (
                abs(
                    next_value
                    - previous_value
                )
                / objective_range
            )

            known_entries[
                index
            ].crowding_distance += distance

# 5. Archive Capacity
@dataclass(frozen=True, slots=True)
class ArchiveCapacityPolicy:
    global_capacity: int = 100
    objective_capacity: int = 50
    cohort_capacity: int = 25
    family_capacity: int = 10

    def capacity_for(
        self,
        scope: ArchiveScope,
    ) -> int:
        mapping = {
            ArchiveScope.GLOBAL: (
                self.global_capacity
            ),
            ArchiveScope.OBJECTIVE: (
                self.objective_capacity
            ),
            ArchiveScope.COHORT: (
                self.cohort_capacity
            ),
            ArchiveScope.FAMILY: (
                self.family_capacity
            ),
        }

        return mapping[scope]

# 6. Archive Eviction
def trim_archive(
    entries: Sequence[
        ParetoArchiveEntry
    ],
    *,
    capacity: int,
) -> tuple[
    list[ParetoArchiveEntry],
    list[ParetoArchiveEntry],
]:
    if capacity <= 0:
        raise ValueError(
            "Archive capacity must be positive"
        )

    fronts = non_dominated_sort(
        entries
    )

    retained: list[
        ParetoArchiveEntry
    ] = []

    evicted: list[
        ParetoArchiveEntry
    ] = []

    for front in fronts:
        assign_crowding_distance(front)

        remaining = (
            capacity
            - len(retained)
        )

        if remaining <= 0:
            evicted.extend(front)
            continue

        if len(front) <= remaining:
            retained.extend(front)
            continue

        ranked_front = sorted(
            front,
            key=lambda entry: (
                entry.promotion_ready,
                entry.crowding_distance,
                -(
                    entry.pareto_rank
                    or 999999
                ),
            ),
            reverse=True,
        )

        retained.extend(
            ranked_front[:remaining]
        )

        evicted.extend(
            ranked_front[remaining:]
        )

    for entry in retained:
        if entry.pareto_rank == 1:
            entry.archive_status = (
                ArchiveStatus.NON_DOMINATED
            )

    for entry in evicted:
        entry.archive_status = (
            ArchiveStatus.EVICTED
        )

    return retained, evicted

@dataclass(slots=True)
class ParetoArchive:
    scope: ArchiveScope
    scope_key: str

    capacity_policy: (
        ArchiveCapacityPolicy
    ) = field(
        default_factory=(
            ArchiveCapacityPolicy
        )
    )

    entries: dict[
        str,
        ParetoArchiveEntry,
    ] = field(
        default_factory=dict
    )

    def admit(
        self,
        candidate: ParetoCandidate,
    ) -> ParetoArchiveEntry:
        errors = candidate.validate()

        if errors:
            raise ValueError(
                "Invalid Pareto candidate: "
                + ", ".join(errors)
            )

        archive_entry_id = stable_id(
            "pareto",
            candidate.trial_id,
            candidate.policy_version,
            self.scope.value,
            self.scope_key,
        )

        existing = self.entries.get(
            archive_entry_id
        )

        if existing is not None:
            if (
                existing.objective_vector_hash
                != candidate.objectives.sha256()
            ):
                raise ValueError(
                    "Idempotency conflict: "
                    "same archive key with "
                    "different objective vector"
                )

            return existing

        entry = ParetoArchiveEntry(
            archive_entry_id=archive_entry_id,
            candidate=candidate,
            archive_scope=self.scope,
            archive_scope_key=self.scope_key,
            archive_status=(
                ArchiveStatus.PROVISIONAL
            ),
        )

        self.entries[
            archive_entry_id
        ] = entry

        self.recompute()

        return entry

    def recompute(self) -> None:
        values = list(
            self.entries.values()
        )

        capacity = (
            self.capacity_policy
            .capacity_for(self.scope)
        )

        retained, evicted = trim_archive(
            values,
            capacity=capacity,
        )

        retained_ids = {
            entry.archive_entry_id
            for entry in retained
        }

        for entry_id in list(
            self.entries
        ):
            if entry_id not in retained_ids:
                self.entries[
                    entry_id
                ].archive_status = (
                    ArchiveStatus.EVICTED
                )

    def active_entries(
        self,
    ) -> list[ParetoArchiveEntry]:
        return [
            entry
            for entry in self.entries.values()
            if (
                entry.archive_status
                != ArchiveStatus.EVICTED
            )
        ]

    def first_front(
        self,
    ) -> list[ParetoArchiveEntry]:
        return sorted(
            (
                entry
                for entry
                in self.active_entries()
                if entry.pareto_rank == 1
            ),
            key=lambda entry: (
                entry.crowding_distance
            ),
            reverse=True,
        )



# ---------------------------------------------------------------------------
# Shadow-mode SQLite Persistence Layer (011_pareto_archive.sql)
# ---------------------------------------------------------------------------
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class ParetoArchiveShadowStore:
    """Persists Pareto Archive entries and snapshots in Shadow Mode (never overrides official gates)."""

    def __init__(self, db_path: str | Path, shadow_mode: bool = True) -> None:
        self.db_path = str(db_path)
        self.shadow_mode = bool(shadow_mode)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def persist_archive(self, archive: ParetoArchive, *, alpha_pool_hash: str = "shadow-pool") -> int:
        """Persist current archive entries idempotently into pareto_archive_entries and record a snapshot."""
        now = datetime.now(timezone.utc).isoformat()
        entries = list(archive.entries.values())
        with self._connect() as conn:
            for e in entries:
                cand = e.candidate
                vec = cand.objectives
                ctx = cand.context
                vals = vec.values()
                cd = None if (e.crowding_distance is None or not math.isfinite(e.crowding_distance)) else float(e.crowding_distance)
                conn.execute(
                    """
                    INSERT INTO pareto_archive_entries (
                        archive_entry_id, trial_id, artifact_id, candidate_id,
                        policy_version, objective_schema_version, normalization_version,
                        region, delay, universe, data_domain, blueprint_family, objective_lane,
                        official_quality, os_stability, perturbation_robustness, pnl_uniqueness,
                        merged_portfolio_utility, cost_efficiency, capacity, information_gain,
                        correlation_risk, drawdown_risk, overfitting_risk, data_quality_risk,
                        complexity_cost, simulation_minutes,
                        objective_known_mask, pareto_rank, crowding_distance,
                        archive_scope, archive_scope_key, archive_status, dominated_by_entry_id,
                        admitted_at, last_recomputed_at, input_snapshot_hash, objective_vector_hash
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    ON CONFLICT(archive_entry_id) DO UPDATE SET
                        pareto_rank = excluded.pareto_rank,
                        crowding_distance = excluded.crowding_distance,
                        archive_status = excluded.archive_status,
                        dominated_by_entry_id = excluded.dominated_by_entry_id,
                        last_recomputed_at = excluded.last_recomputed_at
                    """,
                    (
                        e.archive_entry_id,
                        cand.trial_id,
                        cand.artifact_id,
                        cand.candidate_id,
                        cand.policy_version,
                        cand.objective_schema_version,
                        cand.normalization_version,
                        ctx.region,
                        ctx.delay,
                        ctx.universe,
                        ctx.data_domain,
                        ctx.blueprint_family,
                        ctx.objective_lane.value,
                        vals["official_quality"].value,
                        vals["os_stability"].value,
                        vals["perturbation_robustness"].value,
                        vals["pnl_uniqueness"].value,
                        vals["merged_portfolio_utility"].value,
                        vals["cost_efficiency"].value,
                        vals["capacity"].value,
                        vals["information_gain"].value,
                        vals["correlation_risk"].value,
                        vals["drawdown_risk"].value,
                        vals["overfitting_risk"].value,
                        vals["data_quality_risk"].value,
                        vals["complexity_cost"].value,
                        vals["simulation_minutes"].value,
                        vec.known_mask(),
                        e.pareto_rank,
                        cd,
                        e.archive_scope.value,
                        e.archive_scope_key,
                        e.archive_status.value,
                        (e.dominated_by_entry_ids[0] if e.dominated_by_entry_ids else None),
                        cand.admitted_at,
                        now,
                        cand.input_snapshot_hash,
                        e.objective_vector_hash,
                    ),
                )
            active = archive.active_entries()
            front1 = archive.first_front()
            ranks = {x.pareto_rank for x in active if x.pareto_rank is not None}
            policy_ver = entries[0].candidate.policy_version if entries else "shadow-v1"
            snap_id = stable_id("psnap", archive.scope.value, archive.scope_key, now, str(len(entries)))
            conn.execute(
                """
                INSERT OR IGNORE INTO pareto_archive_snapshots (
                    snapshot_id, policy_version, archive_scope, archive_scope_key,
                    alpha_pool_hash, entry_count, front_count, non_dominated_count,
                    snapshot_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snap_id,
                    policy_ver,
                    archive.scope.value,
                    archive.scope_key,
                    alpha_pool_hash,
                    len(active),
                    len(ranks),
                    len(front1),
                    json.dumps({"shadow_mode": self.shadow_mode, "first_front_ids": [x.archive_entry_id for x in front1]}),
                    now,
                ),
            )
        return len(entries)
