"""Forbidden Edit Motif Memory (Shadow/Record-only + Confidence-Gated Veto Policy).

Tracks context-specific AST edit motifs, gate-specific repair outcomes,
Beta-Binomial posteriors, Wilson lower bounds, and asymmetric veto levels.
"""
from __future__ import annotations
from worldquant_brain.pareto_archive import stable_id

import hashlib
import json
import math

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping, Sequence


class EditMotifType(StrEnum):
    WRAP_OPERATOR = "WRAP_OPERATOR"
    UNWRAP_OPERATOR = "UNWRAP_OPERATOR"
    REPLACE_OPERATOR = "REPLACE_OPERATOR"
    REPLACE_FIELD = "REPLACE_FIELD"

    ADD_CONFIRMATION = "ADD_CONFIRMATION"
    REMOVE_CONFIRMATION = "REMOVE_CONFIRMATION"

    ADD_CONDITION = "ADD_CONDITION"
    REMOVE_CONDITION = "REMOVE_CONDITION"

    CHANGE_WINDOW = "CHANGE_WINDOW"
    CHANGE_GROUP = "CHANGE_GROUP"
    CHANGE_NEUTRALIZATION = (
        "CHANGE_NEUTRALIZATION"
    )
    CHANGE_DECAY = "CHANGE_DECAY"
    CHANGE_TRUNCATION = "CHANGE_TRUNCATION"

    ADD_BACKFILL = "ADD_BACKFILL"
    CHANGE_BACKFILL = "CHANGE_BACKFILL"
    REMOVE_BACKFILL = "REMOVE_BACKFILL"

    ADD_RANK = "ADD_RANK"
    REMOVE_RANK = "REMOVE_RANK"

    ADD_WINSORIZE = "ADD_WINSORIZE"
    CHANGE_WINSORIZE = "CHANGE_WINSORIZE"

    ADD_HUMP = "ADD_HUMP"
    CHANGE_HUMP = "CHANGE_HUMP"

    ADD_TRADE_WHEN = "ADD_TRADE_WHEN"
    CHANGE_TRADE_WHEN = (
        "CHANGE_TRADE_WHEN"
    )

    VECTOR_REDUCER_CHANGE = (
        "VECTOR_REDUCER_CHANGE"
    )

    STRUCTURAL_REWRITE = (
        "STRUCTURAL_REWRITE"
    )

    SETTING_ONLY_CHANGE = (
        "SETTING_ONLY_CHANGE"
    )


class VetoLevel(StrEnum):
    NONE = "NONE"
    COOLDOWN = "COOLDOWN"
    SOFT = "SOFT"
    STRONG = "STRONG"
    PERMANENT = "PERMANENT"


class MotifDecisionAction(StrEnum):
    ALLOW = "ALLOW"
    LOWER_PRIORITY = "LOWER_PRIORITY"
    COOLDOWN = "COOLDOWN"
    ALLOW_EXPLORATION = (
        "ALLOW_EXPLORATION"
    )
    REJECT = "REJECT"


@dataclass(frozen=True, slots=True)
class EditMotif:
    motif_type: EditMotifType

    target_ast_path: str | None

    before_subtree_hash: str | None
    after_subtree_hash: str | None

    added_operators: tuple[str, ...]
    removed_operators: tuple[str, ...]

    changed_fields: tuple[str, ...]
    changed_literals: tuple[
        tuple[str, float | str, float | str],
        ...,
    ]

    setting_changes: Mapping[
        str,
        tuple[Any, Any],
    ]

    structural_change: bool
    parameter_only: bool

    ast_schema_version: str

    def validate(self) -> None:
        if (
            self.structural_change
            and self.parameter_only
        ):
            raise ValueError(
                "Motif cannot be both "
                "structural and parameter-only"
            )

    def canonical_payload(
        self,
    ) -> dict[str, Any]:
        self.validate()

        return {
            "motif_type": self.motif_type.value,
            "target_ast_path": (
                self.target_ast_path
            ),
            "before_subtree_hash": (
                self.before_subtree_hash
            ),
            "after_subtree_hash": (
                self.after_subtree_hash
            ),
            "added_operators": sorted(
                self.added_operators
            ),
            "removed_operators": sorted(
                self.removed_operators
            ),
            "changed_fields": sorted(
                self.changed_fields
            ),
            "changed_literals": sorted(
                self.changed_literals,
                key=lambda item: item[0],
            ),
            "setting_changes": {
                key: self.setting_changes[key]
                for key in sorted(
                    self.setting_changes
                )
            },
            "structural_change": (
                self.structural_change
            ),
            "parameter_only": (
                self.parameter_only
            ),
            "ast_schema_version": (
                self.ast_schema_version
            ),
        }

    def motif_hash(self) -> str:
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

    def motif_id(self) -> str:
        return (
            "motif_"
            + self.motif_hash()[:32]
        )

@dataclass(frozen=True, slots=True)
class MotifContext:
    hypothesis_family: str
    data_domain: str
    failure_gate: str

    region: str
    universe: str
    delay: int

    objective_lane: ObjectiveLane

    parent_turnover_band: str | None
    parent_correlation_band: str | None
    parent_coverage_band: str | None

    parent_ast_signature: str

    field_catalog_version: str | None
    operator_schema_version: str | None

    def validate(self) -> None:
        if self.delay not in (0, 1):
            raise ValueError(
                f"Unsupported delay: {self.delay}"
            )

    def canonical_payload(
        self,
    ) -> dict[str, Any]:
        self.validate()

        return {
            **asdict(self),
            "objective_lane": (
                self.objective_lane.value
            ),
        }

    def context_hash(self) -> str:
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

    def motif_context_id(
        self,
        motif: EditMotif,
    ) -> str:
        payload = (
            motif.motif_hash(),
            self.context_hash(),
        )

        return stable_id(
            "motif_ctx",
            *payload,
        )


@dataclass(frozen=True, slots=True)
class MotifPolicy:
    motif_id: str
    motif_context_id: str
    policy_version: str

    attempts: int
    successes: int
    failures: int

    distinct_parent_count: int
    distinct_batch_count: int

    posterior_failure_probability: float
    wilson_lower_bound: float

    veto_level: VetoLevel

    prior_multiplier: float
    exploration_probability: float

    cooldown_until: str | None

    def validate(self) -> None:
        if (
            self.successes
            + self.failures
            != self.attempts
        ):
            raise ValueError(
                "Successes + failures "
                "must equal attempts"
            )

        for value in (
            self.posterior_failure_probability,
            self.wilson_lower_bound,
            self.prior_multiplier,
            self.exploration_probability,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    "Motif probability outside 0..1"
                )

def wilson_lower_bound(
    failures: int,
    attempts: int,
    *,
    z: float = 1.96,
) -> float:
    if attempts <= 0:
        return 0.0

    proportion = failures / attempts

    denominator = (
        1.0
        + z * z / attempts
    )

    centre = (
        proportion
        + z * z / (2.0 * attempts)
    )

    adjustment = z * math.sqrt(
        (
            proportion
            * (1.0 - proportion)
            + z * z / (4.0 * attempts)
        )
        / attempts
    )

    lower = (
        centre
        - adjustment
    ) / denominator

    return max(
        0.0,
        min(1.0, lower),
    )

PERMANENT_VETO_REASONS = {
    "LOOK_AHEAD",
    "INVALID_OPERATOR",
    "VECTOR_WITHOUT_REDUCER",
    "UNSUPPORTED_SETTING",
    "PARSE_FAILURE",
    "PROHIBITED_STRUCTURE",
}


def determine_veto_level(
    *,
    attempts: int,
    failures: int,
    distinct_parent_count: int,
    distinct_batch_count: int,
    permanent_invalid_reason: str | None,
) -> tuple[
    VetoLevel,
    float,
    float,
]:
    if (
        permanent_invalid_reason
        in PERMANENT_VETO_REASONS
    ):
        return (
            VetoLevel.PERMANENT,
            0.0,
            0.0,
        )

    if attempts <= 0:
        return (
            VetoLevel.NONE,
            1.0,
            0.15,
        )

    failure_rate = (
        failures / attempts
    )

    lower_bound = wilson_lower_bound(
        failures,
        attempts,
    )

    if (
        attempts >= 10
        and failure_rate >= 0.90
        and lower_bound >= 0.75
        and distinct_parent_count >= 3
        and distinct_batch_count >= 3
    ):
        return (
            VetoLevel.STRONG,
            0.05,
            0.03,
        )

    if (
        attempts >= 5
        and failure_rate >= 0.80
        and lower_bound >= 0.60
        and distinct_parent_count >= 2
        and distinct_batch_count >= 2
    ):
        return (
            VetoLevel.SOFT,
            0.20,
            0.15,
        )

    if (
        attempts >= 3
        and failure_rate >= 0.80
    ):
        return (
            VetoLevel.COOLDOWN,
            0.50,
            0.20,
        )

    return (
        VetoLevel.NONE,
        1.0,
        0.15,
    )

def posterior_failure_probability(
    *,
    failures: int,
    attempts: int,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> float:
    if attempts < 0:
        raise ValueError(
            "Attempts cannot be negative"
        )

    if failures < 0 or failures > attempts:
        raise ValueError(
            "Failures outside valid range"
        )

    return (
        failures + alpha_prior
    ) / (
        attempts
        + alpha_prior
        + beta_prior
    )

@dataclass(frozen=True, slots=True)
class MotifDecision:
    action: MotifDecisionAction

    veto_level: VetoLevel

    prior_before: float
    prior_after: float

    exploration_probability: float

    reason: str


def decide_motif_action(
    policy: MotifPolicy | None,
    *,
    base_prior: float,
    exploration_draw: float | None = None,
) -> MotifDecision:
    if not 0.0 <= base_prior <= 1.0:
        raise ValueError(
            "Base prior outside 0..1"
        )

    if policy is None:
        return MotifDecision(
            action=MotifDecisionAction.ALLOW,
            veto_level=VetoLevel.NONE,
            prior_before=base_prior,
            prior_after=base_prior,
            exploration_probability=0.15,
            reason="NO_MOTIF_POLICY",
        )

    policy.validate()

    adjusted_prior = (
        base_prior
        * policy.prior_multiplier
    )

    if policy.veto_level == VetoLevel.PERMANENT:
        return MotifDecision(
            action=MotifDecisionAction.REJECT,
            veto_level=policy.veto_level,
            prior_before=base_prior,
            prior_after=0.0,
            exploration_probability=0.0,
            reason="PERMANENT_INVALID_PATTERN",
        )

    if policy.veto_level == VetoLevel.STRONG:
        explore = (
            exploration_draw is not None
            and exploration_draw
            < policy.exploration_probability
        )

        return MotifDecision(
            action=(
                MotifDecisionAction
                .ALLOW_EXPLORATION
                if explore
                else MotifDecisionAction.REJECT
            ),
            veto_level=policy.veto_level,
            prior_before=base_prior,
            prior_after=(
                adjusted_prior
                if explore
                else 0.0
            ),
            exploration_probability=(
                policy.exploration_probability
            ),
            reason=(
                "STRONG_VETO_EXPLORATION"
                if explore
                else "STRONG_VETO"
            ),
        )

    if policy.veto_level == VetoLevel.SOFT:
        return MotifDecision(
            action=(
                MotifDecisionAction
                .LOWER_PRIORITY
            ),
            veto_level=policy.veto_level,
            prior_before=base_prior,
            prior_after=adjusted_prior,
            exploration_probability=(
                policy.exploration_probability
            ),
            reason="SOFT_VETO",
        )

    if policy.veto_level == VetoLevel.COOLDOWN:
        return MotifDecision(
            action=MotifDecisionAction.COOLDOWN,
            veto_level=policy.veto_level,
            prior_before=base_prior,
            prior_after=adjusted_prior,
            exploration_probability=(
                policy.exploration_probability
            ),
            reason="MOTIF_COOLDOWN",
        )

    return MotifDecision(
        action=MotifDecisionAction.ALLOW,
        veto_level=VetoLevel.NONE,
        prior_before=base_prior,
        prior_after=base_prior,
        exploration_probability=(
            policy.exploration_probability
        ),
        reason="MOTIF_ALLOWED",
    )

def motif_repair_succeeded(
    *,
    failure_gate: str,

    before_sharpe: float | None,
    after_sharpe: float | None,

    before_fitness: float | None,
    after_fitness: float | None,

    before_turnover: float | None,
    after_turnover: float | None,

    before_correlation: float | None,
    after_correlation: float | None,

    before_sub_universe_sharpe: float | None,
    after_sub_universe_sharpe: float | None,

    after_sub_universe_passed: bool | None,

    before_margin: float | None,
    after_margin: float | None,
) -> bool:
    gate = failure_gate.upper()

    if gate == "SUB_UNIVERSE_FAILURE":
        return bool(
            after_sub_universe_passed is True
            and after_sharpe is not None
            and before_sharpe is not None
            and after_sharpe
                >= before_sharpe - 0.10
            and after_fitness is not None
            and after_fitness > 1.0
            and after_turnover is not None
            and 0.01 < after_turnover < 0.70
        )

    if gate == "HIGH_SELF_CORRELATION":
        return bool(
            before_correlation is not None
            and after_correlation is not None
            and after_correlation
                < before_correlation
            and after_correlation < 0.70
            and before_sharpe is not None
            and after_sharpe is not None
            and after_sharpe
                >= before_sharpe - 0.10
            and after_fitness is not None
            and after_fitness > 1.0
        )

    if gate == "HIGH_TURNOVER":
        return bool(
            before_turnover is not None
            and after_turnover is not None
            and after_turnover < before_turnover
            and before_sharpe is not None
            and after_sharpe is not None
            and after_sharpe
                >= before_sharpe - 0.10
            and before_margin is not None
            and after_margin is not None
            and after_margin >= before_margin
        )

    if gate == "LOW_SHARPE":
        return bool(
            before_sharpe is not None
            and after_sharpe is not None
            and after_sharpe
                >= before_sharpe + 0.10
            and after_fitness is not None
            and after_fitness > 1.0
        )

    return False



# ---------------------------------------------------------------------------
# Canonical AST Bridge & Shadow-mode SQLite Store (010_forbidden_edit_motifs.sql)
# ---------------------------------------------------------------------------
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


_DIFF_TO_MOTIF_TYPE: dict[str, EditMotifType] = {
    "ADD_RANK": EditMotifType.ADD_RANK,
    "REMOVE_RANK": EditMotifType.REMOVE_RANK,
    "ADD_HUMP": EditMotifType.ADD_HUMP,
    "ADD_WINSORIZE": EditMotifType.ADD_WINSORIZE,
    "ADD_TRADE_WHEN": EditMotifType.ADD_TRADE_WHEN,
    "ADD_BACKFILL": EditMotifType.ADD_BACKFILL,
    "ADD_DECAY": EditMotifType.WRAP_OPERATOR,
    "ADD_GROUP_NEUTRALIZE": EditMotifType.WRAP_OPERATOR,
    "REPLACE_FIELD": EditMotifType.REPLACE_FIELD,
    "WINDOW_RESIZE": EditMotifType.CHANGE_WINDOW,
    "CHANGE_NEUTRALIZATION": EditMotifType.CHANGE_NEUTRALIZATION,
    "CHANGE_DECAY": EditMotifType.CHANGE_DECAY,
    "CHANGE_TRUNCATION": EditMotifType.CHANGE_TRUNCATION,
    "CHANGE_SETTINGS": EditMotifType.SETTING_ONLY_CHANGE,
    "CUSTOM_STRUCTURAL_EDIT": EditMotifType.STRUCTURAL_REWRITE,
    "PARSE_FAILURE": EditMotifType.STRUCTURAL_REWRITE,
}


def build_edit_motif_from_expressions(
    parent_expression: str,
    child_expression: str,
    parent_settings: Mapping[str, Any] | None = None,
    child_settings: Mapping[str, Any] | None = None,
) -> EditMotif:
    """Construct a validated EditMotif directly from Canonical AST diff."""
    from worldquant_brain.canonical_ast import diff_ast_edit

    d = diff_ast_edit(
        parent_expression,
        child_expression,
        parent_settings=parent_settings,
        child_settings=child_settings,
    )
    mtype = _DIFF_TO_MOTIF_TYPE.get(str(d["motif_type"]), EditMotifType.STRUCTURAL_REWRITE)
    changed_lits = tuple(
        (f"lit_{idx}", item.split("->", 1)[0], item.split("->", 1)[1])
        for idx, item in enumerate(d["changed_literals"])
        if "->" in item
    )
    setting_changes = {
        k: (v["before"], v["after"])
        for k, v in dict(d["setting_changes"]).items()
    }
    motif = EditMotif(
        motif_type=mtype,
        target_ast_path=d["target_ast_path"],
        before_subtree_hash=d["before_subtree_hash"],
        after_subtree_hash=d["after_subtree_hash"],
        added_operators=tuple(d["added_operators"]),
        removed_operators=tuple(d["removed_operators"]),
        changed_fields=tuple(d["changed_fields"]),
        changed_literals=changed_lits,
        setting_changes=setting_changes,
        structural_change=bool(d["structural_change"]),
        parameter_only=bool(d["parameter_only"]),
        ast_schema_version=str(d["ast_schema_version"]),
    )
    motif.validate()
    return motif


class ForbiddenEditMotifShadowStore:
    """Record-only / Shadow-mode SQLite store for edit motifs and contexts."""

    def __init__(self, db_path: str | Path, shadow_mode: bool = True) -> None:
        self.db_path = str(db_path)
        self.shadow_mode = bool(shadow_mode)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def record_motif_and_context(self, motif: EditMotif, context: MotifContext) -> tuple[str, str]:
        """Idempotently insert motif and context into SQLite."""
        motif.validate()
        context.validate()
        now = datetime.now(timezone.utc).isoformat()
        m_id = motif.motif_id()
        c_id = context.motif_context_id(motif)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO research_edit_motifs (
                    motif_id, motif_type, motif_hash, target_ast_path,
                    before_subtree_hash, after_subtree_hash,
                    added_operators_json, removed_operators_json,
                    changed_fields_json, changed_literals_json, setting_changes_json,
                    canonical_edit_json, structural_change, parameter_only,
                    permanent_invalid, permanent_invalid_reason, ast_schema_version,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?, ?)
                """,
                (
                    m_id,
                    motif.motif_type.value,
                    motif.motif_hash(),
                    motif.target_ast_path,
                    motif.before_subtree_hash,
                    motif.after_subtree_hash,
                    json.dumps(list(motif.added_operators)),
                    json.dumps(list(motif.removed_operators)),
                    json.dumps(list(motif.changed_fields)),
                    json.dumps([list(x) for x in motif.changed_literals]),
                    json.dumps({k: list(v) for k, v in motif.setting_changes.items()}, sort_keys=True),
                    json.dumps(motif.canonical_payload(), sort_keys=True),
                    1 if motif.structural_change else 0,
                    1 if motif.parameter_only else 0,
                    motif.ast_schema_version,
                    now,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO research_motif_contexts (
                    motif_context_id, motif_id, parent_context_hash, hypothesis_family,
                    data_domain, failure_gate, region, universe, delay, objective_lane,
                    parent_turnover_band, parent_correlation_band, parent_coverage_band,
                    parent_ast_signature, field_catalog_version, operator_schema_version,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    c_id,
                    m_id,
                    context.context_hash(),
                    context.hypothesis_family,
                    context.data_domain,
                    context.failure_gate,
                    context.region,
                    context.universe,
                    context.delay,
                    context.objective_lane.value,
                    context.parent_turnover_band,
                    context.parent_correlation_band,
                    context.parent_coverage_band,
                    context.parent_ast_signature,
                    context.field_catalog_version,
                    context.operator_schema_version,
                    now,
                    now,
                ),
            )
        return m_id, c_id
