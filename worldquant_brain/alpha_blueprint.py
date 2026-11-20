from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Literal


class AlphaFamily(StrEnum):
    ANALYST_REVISION = "ANALYST_REVISION"
    FUNDAMENTAL_QUALITY = "FUNDAMENTAL_QUALITY"
    FUNDAMENTAL_CHANGE = "FUNDAMENTAL_CHANGE"
    EARNINGS_SURPRISE = "EARNINGS_SURPRISE"
    MOMENTUM = "MOMENTUM"
    MEAN_REVERSION = "MEAN_REVERSION"
    VOLATILITY = "VOLATILITY"
    LIQUIDITY = "LIQUIDITY"
    SENTIMENT = "SENTIMENT"
    OPTIONS = "OPTIONS"
    RELATIONSHIP = "RELATIONSHIP"
    ORTHOGONAL_RESIDUAL = "ORTHOGONAL_RESIDUAL"


class ExpectedTurnover(StrEnum):
    LOW = "LOW"
    LOW_MEDIUM = "LOW_MEDIUM"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True, slots=True)
class DataFieldSpec:
    field_id: str
    dataset_id: str | None
    domain: str
    field_type: Literal["MATRIX", "VECTOR", "GROUP"] = "MATRIX"
    role: Literal["PRIMARY", "CONFIRMATORY", "CONDITION", "GROUP", "RISK_CONTROL"] = "PRIMARY"
    lag_days: int = 0
    minimum_coverage: float = 0.50
    maximum_staleness_days: int | None = None
    expected_update_frequency: str | None = None
    vector_reducer: str | None = None


@dataclass(frozen=True, slots=True)
class TransformSpec:
    operator: str
    parameters: dict[str, Any] = field(default_factory=dict)
    purpose: str = ""


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    expression_template: str
    purpose: str
    failure_mode: str


@dataclass(frozen=True, slots=True)
class ComplexityBudget:
    maximum_tree_depth: int = 6
    maximum_operator_count: int = 12
    maximum_data_domains: int = 3
    maximum_conditions: int = 3
    maximum_ts_parameters: int = 4


@dataclass(frozen=True, slots=True)
class RobustnessPlan:
    yearly_subperiods: bool = True
    universe_variants: tuple[str, ...] = ("TOP3000", "TOP2000", "TOP1000")
    neutralization_variants: tuple[str, ...] = ("INDUSTRY", "SUBINDUSTRY")
    decay_offsets: tuple[int, ...] = (-2, -1, 0, 1, 2)
    truncation_variants: tuple[float, ...] = (0.05, 0.06, 0.08)
    window_perturbation_ratio: float = 0.20
    cost_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0)
    minimum_positive_year_ratio: float = 0.60
    maximum_single_year_pnl_share: float = 0.40
    maximum_sharpe_degradation: float = 0.35
    minimum_parameter_pass_ratio: float = 0.60
    minimum_universe_pass_ratio: float = 0.67
    minimum_regime_pass_ratio: float = 0.60


@dataclass(frozen=True, slots=True)
class AlphaBlueprint:
    blueprint_id: str
    version: int
    family: AlphaFamily
    title: str
    economic_rationale: str
    falsifiable_claim: str
    primary_fields: tuple[DataFieldSpec, ...]
    confirmatory_fields: tuple[DataFieldSpec, ...] = ()
    risk_control_fields: tuple[DataFieldSpec, ...] = ()
    transforms: tuple[TransformSpec, ...] = ()
    conditions: tuple[ConditionSpec, ...] = ()
    peer_groups: tuple[str, ...] = ("SUBINDUSTRY",)
    expected_horizon_days: tuple[int, int] = (20, 60)
    expected_turnover: ExpectedTurnover = ExpectedTurnover.LOW_MEDIUM
    expected_failure_modes: tuple[str, ...] = ()
    expression_template: str = ""
    parameter_space: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    complexity_budget: ComplexityBudget = field(default_factory=ComplexityBudget)
    robustness_plan: RobustnessPlan = field(default_factory=RobustnessPlan)
    tags: tuple[str, ...] = ()
    parent_blueprint_id: str | None = None

    def canonical_dict(self) -> dict[str, Any]:
        return _normalize(asdict(self))

    def sha256(self) -> str:
        payload = json.dumps(self.canonical_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def domains(self) -> set[str]:
        fields = (*self.primary_fields, *self.confirmatory_fields, *self.risk_control_fields)
        return {item.domain for item in fields}

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.primary_fields:
            errors.append("At least one primary field is required")
        if not self.economic_rationale.strip():
            errors.append("Economic rationale is required")
        if not self.falsifiable_claim.strip():
            errors.append("Falsifiable claim is required")
        if self.expected_horizon_days[0] <= 0 or self.expected_horizon_days[1] < self.expected_horizon_days[0]:
            errors.append("Invalid expected horizon")
        if len(self.conditions) > self.complexity_budget.maximum_conditions:
            errors.append("Condition count exceeds complexity budget")
        if len(self.domains()) > self.complexity_budget.maximum_data_domains:
            errors.append("Data-domain count exceeds complexity budget")
        if len(self.transforms) > self.complexity_budget.maximum_operator_count:
            errors.append("Transform count exceeds complexity budget")
        placeholders = _template_placeholders(self.expression_template)
        missing = placeholders - set(self.parameter_space)
        if missing:
            errors.append(f"Missing parameter definitions: {sorted(missing)}")
        return errors


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    return value


def _template_placeholders(template: str) -> set[str]:
    import string
    return {field_name for _, field_name, _, _ in string.Formatter().parse(template) if field_name}


def analyst_revision_blueprint() -> AlphaBlueprint:
    return AlphaBlueprint(
        blueprint_id="ANALYST_REVISION_001",
        version=1,
        family=AlphaFamily.ANALYST_REVISION,
        title="Fresh analyst revision with low dispersion",
        economic_rationale="Analyst estimate revisions are incorporated gradually; fresh and broad revisions with low disagreement should be more reliable than stale or highly dispersed revisions.",
        falsifiable_claim="The signal should outperform its no-freshness and no-dispersion variants, remain positive across sub-universes, and degrade smoothly under nearby windows.",
        primary_fields=(DataFieldSpec("{revision_field}", "ANALYST", "ANALYST", role="PRIMARY", minimum_coverage=0.35, maximum_staleness_days=20),),
        confirmatory_fields=(DataFieldSpec("{dispersion_field}", "ANALYST", "ANALYST", role="CONFIRMATORY", minimum_coverage=0.30),),
        transforms=(TransformSpec("ts_zscore", {"window": "{window}"}, "Normalize revision intensity"), TransformSpec("group_rank", {"group": "{group}"}, "Compare with peers")),
        conditions=(ConditionSpec("{coverage_field} > {coverage_threshold}", "Require analyst coverage", "LOW_COVERAGE"), ConditionSpec("{dispersion_field} < {dispersion_threshold}", "Reject high disagreement", "HIGH_DISPERSION")),
        expression_template="trade_when({coverage_field} > {coverage_threshold}, group_rank(ts_zscore({revision_field}, {window}), {group}), -1)",
        parameter_space={"revision_field": ("est_eps_revision",), "dispersion_field": ("est_eps_dispersion",), "coverage_field": ("analyst_coverage",), "coverage_threshold": (2, 3, 5), "dispersion_threshold": (0.2, 0.3), "window": (20, 40, 60), "group": ("industry", "subindustry")},
        expected_failure_modes=("LOW_COVERAGE", "STALE_ESTIMATE", "HIGH_TURNOVER", "CROWDED_SIGNAL"),
        tags=("hypothesis-driven", "analyst", "low-medium-turnover"),
    )


# Default verified WorldQuant BRAIN field catalog (plus runtime dynamic additions)
DEFAULT_BRAIN_FIELD_CATALOG: dict[str, DataFieldSpec] = {
    "close": DataFieldSpec("close", "PV1", "PRICE_VOLUME", "MATRIX", "PRIMARY", minimum_coverage=0.95),
    "open": DataFieldSpec("open", "PV1", "PRICE_VOLUME", "MATRIX", "PRIMARY", minimum_coverage=0.95),
    "high": DataFieldSpec("high", "PV1", "PRICE_VOLUME", "MATRIX", "PRIMARY", minimum_coverage=0.95),
    "low": DataFieldSpec("low", "PV1", "PRICE_VOLUME", "MATRIX", "PRIMARY", minimum_coverage=0.95),
    "volume": DataFieldSpec("volume", "PV1", "PRICE_VOLUME", "MATRIX", "CONFIRMATORY", minimum_coverage=0.95),
    "vwap": DataFieldSpec("vwap", "PV1", "PRICE_VOLUME", "MATRIX", "PRIMARY", minimum_coverage=0.95),
    "returns": DataFieldSpec("returns", "PV1", "PRICE_VOLUME", "MATRIX", "PRIMARY", minimum_coverage=0.95),
    "cap": DataFieldSpec("cap", "PV1", "PRICE_VOLUME", "MATRIX", "RISK_CONTROL", minimum_coverage=0.90),
    "operating_income": DataFieldSpec("operating_income", "FUNDAMENTAL", "FUNDAMENTAL", "MATRIX", "PRIMARY", minimum_coverage=0.75),
    "cashflow_op": DataFieldSpec("cashflow_op", "FUNDAMENTAL", "FUNDAMENTAL", "MATRIX", "PRIMARY", minimum_coverage=0.75),
    "assets": DataFieldSpec("assets", "FUNDAMENTAL", "FUNDAMENTAL", "MATRIX", "RISK_CONTROL", minimum_coverage=0.80),
    "debt": DataFieldSpec("debt", "FUNDAMENTAL", "FUNDAMENTAL", "MATRIX", "RISK_CONTROL", minimum_coverage=0.75),
    "equity": DataFieldSpec("equity", "FUNDAMENTAL", "FUNDAMENTAL", "MATRIX", "RISK_CONTROL", minimum_coverage=0.75),
    "sales": DataFieldSpec("sales", "FUNDAMENTAL", "FUNDAMENTAL", "MATRIX", "CONFIRMATORY", minimum_coverage=0.80),
    "ebitda": DataFieldSpec("ebitda", "FUNDAMENTAL", "FUNDAMENTAL", "MATRIX", "PRIMARY", minimum_coverage=0.75),
    "est_eps": DataFieldSpec("est_eps", "ANALYST", "ANALYST", "MATRIX", "PRIMARY", minimum_coverage=0.45, maximum_staleness_days=20),
    "target_price": DataFieldSpec("target_price", "ANALYST", "ANALYST", "MATRIX", "CONFIRMATORY", minimum_coverage=0.45),
    "scl12_buzz": DataFieldSpec("scl12_buzz", "SENTIMENT", "SENTIMENT", "MATRIX", "PRIMARY", minimum_coverage=0.35),
    "opt6_slopeavg1m": DataFieldSpec("opt6_slopeavg1m", "OPTIONS", "OPTIONS", "MATRIX", "PRIMARY", minimum_coverage=0.40),
    "opt6_fcstr2imp": DataFieldSpec("opt6_fcstr2imp", "OPTIONS", "OPTIONS", "MATRIX", "CONDITION", minimum_coverage=0.40),
}


def fundamental_quality_blueprint() -> AlphaBlueprint:
    return AlphaBlueprint(
        blueprint_id="FUNDAMENTAL_QUALITY_001",
        version=1,
        family=AlphaFamily.FUNDAMENTAL_QUALITY,
        title="Cashflow-to-enterprise-value yield with de-leveraging confirmation",
        economic_rationale="Firms generating strong operating cashflow relative to enterprise value while reducing debt-to-assets exhibit persistent fundamental outperformance.",
        falsifiable_claim="Outperforms un-neutralized raw cashflow yield, remains stable across TOP3000/TOP1000, and survives 1.5x/2.0x cost stress.",
        primary_fields=(DataFieldSpec("{cf_field}", "FUNDAMENTAL", "FUNDAMENTAL", role="PRIMARY", minimum_coverage=0.75),),
        confirmatory_fields=(DataFieldSpec("{debt_field}", "FUNDAMENTAL", "FUNDAMENTAL", role="CONFIRMATORY", minimum_coverage=0.75),),
        risk_control_fields=(DataFieldSpec("assets", "FUNDAMENTAL", "FUNDAMENTAL", role="RISK_CONTROL", minimum_coverage=0.80),),
        transforms=(TransformSpec("winsorize", {"std": 4}, "Clip extreme fundamental outliers"), TransformSpec("group_neutralize", {"group": "{group}"}, "Peer group neutralization"), TransformSpec("hump", {"hump": "{hump_val}"}, "Suppress micro-turnover")),
        conditions=(ConditionSpec("assets > 0", "Valid balance sheet", "MISSING_ASSETS"),),
        expression_template="hump(group_neutralize(winsorize(rank({cf_field} / (cap + {debt_field} + 0.001)), std=4), {group}), hump={hump_val})",
        parameter_space={"cf_field": ("cashflow_op", "operating_income"), "debt_field": ("debt",), "group": ("subindustry", "industry"), "hump_val": (0.01, 0.015)},
        expected_failure_modes=("STALE_FUNDAMENTALS", "VALUE_TRAP_IN_HIGH_BETA_RALLY"),
        tags=("fundamental", "low-turnover", "verified-fields"),
    )


def builtin_blueprints() -> list[AlphaBlueprint]:
    return [analyst_revision_blueprint(), fundamental_quality_blueprint()]


def validate_blueprint_fields_against_catalog(
    blueprint: AlphaBlueprint,
    parameters: dict[str, Any],
    catalog: dict[str, DataFieldSpec] | None = None,
) -> list[str]:
    """Stage 3: Verify every resolved data field exists in the BRAIN Field Catalog and Vector fields specify a reducer."""
    cat = catalog if catalog is not None else DEFAULT_BRAIN_FIELD_CATALOG
    errors: list[str] = []
    all_specs = (*blueprint.primary_fields, *blueprint.confirmatory_fields, *blueprint.risk_control_fields)
    for spec in all_specs:
        raw_id = spec.field_id
        resolved_id = raw_id.format(**parameters) if "{" in raw_id else raw_id
        if resolved_id not in cat:
            errors.append(f"FIELD_NOT_IN_CATALOG:{resolved_id}")
            continue
        cat_spec = cat[resolved_id]
        if (spec.field_type == "VECTOR" or cat_spec.field_type == "VECTOR") and not (spec.vector_reducer or cat_spec.vector_reducer):
            errors.append(f"VECTOR_FIELD_MISSING_REDUCER:{resolved_id}")
    return errors


def check_expression_complexity_budget(expression: str, budget: ComplexityBudget) -> list[str]:
    """Stage 4: Enforce AST tree depth <= 6 and operator count <= 12."""
    import re
    errors: list[str] = []
    ops = re.findall(r"([a-z_][a-z0-9_]*)\s*\(", (expression or "").lower())
    if len(ops) > budget.maximum_operator_count:
        errors.append(f"REJECT_COMPLEXITY:operator_count={len(ops)}>{budget.maximum_operator_count}")
    depth = 0
    max_depth = 0
    for ch in expression or "":
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth = max(0, depth - 1)
    if max_depth > budget.maximum_tree_depth:
        errors.append(f"REJECT_COMPLEXITY:tree_depth={max_depth}>{budget.maximum_tree_depth}")
    return errors


def generate_controlled_candidates(
    blueprint: AlphaBlueprint,
    *,
    catalog: dict[str, DataFieldSpec] | None = None,
    blocked_signatures: set[tuple[str, str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Stage 4 & 8: Generate STRUCTURAL_VARIANT vs PARAMETER_VARIANT candidates while skipping failed Research Memory lessons."""
    val_errs = blueprint.validate()
    if val_errs:
        raise ValueError(f"Invalid blueprint: {val_errs}")
    import itertools
    keys = sorted(blueprint.parameter_space.keys())
    values_list = [blueprint.parameter_space[k] for k in keys]
    blocked = blocked_signatures or set()
    candidates: list[dict[str, Any]] = []
    structural_keys = {"revision_field", "dispersion_field", "coverage_field", "cf_field", "debt_field", "group"}

    seen_structural: set[tuple[Any, ...]] = set()
    for combo in itertools.product(*values_list):
        params = dict(zip(keys, combo))
        field_errs = validate_blueprint_fields_against_catalog(blueprint, params, catalog=catalog)
        if field_errs:
            continue
        expr = blueprint.expression_template.format(**params)
        comp_errs = check_expression_complexity_budget(expr, blueprint.complexity_budget)
        if comp_errs:
            continue
        struct_sig = tuple((k, params[k]) for k in keys if k in structural_keys)
        variant_kind = "STRUCTURAL_VARIANT" if struct_sig not in seen_structural else "PARAMETER_VARIANT"
        seen_structural.add(struct_sig)

        dom_key = ",".join(sorted(blueprint.domains()))
        if (blueprint.family.value, dom_key, variant_kind) in blocked and variant_kind == "PARAMETER_VARIANT":
            continue

        candidates.append({
            "blueprint_id": blueprint.blueprint_id,
            "blueprint_hash": blueprint.sha256(),
            "family": blueprint.family.value,
            "variant_kind": variant_kind,
            "parameters": params,
            "expression": expr,
        })
    return candidates
