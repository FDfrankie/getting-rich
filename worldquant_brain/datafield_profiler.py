"""Stage-A Datafield Characterization & Qualification Pipeline for WorldQuant BRAIN.

Profiles each datafield BEFORE formula generation so the miner selects
frequency-matched windows, requires vector reducers when needed, predicts
sub-universe coverage fragility, and blocks unverified PIT fields from promotion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DatafieldProfile:
    field_id: str
    dataset_id: str
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    field_type: str = "MATRIX"  # MATRIX | VECTOR
    coverage: float = 0.80
    zero_rate: float = 0.05
    nan_rate: float = 0.15
    update_interval_median_days: float = 1.0
    staleness_p90_days: float = 5.0
    subindustry_coverage_dispersion: float = 0.10
    point_in_time_verified: bool = True
    profile_age_days: int = 0
    profile_version: str = "df-profile-v1"


@dataclass(frozen=True)
class FieldQualificationResult:
    field_id: str
    qualified_for_search: bool
    promotion_eligible: bool
    requires_vector_reducer: bool
    recommended_windows: tuple[int, ...]
    allow_short_backfill: bool
    sub_universe_risk: str  # LOW | MEDIUM | HIGH
    reasons: tuple[str, ...]


def qualify_datafield(profile: DatafieldProfile) -> FieldQualificationResult:
    """Qualify a datafield in Stage A before allowing it into Stage B formula search."""
    reasons: list[str] = []
    qualified = True
    promotion_ok = True

    if profile.coverage < 0.30:
        qualified = False
        reasons.append(f"COVERAGE_TOO_LOW({profile.coverage:.2f}<0.30)")
    if profile.zero_rate > 0.85:
        qualified = False
        reasons.append(f"EXCESSIVE_ZERO_RATE({profile.zero_rate:.2f}>0.85)")
    if not profile.point_in_time_verified:
        promotion_ok = False
        reasons.append("PIT_UNVERIFIED_BLOCK_PROMOTION")
    if profile.profile_age_days > 30:
        reasons.append(f"PROFILE_STALE({profile.profile_age_days}d>30d)")

    ftype = profile.field_type.upper()
    requires_reducer = ftype == "VECTOR"

    # Match lookback windows to update frequency
    if profile.update_interval_median_days >= 30.0:
        # Slow fundamental / quarterly data -> long windows only
        rec_windows = (63, 126, 252)
        allow_short_bf = False
    elif profile.update_interval_median_days >= 5.0:
        # Weekly / analyst revision data -> medium windows
        rec_windows = (20, 40, 60)
        allow_short_bf = profile.coverage >= 0.50
    else:
        # Daily PV / News / Buzz -> short/medium windows
        rec_windows = (5, 10, 20)
        allow_short_bf = True

    if profile.coverage < 0.55 or profile.subindustry_coverage_dispersion > 0.30:
        sub_risk = "HIGH"
    elif profile.coverage < 0.75 or profile.subindustry_coverage_dispersion > 0.18:
        sub_risk = "MEDIUM"
    else:
        sub_risk = "LOW"

    return FieldQualificationResult(
        field_id=profile.field_id,
        qualified_for_search=qualified,
        promotion_eligible=qualified and promotion_ok,
        requires_vector_reducer=requires_reducer,
        recommended_windows=rec_windows,
        allow_short_backfill=allow_short_bf,
        sub_universe_risk=sub_risk,
        reasons=tuple(reasons),
    )
