"""Stratified Field Survey, FieldOpportunityScore, and Metadata-Hash Cache-First Probe."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable


CATEGORY_QUOTAS: dict[str, int] = {
    "FUNDAMENTAL": 80,
    "ANALYST": 80,
    "NEWS": 60,
    "SENTIMENT": 60,
    "OPTIONS": 60,
    "RELATIONSHIP": 60,
    "MODEL": 40,
    "PRICE_VOLUME": 40,
    "OTHER": 20,
}


@dataclass(frozen=True, slots=True)
class FieldProfile:
    field_id: str
    dataset_id: str
    category: str
    field_type: str = "MATRIX"  # MATRIX | VECTOR | GROUP
    coverage: float = 0.80
    alpha_count: int = 10
    user_count: int = 5
    freshness: float = 0.70
    interpretability: float = 0.80
    pit_confidence: float = 0.90
    vector_reducer: str | None = None
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    description: str = ""

    def metadata_hash(self) -> str:
        payload = json.dumps(
            {
                "field_id": self.field_id,
                "dataset_id": self.dataset_id,
                "category": self.category,
                "field_type": self.field_type,
                "coverage": round(float(self.coverage), 4),
                "alpha_count": int(self.alpha_count),
                "user_count": int(self.user_count),
                "region": self.region,
                "universe": self.universe,
                "delay": int(self.delay),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SurveyScore:
    field: FieldProfile
    score: float
    unseen: bool


def validate_vector_field_reducer(f: FieldProfile) -> tuple[bool, str]:
    if str(f.field_type).upper() == "VECTOR" and not (f.vector_reducer or "").strip():
        return False, f"VECTOR_FIELD_MISSING_REDUCER:{f.field_id}"
    return True, "OK"


def should_probe_field(f: FieldProfile, cached_metadata_hash: str | None) -> bool:
    """Metadata-first, Cache-first: skip BRAIN probe when metadata_hash is unchanged."""
    ok, _ = validate_vector_field_reducer(f)
    if not ok:
        return False
    if cached_metadata_hash and cached_metadata_hash == f.metadata_hash():
        return False
    return True


def score_field(f: FieldProfile, seen_fields: set[str], domain_gaps: set[str] | None = None) -> SurveyScore:
    unseen = f.field_id not in seen_fields
    unseen_score = 1.0 if unseen else 0.0
    low_alpha = max(0.0, 1.0 - min(float(f.alpha_count), 500.0) / 500.0)
    low_user = max(0.0, 1.0 - min(float(f.user_count), 300.0) / 300.0)
    cov = max(0.0, min(1.0, float(f.coverage)))
    cov_suit = cov if cov >= 0.30 else cov * 0.25
    gap_score = 1.0 if (domain_gaps and f.category.upper() in domain_gaps) or f.category.upper() != "PRICE_VOLUME" else 0.30

    total = (
        0.20 * unseen_score
        + 0.15 * low_alpha
        + 0.10 * low_user
        + 0.15 * cov_suit
        + 0.10 * float(f.freshness)
        + 0.10 * gap_score
        + 0.10 * float(f.interpretability)
        + 0.10 * float(f.pit_confidence)
    )
    return SurveyScore(field=f, score=round(total, 4), unseen=unseen)


def rank_fields(
    fields: Iterable[FieldProfile],
    seen_fields: Iterable[str],
    *,
    per_category: int | None = None,
    total: int = 500,
    quotas: dict[str, int] | None = None,
) -> list[SurveyScore]:
    seen = set(seen_fields)
    q_map = quotas or CATEGORY_QUOTAS
    buckets: dict[str, list[SurveyScore]] = {}
    for f in fields:
        ok, _ = validate_vector_field_reducer(f)
        if not ok:
            continue
        cat = f.category.upper()
        if cat not in q_map:
            cat = "OTHER"
        sc = score_field(f, seen)
        buckets.setdefault(cat, []).append(sc)

    selected: list[SurveyScore] = []
    for cat, items in buckets.items():
        items.sort(key=lambda x: x.score, reverse=True)
        cap = per_category if per_category is not None else q_map.get(cat, 20)
        selected.extend(items[:cap])
    selected.sort(key=lambda x: x.score, reverse=True)
    return selected[:total]
