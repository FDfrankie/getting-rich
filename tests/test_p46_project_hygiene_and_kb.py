"""Offline unit tests for P46 Project Hygiene & KB Consolidation."""
from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.project_hygiene_audit import (
    audit_project,
    audit_survey_file,
    is_tier0_banned,
    normalize_title,
    normalize_url,
    unwrap_url,
)


def test_url_unwrapping_and_canonical_arxiv() -> None:
    wrapped = "https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fpdf%2F2501.06758v2.pdf"
    assert unwrap_url(wrapped) == "https://arxiv.org/pdf/2501.06758v2.pdf"
    assert normalize_url(wrapped) == "arxiv:2501.06758"
    assert normalize_url("https://arxiv.org/abs/2501.06758") == "arxiv:2501.06758"


def test_tier0_domain_blocking() -> None:
    assert is_tier0_banned("https://www.researchgate.net/publication/12345")
    assert is_tier0_banned("https://www.mdpi.com/2227-7390/9/19/2385")
    assert is_tier0_banned("https://medium.com/@quant/post")
    assert not is_tier0_banned("https://arxiv.org/abs/2604.22528")
    assert not is_tier0_banned("https://www.nber.org/papers/w35617")


def test_normalize_title_strips_markdown_and_authors() -> None:
    t = "**Signature SDEs** — Christa Cuchiero (2023)"
    assert normalize_title(t) == "signature sdes"


def test_audit_survey_file_supports_5col_and_7col(tmp_path: Path) -> None:
    f5 = tmp_path / "V84_ROUND1_MEGA_QUANT_LITERATURE_SURVEY.md"
    f5.write_text(
        "| # | 標題 | 來源 | 發現 | FastExpr |\n"
        "|---|---|---|---|---|\n"
        "| 1 | Rough Volatility — Gatheral (2018) | [arXiv](https://arxiv.org/abs/1410.3394) | H~0.1 | rank(x) |\n",
        encoding="utf-8",
    )
    r5 = audit_survey_file(f5)
    assert r5["raw_rows"] == 1
    assert r5["missing_url"] == 0
    assert r5["banned_count"] == 0

    f7 = tmp_path / "V97_ROUND1_MEGA_QUANT_LITERATURE_SURVEY.md"
    f7.write_text(
        "| # | Title | Authors | Year | Venue | URL | Mechanism |\n"
        "|---|---|---|---|---|---|---|\n"
        "| 1 | Signature SDEs | Cuchiero et al. | 2026 | arXiv | [arXiv:2302.01362](https://arxiv.org/abs/2302.01362) | Riccati |\n",
        encoding="utf-8",
    )
    r7 = audit_survey_file(f7)
    assert r7["raw_rows"] == 1
    assert r7["missing_url"] == 0
    assert r7["banned_count"] == 0


def test_audit_project_detects_and_cleans_caches(tmp_path: Path) -> None:
    refs = tmp_path / "docs" / "references"
    refs.mkdir(parents=True)
    (refs / "README.md").write_text("# Master Index\n", encoding="utf-8")
    (refs / "V97_ROUND1_MEGA_QUANT_LITERATURE_SURVEY.md").write_text(
        "| 1 | Test Paper | Author | 2026 | arXiv | [arXiv](https://arxiv.org/abs/2601.00001) | Alpha |\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / "scripts" / "__pycache__"
    cache_dir.mkdir(parents=True)
    (cache_dir / "dummy.pyc").write_bytes(b"00")

    rep_ro = audit_project(tmp_path, fix_caches=False)
    assert rep_ro["status"] == "PASS"
    assert rep_ro["cache_dirs_found"] == 1
    assert rep_ro["cache_dirs_cleaned"] == 0
    assert cache_dir.exists()

    rep_fix = audit_project(tmp_path, fix_caches=True)
    assert rep_fix["cache_dirs_cleaned"] == 1
    assert not cache_dir.exists()
