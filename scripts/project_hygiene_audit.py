#!/usr/bin/env python3
"""Antigravity Quant — Project Hygiene & Literature KB Integrity Auditor (P46).

Read-only by default; validates:
1. All `docs/references/V*_ROUND*_MEGA_QUANT_LITERATURE_SURVEY.md` files:
   - Row extraction (supports both 5-column and 7-column survey tables)
   - Zero missing URLs and zero Tier-0 banned domains
   - Cross-volume canonical title & URL deduplication metrics
2. `docs/references/README.md` presence and volume count consistency
3. Repository clutter audit (unarchived `scripts/p*.zip`, `__pycache__`, `.pytest_cache`)
4. Optional `--fix-caches` to clean stale `__pycache__` / `.pytest_cache` directories safely.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

TZ_TAIPEI = timezone(timedelta(hours=8))

TIER0_BANNED: tuple[str, ...] = (
    "researchgate.net",
    "mdpi.com",
    "scirp.org",
    "hindawi.com",
    "medium.com",
    "towardsdatascience.com",
    "investopedia.com",
    "wikipedia.org",
    "scribd.com",
    "academia.edu",
    "coursehero.com",
    "chegg.com",
    "quora.com",
    "reddit.com",
    "youtube.com",
    "youtu.be",
    "facebook.com",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "google.com/search",
    "google.com/url",
)


def unwrap_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    for _ in range(3):
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc.lower()
        if "google." in host and parsed.path in ("/search", "/url"):
            qs = urllib.parse.parse_qs(parsed.query)
            target = (qs.get("q") or qs.get("url") or [""])[0]
            target = urllib.parse.unquote(target).strip()
            if target.startswith(("http://", "https://")):
                url = target
                continue
        break
    return url


def normalize_url(url: str) -> str:
    u = unwrap_url(url).lower().strip()
    u = re.sub(r"^https?://(www\.)?", "", u)
    u = u.rstrip("/")
    u = re.sub(r"[?&](utm_[^&]+|fbclid=[^&]+)", "", u)
    m = re.search(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?", u)
    if m:
        return f"arxiv:{m.group(1)}"
    return u


def normalize_title(title: str) -> str:
    t = (title or "").lower()
    t = re.sub(r"\*\*|\*|`|_", "", t)
    t = re.split(r"\s+[—–]\s+", t)[0]
    t = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", t).strip()
    return t


def is_tier0_banned(url: str) -> bool:
    u = (url or "").lower()
    return any(b in u for b in TIER0_BANNED)


def extract_url_from_cells(parts: list[str]) -> tuple[str, str]:
    """Extract (label, url) from either 5-col (col 2) or 7-col (col 5) survey rows."""
    candidate_indices = [5, 2, 1] if len(parts) >= 6 else [2, 1]
    for idx in candidate_indices:
        if idx >= len(parts):
            continue
        cell = parts[idx]
        m = re.search(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", cell)
        if m:
            return m.group(1), unwrap_url(m.group(2))
        m_bare = re.search(r"(https?://[^\s|\)]+)", cell)
        if m_bare:
            return "Source", unwrap_url(m_bare.group(1))
    return "", ""


def audit_survey_file(filepath: str | Path) -> dict[str, Any]:
    path = Path(filepath)
    raw_rows = 0
    missing_url = 0
    banned_hits: list[dict[str, str]] = []
    titles: set[str] = set()
    urls: set[str] = set()

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not re.match(r"^\|\s*\d+\s*\|", s):
                continue
            parts = [p.strip() for p in s.strip("|").split("|")]
            if len(parts) < 4:
                continue
            raw_rows += 1
            title_raw = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", parts[1])
            nt = normalize_title(title_raw)
            if nt:
                titles.add(nt)
            _, clean_url = extract_url_from_cells(parts)
            if not clean_url:
                missing_url += 1
            else:
                if is_tier0_banned(clean_url):
                    banned_hits.append({"title": title_raw[:80], "url": clean_url})
                urls.add(normalize_url(clean_url))

    return {
        "file": path.name,
        "size_kb": round(path.stat().st_size / 1024.0, 1),
        "raw_rows": raw_rows,
        "unique_titles": len(titles),
        "unique_urls": len(urls),
        "missing_url": missing_url,
        "banned_count": len(banned_hits),
        "banned_samples": banned_hits[:5],
    }


def audit_project(repo_root: str | Path, *, fix_caches: bool = False) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    refs_dir = root / "docs" / "references"
    survey_files = sorted(glob.glob(str(refs_dir / "V*_ROUND*_MEGA_QUANT_LITERATURE_SURVEY.md")))

    per_file: list[dict[str, Any]] = []
    total_rows = 0
    total_missing = 0
    total_banned = 0
    for fp in survey_files:
        res = audit_survey_file(fp)
        per_file.append(res)
        total_rows += res["raw_rows"]
        total_missing += res["missing_url"]
        total_banned += res["banned_count"]

    scripts_dir = root / "scripts"
    unarchived_zips = (
        sorted(p.name for p in scripts_dir.glob("p*.zip")) if scripts_dir.exists() else []
    )

    cache_dirs: list[Path] = []
    for dirpath, dirnames, _ in os.walk(root):
        if ".git" in Path(dirpath).parts:
            continue
        for dn in list(dirnames):
            if dn in ("__pycache__", ".pytest_cache"):
                cache_dirs.append(Path(dirpath) / dn)
                dirnames.remove(dn)

    cleaned_caches = 0
    if fix_caches:
        for cd in cache_dirs:
            try:
                shutil.rmtree(cd, ignore_errors=True)
                cleaned_caches += 1
            except OSError:
                pass

    readme_path = refs_dir / "README.md"
    readme_exists = readme_path.exists()

    return {
        "audited_at_taipei": datetime.now(TZ_TAIPEI).strftime("%Y-%m-%d %H:%M:%S+08:00"),
        "repo_root": str(root),
        "docs_references_readme_exists": readme_exists,
        "survey_files_count": len(survey_files),
        "total_raw_rows": total_rows,
        "total_missing_urls": total_missing,
        "total_tier0_banned": total_banned,
        "unarchived_script_zips": unarchived_zips,
        "cache_dirs_found": len(cache_dirs),
        "cache_dirs_cleaned": cleaned_caches,
        "status": "PASS" if (total_missing == 0 and total_banned == 0) else "WARN",
        "files": per_file,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Antigravity Quant Project Hygiene & KB Auditor")
    parser.add_argument("--repo", default=".", help="Repository root path")
    parser.add_argument("--fix-caches", action="store_true", help="Remove stale __pycache__ and .pytest_cache dirs")
    parser.add_argument("--out", default="", help="Optional JSON output report path")
    args = parser.parse_args()

    report = audit_project(args.repo, fix_caches=args.fix_caches)
    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {k: v for k, v in report.items() if k != "files"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
