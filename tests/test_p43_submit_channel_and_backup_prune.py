"""P43 unit tests: Bronze likely-duplicate + worst-first priority, rejection pruning, and XLSX unsubmitted auto-sync."""
from __future__ import annotations

import json
import sqlite3
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.backfill_brain_passed_alphas import export_vault_to_xlsx
from scripts.headless_brain_miner import sync_vault_snapshot_to_drive
from worldquant_brain.brain_client import BrainClient


def _make_client(tmp_path: Path) -> BrainClient:
    db_path = tmp_path / "alpha_vault.db"
    client = BrainClient(email="test@example.com", password="secret")
    client.authenticated = True
    client._quota_db = lambda: str(db_path)  # type: ignore[method-assign]
    client.vault_xlsx_path = tmp_path / "worldquant_brain_qualified_alphas_latest.xlsx"
    client.vault_json_paths = [tmp_path / "qualified_alpha_formulas_vault.json"]
    return client


def _read_xlsx_sheets(xlsx_path: Path) -> dict[str, list[list[str]]]:
    """Read sheet names and cell text from an .xlsx file using stdlib zipfile + xml (works without openpyxl)."""
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(xlsx_path, "r") as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            sst = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in sst.findall("a:si", ns):
                parts = [t.text or "" for t in si.findall(".//a:t", ns)]
                shared_strings.append("".join(parts))

        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rid_to_target = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels
        }

        out: dict[str, list[list[str]]] = {}
        for sheet in wb.findall(".//a:sheet", ns):
            name = sheet.attrib["name"]
            rid = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id", "")
            target = rid_to_target.get(rid, "")
            xml_path = target if target.startswith("xl/") else f"xl/{target.lstrip('/')}"
            ws = ET.fromstring(zf.read(xml_path))
            rows_data: list[list[str]] = []
            for row in ws.findall(".//a:sheetData/a:row", ns):
                row_vals: list[str] = []
                for c in row.findall("a:c", ns):
                    t_attr = c.attrib.get("t", "")
                    v_el = c.find("a:v", ns)
                    is_el = c.find("a:is", ns)
                    if t_attr == "s" and v_el is not None and v_el.text is not None:
                        row_vals.append(shared_strings[int(v_el.text)])
                    elif t_attr == "inlineStr" and is_el is not None:
                        row_vals.append("".join(t.text or "" for t in is_el.findall(".//a:t", ns)))
                    elif v_el is not None and v_el.text is not None:
                        row_vals.append(v_el.text)
                    else:
                        row_vals.append("")
                rows_data.append(row_vals)
            out[name] = rows_data
        return out


def test_p43_worst_first_prioritizes_likely_duplicates_and_allows_clones_in_bronze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BRAIN_CONSULTANT_MODE", raising=False)
    monkeypatch.setenv("BRAIN_SUBMIT_ORDER", "worst_first")
    monkeypatch.setenv("BRAIN_RESERVE_MIN_SHARPE", "1.50")
    monkeypatch.setenv("BRAIN_RESERVE_MIN_FITNESS", "1.20")

    client = _make_client(tmp_path)

    # Unique low-fitness candidate (not a duplicate)
    st_unique = client.stage_candidate_for_daily_selection(
        "UNIQ_LOW", "rank(ts_delta(fnd6_ebitda, 60))",
        sharpe=1.28, fitness=1.02, turnover=0.15, margin=0.0010, day_str="2026-11-25",
    )
    # Two structural/parameter clones (slightly higher fitness than UNIQ_LOW, but dup_priority=1!)
    st_clone_hi = client.stage_candidate_for_daily_selection(
        "CLONE_HI", "group_rank(ts_scale(cashflow_op / cap, 120), industry)",
        sharpe=1.42, fitness=1.15, turnover=0.18, margin=0.0012, day_str="2026-11-25",
    )
    st_clone_lo = client.stage_candidate_for_daily_selection(
        "CLONE_LO", "group_rank(ts_scale(cashflow_op / cap, 60), industry)",
        sharpe=1.34, fitness=1.08, turnover=0.18, margin=0.0011, day_str="2026-11-25",
    )
    assert st_unique == "PENDING"
    assert st_clone_hi == "PENDING"
    assert st_clone_lo == "PENDING"

    submitted_order: list[str] = []

    def _fake_submit(aid: str, **kwargs):
        submitted_order.append(aid)
        return {"success": True, "status": "SUBMITTED", "alpha_id": aid}

    client.submit_alpha = _fake_submit  # type: ignore[method-assign]
    client.flush_daily_candidates(day_str="2026-11-25", max_daily_submissions=2)

    # Likely-duplicate clone with lower fitness (CLONE_LO) MUST go first, then UNIQ_LOW
    # (CLONE_HI is skipped in the same round because CLONE_LO's skeleton succeeded)
    assert submitted_order == ["CLONE_LO", "UNIQ_LOW"]

    # Exact hash match to an already-SUBMITTED alpha is still blocked from re-staging
    st_exact_dup = client.stage_candidate_for_daily_selection(
        "EXACT_DUP_OF_SUB", "rank(ts_delta(fnd6_ebitda, 60))",
        sharpe=1.29, fitness=1.03, turnover=0.15, margin=0.0010, day_str="2026-11-25",
    )
    assert st_exact_dup == "DUPLICATE_ARCHIVED"


def test_p43_rejected_alpha_pruned_from_backup_and_never_rearchived(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BRAIN_CONSULTANT_MODE", raising=False)
    monkeypatch.setenv("BRAIN_SUBMIT_ORDER", "worst_first")
    monkeypatch.setenv("BRAIN_RESERVE_MIN_SHARPE", "1.50")
    monkeypatch.setenv("BRAIN_RESERVE_MIN_FITNESS", "1.20")

    client = _make_client(tmp_path)
    client.stage_candidate_for_daily_selection(
        "BAD_CORR_1", "rank(close / ts_delay(close, 5))",
        sharpe=1.30, fitness=1.04, turnover=0.20, margin=0.0010, day_str="2026-11-25",
    )
    client.stage_candidate_for_daily_selection(
        "GOOD_KEEP_1", "rank(ts_zscore(anl4_eps, 20))",
        sharpe=1.35, fitness=1.09, turnover=0.16, margin=0.0012, day_str="2026-11-25",
    )

    def _fake_submit(aid: str, **kwargs):
        if aid == "BAD_CORR_1":
            return {
                "success": False,
                "status": "SUBMIT_REJECTED",
                "alpha_id": aid,
                "corr_rejected": True,
                "error": "SELF_CORRELATION",
            }
        return {"success": False, "status": "DAILY_QUOTA_REACHED", "alpha_id": aid}

    client.submit_alpha = _fake_submit  # type: ignore[method-assign]
    client.flush_daily_candidates(day_str="2026-11-25", max_daily_submissions=2)

    # 1. BAD_CORR_1 MUST be deleted from qualified_alpha_archive and recorded in brain_rejected_alphas
    conn = sqlite3.connect(client._quota_db())
    arch_ids = {r[0] for r in conn.execute("SELECT alpha_id FROM qualified_alpha_archive").fetchall()}
    rej_ids = {r[0] for r in conn.execute("SELECT alpha_id FROM brain_rejected_alphas").fetchall()}
    pool_st = conn.execute("SELECT status FROM brain_candidate_pool WHERE alpha_id='BAD_CORR_1'").fetchone()[0]
    conn.close()
    assert "BAD_CORR_1" not in arch_ids
    assert "GOOD_KEEP_1" in arch_ids
    assert "BAD_CORR_1" in rej_ids
    assert pool_st == "FAILED_CORR"

    # 2. Auto-synced XLSX MUST NOT contain BAD_CORR_1
    assert client.vault_xlsx_path.exists()
    sheets = _read_xlsx_sheets(client.vault_xlsx_path)
    all_unsub_ids = [r[0] for r in sheets["All Unsubmitted Backups"][1:]]
    assert "BAD_CORR_1" not in all_unsub_ids
    assert "GOOD_KEEP_1" in all_unsub_ids

    # 3. Direct archive_qualified_alpha MUST refuse re-archiving BAD_CORR_1
    res_arch = client.archive_qualified_alpha(
        "BAD_CORR_1", "rank(close / ts_delay(close, 5))", 1.30, 1.04, 0.20, 0.0010
    )
    assert res_arch == "REJECTED_SKIPPED"

    # 4. backfill_all_passed_alphas & harvest_unsubmitted_candidates MUST skip BAD_CORR_1 even when API returns it UNSUBMITTED
    fake_unsub = {
        "results": [{
            "id": "BAD_CORR_1",
            "regular": {"code": "rank(close / ts_delay(close, 5))"},
            "settings": {"delay": 1, "universe": "TOP3000"},
            "is": {"sharpe": 1.30, "fitness": 1.04, "turnover": 0.20, "margin": 0.0010, "checks": []},
        }]
    }

    def _fake_get(url, params=None, timeout=20.0):
        resp = MagicMock()
        resp.status_code = 200
        resp.content = b"{}"
        resp.json.return_value = {"results": []} if (params or {}).get("status") == "ACTIVE" else fake_unsub
        return resp

    client.session = MagicMock()
    client.session.get.side_effect = _fake_get

    bf = client.backfill_all_passed_alphas(max_pages=2, page_size=50, vault_json_path=client.vault_json_paths[0])
    assert bf["unsubmitted_archived"] == 0
    staged = client.harvest_unsubmitted_candidates(limit=20, day_str="2026-11-25")
    assert staged == 0

    conn = sqlite3.connect(client._quota_db())
    arch_ids_after = {r[0] for r in conn.execute("SELECT alpha_id FROM qualified_alpha_archive").fetchall()}
    conn.close()
    assert "BAD_CORR_1" not in arch_ids_after


def test_p43_submitted_alpha_removed_from_xlsx_and_consultant_reserve_protected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BRAIN_CONSULTANT_MODE", raising=False)
    monkeypatch.setenv("BRAIN_SUBMIT_ORDER", "worst_first")
    monkeypatch.setenv("BRAIN_RESERVE_MIN_SHARPE", "1.50")
    monkeypatch.setenv("BRAIN_RESERVE_MIN_FITNESS", "1.20")

    client = _make_client(tmp_path)
    client.stage_candidate_for_daily_selection(
        "GEM_PROT", "rank(ts_delta(fnd6_ebitda, 60))",
        sharpe=1.85, fitness=1.45, turnover=0.12, margin=0.0025, day_str="2026-11-25",
    )
    client.stage_candidate_for_daily_selection(
        "BRONZE_SUB", "rank(-goodwill / sales)",
        sharpe=1.29, fitness=1.03, turnover=0.15, margin=0.0010, day_str="2026-11-25",
    )
    client.stage_candidate_for_daily_selection(
        "BRONZE_LEFT", "rank(ts_rank(anl4_eps, 20))",
        sharpe=1.38, fitness=1.12, turnover=0.16, margin=0.0011, day_str="2026-11-25",
    )

    submitted_ids: list[str] = []
    client.submit_alpha = lambda aid, **kw: (submitted_ids.append(aid) or {"success": True, "status": "SUBMITTED", "alpha_id": aid})  # type: ignore[method-assign]
    client.flush_daily_candidates(day_str="2026-11-25", max_daily_submissions=1)

    assert submitted_ids == ["BRONZE_SUB"]
    assert client.vault_xlsx_path.exists()

    sheets = _read_xlsx_sheets(client.vault_xlsx_path)
    assert set(sheets.keys()) >= {"Summary", "All Unsubmitted Backups", "Consultant Reserve", "Bronze Submittable"}

    all_unsub_ids = [r[0] for r in sheets["All Unsubmitted Backups"][1:]]
    reserve_ids = [r[0] for r in sheets["Consultant Reserve"][1:]]
    bronze_ids = [r[0] for r in sheets["Bronze Submittable"][1:]]

    # Submitted alpha MUST be removed from all unsubmitted backup sheets
    assert "BRONZE_SUB" not in all_unsub_ids
    assert "BRONZE_SUB" not in bronze_ids
    # Consultant Reserve gem MUST be untouched and present
    assert reserve_ids == ["GEM_PROT"]
    assert bronze_ids == ["BRONZE_LEFT"]
    assert set(all_unsub_ids) == {"GEM_PROT", "BRONZE_LEFT"}

    # Sync snapshot also writes clean .xlsx
    status_path = tmp_path / "sync_out" / "brain_pending_status.json"
    sync_res = sync_vault_snapshot_to_drive(
        client._quota_db(),
        mirror_dir=tmp_path / "mirror",
        status_json_path=status_path,
    )
    assert sync_res.get("xlsx_written") is True
    sync_xlsx = status_path.parent / "worldquant_brain_qualified_alphas_latest.xlsx"
    assert sync_xlsx.exists()
    sync_sheets = _read_xlsx_sheets(sync_xlsx)
    assert "BRONZE_SUB" not in [r[0] for r in sync_sheets["All Unsubmitted Backups"][1:]]
