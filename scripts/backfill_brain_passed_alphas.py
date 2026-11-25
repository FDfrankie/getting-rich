"""P42 CLI: Full-history backfill of all passed (ACTIVE + UNSUBMITTED) WorldQuant BRAIN Alphas.

Usage:
    python scripts/backfill_brain_passed_alphas.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass

from worldquant_brain.brain_client import BrainClient


def export_vault_to_xlsx(vault_payload: dict, xlsx_path: Path) -> dict[str, int]:
    """Export ONLY active unsubmitted non-rejected qualified alphas to the backup spreadsheet.

    Primary sheets:
      - Summary
      - All Unsubmitted Backups (Consultant Reserve + Bronze Submittable ONLY; excludes SUBMITTED & FAILED_*)
      - Consultant Reserve (unsubmitted non-rejected)
      - Bronze Submittable (unsubmitted non-rejected, worst/dup-first)
    """
    summary = vault_payload.get("summary") or {}
    excluded_statuses = {"SUBMITTED", "FAILED_CORR", "FAILED_CHECKS", "SUBMIT_REJECTED"}

    def _is_active_unsubmitted(r: dict) -> bool:
        st = str(r.get("submit_status") or "").strip().upper()
        return st not in excluded_statuses

    reserve_rows = [r for r in (vault_payload.get("consultant_reserve") or []) if _is_active_unsubmitted(r)]
    bronze_rows = [r for r in (vault_payload.get("bronze_submittable_worst_first") or []) if _is_active_unsubmitted(r)]
    all_unsub_rows = [
        ("CONSULTANT_RESERVE", r) for r in reserve_rows
    ] + [
        ("BRONZE_SUBMITTABLE", r) for r in bronze_rows
    ]

    headers = [
        "Alpha ID", "Bucket", "Submit Status", "Sharpe", "Fitness", "Turnover",
        "Margin", "Returns", "Drawdown", "Sub-Univ Sharpe", "Region", "Universe",
        "Delay", "Neutralization", "Decay", "Truncation", "Source", "Expression",
    ]
    counts = {
        "all_unsubmitted": len(all_unsub_rows),
        "consultant_reserve": len(reserve_rows),
        "bronze_submittable": len(bronze_rows),
        "submitted_excluded": int(summary.get("submitted_count", len(vault_payload.get("submitted") or []))),
    }
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)

    def _row_values(b_name: str, r: dict) -> list:
        sett = r.get("settings") or {}
        return [
            r.get("alpha_id", ""),
            b_name,
            r.get("submit_status", ""),
            r.get("sharpe"),
            r.get("fitness"),
            r.get("turnover"),
            r.get("margin"),
            r.get("returns"),
            r.get("drawdown"),
            r.get("sub_universe_sharpe"),
            sett.get("region", "USA"),
            sett.get("universe", "TOP3000"),
            sett.get("delay", 1),
            sett.get("neutralization", ""),
            sett.get("decay", ""),
            sett.get("truncation", ""),
            r.get("source", ""),
            r.get("expression", ""),
        ]

    try:
        import openpyxl
        from openpyxl.styles import Alignment, Font, PatternFill

        wb = openpyxl.Workbook()
        ws_sum = wb.active
        ws_sum.title = "Summary"
        ws_sum.append(["WorldQuant BRAIN Active Unsubmitted Alpha Backup — Summary"])
        ws_sum.append(["Updated (Taipei)", vault_payload.get("updated_at_taipei", "")])
        ws_sum.append(["Active Unsubmitted Backups", counts["all_unsubmitted"]])
        ws_sum.append(["Consultant Reserve (Unsubmitted)", counts["consultant_reserve"]])
        ws_sum.append(["Bronze Submittable (Worst/Dup-First)", counts["bronze_submittable"]])
        ws_sum.append(["Submitted (Excluded from Backup Sheets)", counts["submitted_excluded"]])
        ws_sum["A1"].font = Font(bold=True, size=13)

        header_fill = PatternFill("solid", fgColor="1F4E78")
        header_font = Font(bold=True, color="FFFFFF")

        def _write_sheet_openpyxl(title: str, tagged_rows: list[tuple[str, dict]]) -> None:
            ws = wb.create_sheet(title)
            ws.append(headers)
            for col_idx in range(1, len(headers) + 1):
                c = ws.cell(1, col_idx)
                c.fill = header_fill
                c.font = header_font
                c.alignment = Alignment(horizontal="center")
            for b_name, r in tagged_rows:
                ws.append(_row_values(b_name, r))
            ws.freeze_panes = "A2"

        _write_sheet_openpyxl("All Unsubmitted Backups", all_unsub_rows)
        _write_sheet_openpyxl("Consultant Reserve", [("CONSULTANT_RESERVE", r) for r in reserve_rows])
        _write_sheet_openpyxl("Bronze Submittable", [("BRONZE_SUBMITTABLE", r) for r in bronze_rows])

        tmp_xlsx = xlsx_path.parent / f".{xlsx_path.name}.tmp.xlsx"
        wb.save(tmp_xlsx)
        os.replace(tmp_xlsx, xlsx_path)
        return counts
    except ImportError:
        pass

    try:
        import xlsxwriter

        tmp_xlsx = xlsx_path.parent / f".{xlsx_path.name}.tmp.xlsx"
        wb = xlsxwriter.Workbook(str(tmp_xlsx))
        title_fmt = wb.add_format({"bold": True, "font_size": 13})
        hdr_fmt = wb.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#1F4E78", "align": "center"})

        ws_sum = wb.add_worksheet("Summary")
        sum_rows = [
            ["WorldQuant BRAIN Active Unsubmitted Alpha Backup — Summary"],
            ["Updated (Taipei)", vault_payload.get("updated_at_taipei", "")],
            ["Active Unsubmitted Backups", counts["all_unsubmitted"]],
            ["Consultant Reserve (Unsubmitted)", counts["consultant_reserve"]],
            ["Bronze Submittable (Worst/Dup-First)", counts["bronze_submittable"]],
            ["Submitted (Excluded from Backup Sheets)", counts["submitted_excluded"]],
        ]
        for r_idx, row_vals in enumerate(sum_rows):
            for c_idx, val in enumerate(row_vals):
                ws_sum.write(r_idx, c_idx, val, title_fmt if (r_idx == 0 and c_idx == 0) else None)

        def _write_sheet_xlsxwriter(title: str, tagged_rows: list[tuple[str, dict]]) -> None:
            ws = wb.add_worksheet(title)
            for c_idx, h in enumerate(headers):
                ws.write(0, c_idx, h, hdr_fmt)
            for out_row, (b_name, r) in enumerate(tagged_rows, start=1):
                for c_idx, val in enumerate(_row_values(b_name, r)):
                    ws.write(out_row, c_idx, val)
            ws.freeze_panes(1, 0)

        _write_sheet_xlsxwriter("All Unsubmitted Backups", all_unsub_rows)
        _write_sheet_xlsxwriter("Consultant Reserve", [("CONSULTANT_RESERVE", r) for r in reserve_rows])
        _write_sheet_xlsxwriter("Bronze Submittable", [("BRONZE_SUBMITTABLE", r) for r in bronze_rows])

        wb.close()
        os.replace(tmp_xlsx, xlsx_path)
    except ImportError:
        print("[WARN] Neither openpyxl nor xlsxwriter installed; skipping xlsx export.")
    return counts


def main() -> int:
    vault_json = ROOT / "observability" / "qualified_alpha_formulas_vault.json"
    if not vault_json.parent.exists():
        vault_json = ROOT / "qualified_alpha_formulas_vault.json"
    dp_vault_json = ROOT / "data_persistence" / "qualified_alpha_formulas_vault.json"
    xlsx_out = ROOT / "BRAIN_官方知識庫與研究紀錄" / "worldquant_brain_qualified_alphas_latest.xlsx"

    client = BrainClient()
    res = client.backfill_all_passed_alphas(max_pages=20, page_size=100, vault_json_path=vault_json)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if res.get("ok") and vault_json.exists():
        if dp_vault_json.parent.exists() and dp_vault_json.resolve() != vault_json.resolve():
            client.export_qualified_formula_vault(output_path=dp_vault_json)
        payload = json.loads(vault_json.read_text(encoding="utf-8"))
        export_vault_to_xlsx(payload, xlsx_out)
        print(f"[OK] Exported {res.get('total_archived')} qualified alphas to {xlsx_out}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
