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


def export_vault_to_xlsx(vault_payload: dict, xlsx_path: Path) -> None:
    summary = vault_payload.get("summary") or {}
    headers = [
        "Alpha ID", "Bucket", "Submit Status", "Sharpe", "Fitness", "Turnover",
        "Margin", "Returns", "Drawdown", "Sub-Univ Sharpe", "Region", "Universe",
        "Delay", "Neutralization", "Decay", "Truncation", "Source", "Expression",
    ]
    buckets = [
        ("SUBMITTED", vault_payload.get("submitted") or []),
        ("CONSULTANT_RESERVE", vault_payload.get("consultant_reserve") or []),
        ("BRONZE_SUBMITTABLE", vault_payload.get("bronze_submittable_worst_first") or []),
    ]
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import openpyxl
        from openpyxl.styles import Alignment, Font, PatternFill

        wb = openpyxl.Workbook()
        ws_sum = wb.active
        ws_sum.title = "Summary"
        ws_sum.append(["WorldQuant BRAIN Full Qualified Alpha Vault — Backfill Summary"])
        ws_sum.append(["Updated (Taipei)", vault_payload.get("updated_at_taipei", "")])
        ws_sum.append(["Total Archived", summary.get("total_archived", 0)])
        ws_sum.append(["Submitted (ACTIVE)", summary.get("submitted_count", 0)])
        ws_sum.append(["Consultant Reserve", summary.get("consultant_reserve_count", 0)])
        ws_sum.append(["Bronze Submittable (Worst-First)", summary.get("bronze_submittable_count", 0)])
        ws_sum["A1"].font = Font(bold=True, size=13)

        ws = wb.create_sheet("All Qualified Alphas")
        ws.append(headers)
        header_fill = PatternFill("solid", fgColor="1F4E78")
        header_font = Font(bold=True, color="FFFFFF")
        for col_idx in range(1, len(headers) + 1):
            c = ws.cell(1, col_idx)
            c.fill = header_fill
            c.font = header_font
            c.alignment = Alignment(horizontal="center")

        for b_name, rows in buckets:
            for r in rows:
                sett = r.get("settings") or {}
                ws.append([
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
                ])

        ws.freeze_panes = "A2"
        wb.save(xlsx_path)
        return
    except ImportError:
        pass

    try:
        import xlsxwriter

        wb = xlsxwriter.Workbook(str(xlsx_path))
        title_fmt = wb.add_format({"bold": True, "font_size": 13})
        hdr_fmt = wb.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#1F4E78", "align": "center"})

        ws_sum = wb.add_worksheet("Summary")
        sum_rows = [
            ["WorldQuant BRAIN Full Qualified Alpha Vault — Backfill Summary"],
            ["Updated (Taipei)", vault_payload.get("updated_at_taipei", "")],
            ["Total Archived", summary.get("total_archived", 0)],
            ["Submitted (ACTIVE)", summary.get("submitted_count", 0)],
            ["Consultant Reserve", summary.get("consultant_reserve_count", 0)],
            ["Bronze Submittable (Worst-First)", summary.get("bronze_submittable_count", 0)],
        ]
        for r_idx, row_vals in enumerate(sum_rows):
            for c_idx, val in enumerate(row_vals):
                ws_sum.write(r_idx, c_idx, val, title_fmt if (r_idx == 0 and c_idx == 0) else None)

        ws = wb.add_worksheet("All Qualified Alphas")
        for c_idx, h in enumerate(headers):
            ws.write(0, c_idx, h, hdr_fmt)
        out_row = 1
        for b_name, rows in buckets:
            for r in rows:
                sett = r.get("settings") or {}
                vals = [
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
                for c_idx, val in enumerate(vals):
                    ws.write(out_row, c_idx, val)
                out_row += 1
        ws.freeze_panes(1, 0)
        wb.close()
    except ImportError:
        print("[WARN] Neither openpyxl nor xlsxwriter installed; skipping xlsx export.")


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
