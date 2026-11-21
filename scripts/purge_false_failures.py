"""One-shot safe recovery script: release unsimulated API-failure entries (sharpe IS NULL) from funnel_tried."""
from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from worldquant_brain.alpha_vault import AlphaVault
from worldquant_brain.two_stage_funnel import TwoStageAlphaFunnel
from scripts.headless_brain_miner import sync_vault_snapshot_to_drive


def main() -> int:
    vault = AlphaVault()
    funnel = TwoStageAlphaFunnel(vault.db_path)
    res = funnel.purge_false_failures()
    print(f"[PURGE_FALSE_FAILURES] db={vault.db_path} result={res}")
    sync_res = sync_vault_snapshot_to_drive(vault.db_path)
    print(f"[DRIVE_SYNC] {sync_res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
