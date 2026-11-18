from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.headless_brain_miner import run_headless_miner


class _OfflineDummyClient:
    offline_mode = True
    is_authenticated = False


def test_headless_miner_exits_cleanly_in_offline_mode() -> None:
    stats = run_headless_miner(duration_minutes=0.01, brain_client=_OfflineDummyClient())  # type: ignore[arg-type]
    assert stats["mined"] == 0
    assert stats["submitted"] == 0
