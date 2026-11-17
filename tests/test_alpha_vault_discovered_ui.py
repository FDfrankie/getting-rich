from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from worldquant_brain.alpha_vault import AlphaVault


def test_verified_count_and_averages_use_same_rows(tmp_path: Path) -> None:
    vault = AlphaVault(db_path=str(tmp_path / "v.db"))
    vault.save_alpha("V1", "rank(close - open)", sharpe=2.15, fitness=2.05, turnover=0.22, status_override="VERIFIED")
    vault.save_alpha("V2", "rank(volume)", sharpe=1.76, fitness=1.65, turnover=0.35, status_override="VERIFIED")
    vault.save_alpha("V3", "-1 * ts_rank(close, 10)", sharpe=1.58, fitness=1.42, turnover=0.28, status_override="VERIFIED")

    # Add 4 BRAIN_COMPLETE records — must NOT corrupt VERIFIED averages
    vault.save_alpha("BRAIN_Grmw87qQ", "rank(vwap - close)", sharpe=1.33, fitness=1.17, turnover=0.021, margin=None, status_override="BRAIN_COMPLETE", brain_alpha_id="Grmw87qQ", force=True)
    vault.save_alpha("BRAIN_E5QwaqLG", "rank(high - low)", sharpe=1.31, fitness=1.14, turnover=0.024, margin=None, status_override="BRAIN_COMPLETE", brain_alpha_id="E5QwaqLG", force=True)
    vault.save_alpha("BRAIN_xAvxmLnp", "rank(open - close)", sharpe=1.27, fitness=1.03, turnover=0.026, margin=None, status_override="BRAIN_COMPLETE", brain_alpha_id="xAvxmLnp", force=True)
    vault.save_alpha("BRAIN_LLKpX0em", "rank(returns)", sharpe=1.26, fitness=1.02, turnover=0.026, margin=None, status_override="BRAIN_COMPLETE", brain_alpha_id="LLKpX0em", force=True)

    stats = vault.get_stats()
    assert stats["verified_count"] == 3
    assert stats["avg_sharpe"] == 1.83
    assert stats["avg_fitness"] == 1.71
    assert stats["avg_turnover_pct"] == 28.33
    assert stats["discovered_count"] == 4

    discovered = vault.list_discovered_alphas()
    assert len(discovered) == 4
    names = {a["name"] for a in discovered}
    assert names == {"BRAIN_Grmw87qQ", "BRAIN_E5QwaqLG", "BRAIN_xAvxmLnp", "BRAIN_LLKpX0em"}
    assert all(a["margin"] is None for a in discovered)


def test_schema_migration_preserves_existing_rows_and_backfills_hash(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE alphas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            expression TEXT NOT NULL,
            category TEXT DEFAULT 'Momentum',
            universe TEXT DEFAULT 'TOP3000',
            decay INTEGER DEFAULT 4,
            delay INTEGER DEFAULT 1,
            sharpe REAL NOT NULL,
            fitness REAL NOT NULL,
            turnover REAL NOT NULL,
            returns REAL DEFAULT 0.0,
            drawdown REAL DEFAULT 0.0,
            margin REAL DEFAULT 0.0,
            status TEXT DEFAULT 'VERIFIED',
            tags TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO alphas (name, expression, sharpe, fitness, turnover, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("Legacy1", "rank(close)", 1.5, 1.2, 0.2, "VERIFIED", "2026-11-14 00:00:00"),
    )
    conn.commit()
    conn.close()

    vault = AlphaVault(db_path=str(db_path))
    rows = vault.list_alphas()
    assert len(rows) == 1
    assert rows[0]["expression_hash"]
    assert "brain_alpha_id" in rows[0]
