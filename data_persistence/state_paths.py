"""Resolve non-synced local SQLite state paths (avoiding Google Drive disk I/O locks) with safe backup."""
from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path


def is_cloud_synced_path(path: str | Path) -> bool:
    s = str(path).lower()
    return any(tok in s for tok in ("google drive", "我的雲端硬碟", "googledrive", "onedrive", "dropbox"))


def resolve_sqlite_path(
    env_var: str,
    default_filename: str,
    legacy_path: str | Path,
    *,
    local_state_dir: str | Path | None = None,
) -> str:
    """Return env override if set; otherwise use non-synced local dir on Windows cloud-synced repos."""
    override = os.getenv(env_var, "").strip()
    if override:
        return override

    legacy = Path(legacy_path)
    state_dir_env = os.getenv("QUANT_LOCAL_STATE_DIR", "").strip()
    if local_state_dir is not None:
        target_dir = Path(local_state_dir)
    elif state_dir_env:
        target_dir = Path(state_dir_env)
    elif os.name == "nt" and is_cloud_synced_path(legacy):
        target_dir = Path(r"C:\quant_envs\state")
    else:
        return str(legacy)

    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / default_filename
    if not target.exists() and legacy.exists():
        try:
            shutil.copy2(legacy, target)
        except Exception:
            return str(legacy)
    return str(target)


def backup_sqlite_to_drive(local_db_path: str | Path, backup_dir: str | Path, filename: str | None = None) -> str | None:
    """Safely snapshot a local SQLite DB back to Google Drive using sqlite3.backup()."""
    src_path = Path(local_db_path)
    if not src_path.exists() or str(local_db_path) == ":memory:":
        return None
    dest_dir = Path(backup_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / (filename or src_path.name)
    tmp_path = dest_dir / f".{dest_path.name}.tmp"
    src_conn = sqlite3.connect(str(src_path), timeout=10.0)
    dst_conn = sqlite3.connect(str(tmp_path), timeout=10.0)
    try:
        src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()
    os.replace(tmp_path, dest_path)
    return str(dest_path)
