"""Dataset integrity manifests for cached OHLCV files."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_dataset_manifest(
    csv_path: str | Path,
    *, symbol: str,
    timeframe: str,
    source: str,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    path = Path(csv_path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    freq = pd.Timedelta(timeframe)
    expected_index = pd.date_range(idx.min(), idx.max(), freq=freq, tz="UTC") if len(idx) else pd.DatetimeIndex([])
    missing = expected_index.difference(idx)
    duplicates = int(idx.duplicated().sum())
    invalid_ohlc = 0
    if len(df):
        required = {"Open", "High", "Low", "Close", "Volume"}
        if required.issubset(df.columns):
            invalid_ohlc = int(((df["High"] < df[["Open", "Close", "Low"]].max(axis=1)) |
                                (df["Low"] > df[["Open", "Close", "High"]].min(axis=1)) |
                                (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1) |
                                (df["Volume"] < 0)).sum())
    inferred_expected = len(expected_index)
    manifest = {
        "schema_version": 1,
        "symbol": symbol,
        "timeframe": timeframe,
        "source": source,
        "start": idx.min().isoformat() if len(idx) else None,
        "end": idx.max().isoformat() if len(idx) else None,
        "expected_rows": int(expected_rows if expected_rows is not None else inferred_expected),
        "actual_rows": int(len(df)),
        "duplicate_timestamps": duplicates,
        "missing_intervals": int(len(missing)),
        "missing_timestamps": [x.isoformat() for x in missing[:1000]],
        "missing_timestamps_truncated": len(missing) > 1000,
        "invalid_ohlc_rows": invalid_ohlc,
        "closed_candles_only": True,
        "timezone": "UTC",
        "sha256": _sha256(path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "quality_passed": duplicates == 0 and invalid_ohlc == 0 and len(missing) == 0,
    }
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
