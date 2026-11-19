"""Batch/single CLI scorer for Alpha PnL Jump & Neutralization diagnostics (zero external deps)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from worldquant_brain.pnl_jump_scoring import (
    build_pnl_jump_input_from_trusted_record,
    evaluate_pnl_jump,
)


def _finite_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(v) for v in value]
    return value


def score_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    inp = build_pnl_jump_input_from_trusted_record(payload)
    rep = evaluate_pnl_jump(inp)
    return _finite_json({
        "alpha_id": rep.alpha_id,
        "blocked": rep.blocked,
        "report": rep.to_dict(),
    })


def main() -> int:
    parser = argparse.ArgumentParser(description="Score Alpha candidate JSON file(s) using pnl_jump_scoring.")
    parser.add_argument("input", type=Path, help="Candidate JSON file or directory of *.json files")
    parser.add_argument("--out-dir", type=Path, default=None, help="Optional output directory for batch *.scored.json")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args()

    if args.input.is_dir():
        files = sorted(args.input.glob("*.json"))
        results = []
        if args.out_dir is not None:
            args.out_dir.mkdir(parents=True, exist_ok=True)
        for p in files:
            res = score_file(p)
            results.append(res)
            if args.out_dir is not None:
                out_path = args.out_dir / f"{p.stem}.scored.json"
                out_path.write_text(
                    json.dumps(res, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8",
                )
        print(json.dumps({"count": len(results), "results": results}, ensure_ascii=False, indent=2 if args.pretty else None, allow_nan=False))
        return 0

    res = score_file(args.input)
    print(json.dumps(res, ensure_ascii=False, indent=2 if args.pretty else None, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
