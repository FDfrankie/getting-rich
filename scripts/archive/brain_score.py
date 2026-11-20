#!/usr/bin/env python3
"""Inspect or record authenticated BRAIN score snapshots."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from worldquant_brain.score_tracker import ScoreTracker, parse_score_payload


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument('--db',type=Path,default=Path('worldquant_brain/alpha_vault.db'))
    p.add_argument('--record-json',type=Path)
    args=p.parse_args()
    tracker=ScoreTracker(args.db)
    if args.record_json:
        payload=json.loads(args.record_json.read_text(encoding='utf-8'))
        tracker.record(parse_score_payload(payload,source='MANUAL_AUTHENTICATED_EXPORT'))
    print(json.dumps(tracker.summary(),indent=2,ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())
