#!/usr/bin/env python3
"""Read-only operational CLI plus manual simulation queue ingress."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from worldquant_brain.brain_scheduler import BrainDailyScheduler
from worldquant_brain.manual_priority_queue import ManualPriorityQueue


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=Path("worldquant_brain/alpha_vault.db"))
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("schedule")
    q = sub.add_parser("manual")
    q.add_argument("expression")
    q.add_argument("--requested-by", default="cli")
    q.add_argument("--note")
    q.add_argument("--priority", type=int, default=100)
    sub.add_parser("discovered")
    args = p.parse_args()
    if not args.db.exists():
        print(json.dumps({"ok": False, "error": "DB_NOT_FOUND", "db": str(args.db)}))
        return 2
    if args.command == "schedule":
        print(json.dumps(BrainDailyScheduler(args.db).status(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "manual":
        queue = ManualPriorityQueue(args.db)
        job, created = queue.enqueue(
            args.expression, requested_by=args.requested_by, manual_note=args.note,
            priority=args.priority, source="MANUAL", simulation_only=True,
        )
        print(json.dumps({
            "ok": True, "created": created, "job_id": job["job_id"],
            "queue_position": queue.queue_position(job["job_id"]),
            "priority": job["priority"], "simulation_only": bool(job["simulation_only"]),
        }, indent=2, ensure_ascii=False))
        return 0
    with sqlite3.connect(args.db) as conn:
        conn.row_factory = sqlite3.Row
        cols = {r[1] for r in conn.execute("PRAGMA table_info(alphas)")}
        sim_col = "brain_simulation_id" if "brain_simulation_id" in cols else "NULL"
        rows = conn.execute(f"""
            SELECT id, name, expression, sharpe, fitness, turnover, margin, status,
                   {sim_col} AS brain_simulation_id, created_at
            FROM alphas
            WHERE status IN ('BRAIN_COMPLETE','IS_CHECK_PASSED','CORRELATION_PASSED',
                             'SUBMISSION_CANDIDATE','SUBMITTING','SUBMITTED')
            ORDER BY created_at DESC
        """).fetchall()
    print(json.dumps({"count": len(rows), "alphas": [dict(r) for r in rows]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
