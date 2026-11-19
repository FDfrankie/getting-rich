#!/usr/bin/env python3
"""Headless WorldQuant BRAIN Alpha Miner for GitHub Actions / Cron / VPS."""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from worldquant_brain.alpha_miner import ContinuousAlphaMiner
from worldquant_brain.alpha_vault import AlphaVault
from worldquant_brain.brain_client import BrainClient
from worldquant_brain.brain_job_queue import BrainJobQueue


def run_headless_miner(
    *,
    duration_minutes: float = 38.0,
    target_queue_depth: int = 6,
    num_workers: int = 2,
    brain_client: BrainClient | None = None,
) -> dict[str, int]:
    vault = AlphaVault()
    client = brain_client or BrainClient()
    miner = ContinuousAlphaMiner()
    stats = {"mined": 0, "completed": 0, "accepted": 0, "submitted": 0, "refined": 0, "failed": 0}
    refined_keys: set[str] = set()

    if client.offline_mode:
        print("[ERROR] BRAIN_EMAIL / BRAIN_PASSWORD not configured (offline_mode=True). Exiting.")
        return stats

    if not client.is_authenticated:
        ok = client.authenticate()
        if not ok:
            print("[ERROR] WorldQuant BRAIN authentication returned False.")
            return stats

    def on_complete(job) -> None:
        stats["completed"] += 1
        m = job.metrics or {}
        sharpe = m.get("sharpe")
        fitness = m.get("fitness")
        turnover = m.get("turnover") if m.get("turnover") is not None else 0.25
        margin = m.get("margin")
        checks_passed = m.get("checks_passed", True)
        failed_checks = m.get("failed_checks") or []
        alpha_id = job.alpha_id
        if sharpe is None or fitness is None or not alpha_id:
            return

        norm_margin = (
            float(margin) / 10000.0
            if (margin is not None and float(margin) > 0.5)
            else (float(margin) if margin is not None else None)
        )
        qualified = (
            checks_passed
            and (norm_margin is None or norm_margin >= 0.0010)
            and vault.passes_qualification(float(sharpe), float(fitness), float(turnover))
        )
        if not qualified:
            expr = str(getattr(job, "expression", "") or "").strip()
            if expr and expr not in refined_keys and hasattr(miner, "generate_near_miss_refinements"):
                refined_keys.add(expr)
                variants = miner.generate_near_miss_refinements(
                    expr,
                    getattr(job, "settings", None),
                    sharpe=float(sharpe),
                    fitness=float(fitness),
                    turnover=float(turnover),
                    margin=margin,
                    failed_checks=failed_checks,
                    max_variants=3,
                )
                for v in variants:
                    refined_keys.add(v["expression"])
                    _, is_new = queue.create_job(v["expression"], settings=v["settings"])
                    if is_new:
                        stats["refined"] += 1
                        print(f"[REFINE] Enqueued {v['name']} ({v['reason']})")
            return

        stats["accepted"] += 1
        auto_submit = os.getenv("BRAIN_AUTO_SUBMIT_ENABLED", "true").lower() == "true"
        high_bar = float(os.getenv("BRAIN_HIGH_BAR_SHARPE", "1.80"))
        submitted_ok = False
        if auto_submit and float(sharpe) >= high_bar:
            sub = client.submit_alpha(str(alpha_id), enforce_checks=True)
            if sub.get("success"):
                submitted_ok = True
                stats["submitted"] += 1
                print(f"[SUBMITTED] Alpha {alpha_id} Sharpe={sharpe} Fitness={fitness}")
            elif sub.get("corr_rejected") or "CORR" in str(sub.get("error") or "").upper():
                print(f"[CORR_REJECTED_SKIPPED] Alpha {alpha_id}: {sub.get('error')}")
                return

        if not submitted_ok:
            vault.save_alpha(
                name=f"BRAIN_{alpha_id}",
                expression=job.expression,
                sharpe=float(sharpe),
                fitness=float(fitness),
                turnover=float(turnover),
                margin=float(margin or 0.0),
                category="BRAIN_COMPLETE",
                notes=f"alpha_id={alpha_id} (headless miner)",
                status_override="BRAIN_COMPLETE",
            )
            if auto_submit:
                client.stage_candidate_for_daily_selection(
                    str(alpha_id), job.expression, float(sharpe), float(fitness), float(turnover), float(margin or 0.0)
                )
            print(f"[SAVED] Alpha {alpha_id} Sharpe={sharpe} Fitness={fitness}")

    def on_fail(job) -> None:
        stats["failed"] += 1
        print(f"[FAIL] {job.job_id}: {job.safe_error}")

    queue = BrainJobQueue(
        client,
        on_complete_callback=on_complete,
        on_fail_callback=on_fail,
        db_path=vault.db_path,
        num_workers=num_workers,
    )

    deadline = time.monotonic() + max(0.05, float(duration_minutes)) * 60.0
    try:
        while time.monotonic() < deadline:
            while queue.queue_depth() < target_queue_depth and time.monotonic() < deadline:
                cand = miner.generate_unique_candidate()
                sim_settings = miner.generate_candidate_settings()
                _, is_new = queue.create_job(cand["expression"], settings=sim_settings)
                if is_new:
                    stats["mined"] += 1
                else:
                    break
            time.sleep(2.0)

        if os.getenv("BRAIN_AUTO_SUBMIT_ENABLED", "true").lower() == "true":
            taipei_now = datetime.now(timezone(timedelta(hours=8)))
            if taipei_now.hour >= int(os.getenv("BRAIN_DAILY_FLUSH_HOUR", "22")):
                flushed = client.flush_daily_candidates()
                for sr in flushed:
                    if sr.get("success"):
                        stats["submitted"] += 1
                        print(f"[FLUSH_SUBMITTED] {sr.get('alpha_id')}")
    finally:
        queue.stop()
        queue.join(timeout=10.0)

    print(f"[SUMMARY] {stats}")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Headless WorldQuant BRAIN Alpha Miner")
    parser.add_argument("--duration-minutes", type=float, default=38.0)
    parser.add_argument("--queue-depth", type=int, default=6)
    parser.add_argument("--workers", type=int, default=int(os.getenv("BRAIN_QUEUE_WORKERS", "2")))
    args = parser.parse_args()
    run_headless_miner(
        duration_minutes=args.duration_minutes,
        target_queue_depth=args.queue_depth,
        num_workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
