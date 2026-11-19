#!/usr/bin/env python3
"""Standalone High-Throughput Multi-Simulation WorldQuant BRAIN Alpha Miner.

Closed-loop workflow:
  1. Generate & batch-simulate via Multi-Simulation API using experience-guided
     TwoStageAlphaFunnel (learns from past PASSED / REJECTED / SUBMIT_FAILED).
  2. Three-way verdict per simulated alpha (no pre-filtering on self-correlation):
       - PASSED -> Saved to AlphaVault (app list) and staged in daily Pending pool
         waiting for end-of-day (22:00 Taipei) submission.
       - NEAR_MISS -> Promoted for Stage 2 fine-tuning (sign-flip, neut/decay/trunc, smoothing).
       - REJECTED -> Recorded in funnel_reference_db with explicit reason.
  3. End-of-day flush (`flush_daily_candidates`):
       - Submits highest-scoring pending alphas.
       - If submission fails (including self-correlation), records SUBMIT_FAILED + reason
         in funnel_reference_db and removes the alpha from AlphaVault (app list).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env", override=False)
except ImportError:
    pass

from worldquant_brain.alpha_vault import AlphaVault
from worldquant_brain.brain_client import BrainClient
from worldquant_brain.multi_sim_client import fetch_dataset_fields, simulate_multi_alpha
from worldquant_brain.two_stage_funnel import TwoStageAlphaFunnel, classify_verdict

DEFAULT_DATASETS: tuple[str, ...] = (
    "analyst4",
    "fundamental6",
    "model16",
    "option8",
    "news12",
    "sentiment1",
)


def _is_rate_or_auth_cooldown(text: str) -> bool:
    low = (text or "").lower()
    return "429" in low or "rate" in low or "captcha" in low or "authentication" in low


def run_headless_miner(
    *,
    duration_minutes: float = 38.0,
    target_queue_depth: int = 6,
    num_workers: int = 2,
    batch_size: int = 5,
    datasets: Sequence[str] | None = None,
    seed_sharpe_threshold: float = 0.90,
    brain_client: BrainClient | None = None,
    vault_db_path: str | Path | None = None,
) -> dict[str, int]:
    """Run standalone Multi-Simulation Two-Stage Alpha Miner.

    Pass duration_minutes=0 for continuous 24/7 daemon mode.
    """
    vault = AlphaVault(str(vault_db_path)) if vault_db_path else AlphaVault()
    client = brain_client or BrainClient()
    stats = {
        "mined": 0,
        "batches": 0,
        "completed": 0,
        "seeds_promoted": 0,
        "accepted": 0,
        "rejected": 0,
        "submitted": 0,
        "submit_failed_deleted": 0,
        "refined": 0,
        "failed": 0,
    }

    if getattr(client, "offline_mode", True):
        print("[ERROR] BRAIN_EMAIL / BRAIN_PASSWORD not configured (offline_mode=True). Exiting.")
        return stats

    if not getattr(client, "is_authenticated", False):
        if not client.authenticate():
            print("[ERROR] WorldQuant BRAIN authentication returned False.")
            return stats

    ds_list = list(datasets) if datasets else list(DEFAULT_DATASETS)
    funnel = TwoStageAlphaFunnel(vault.db_path, seed_sharpe_threshold=seed_sharpe_threshold)

    datasets_fields: dict[str, list[str]] = {}
    for ds_id in ds_list:
        fields = fetch_dataset_fields(client, ds_id, limit=80)
        if fields:
            datasets_fields[ds_id] = fields
            print(f"[DISCOVERY] Dataset {ds_id}: loaded {len(fields)} candidate fields")

    if not datasets_fields:
        print("[WARN] No dataset fields available to screen. Exiting.")
        return stats

    eff_workers = max(1, int(num_workers))
    eff_batch = max(2, min(10, int(batch_size)))
    infinite = float(duration_minutes) <= 0.0
    deadline = float("inf") if infinite else (time.monotonic() + max(0.05, float(duration_minutes)) * 60.0)
    auto_submit = os.getenv("BRAIN_AUTO_SUBMIT_ENABLED", "true").lower() == "true"
    flush_hour = int(os.getenv("BRAIN_DAILY_FLUSH_HOUR", "22") or "22")
    max_daily = int(os.getenv("BRAIN_MAX_DAILY_SUBMISSIONS", "2") or "2")

    def _remove_from_app_list(alpha_id: str) -> None:
        for a in vault.list_alphas():
            if a.get("brain_alpha_id") == alpha_id or a.get("name") == f"BRAIN_{alpha_id}":
                vault.delete_alpha(int(a["id"]))

    def _handle_batch_results(results: list[dict[str, Any]]) -> bool:
        hit_cooldown = False
        ing = funnel.ingest_results(results)
        stats["seeds_promoted"] += ing["promoted"]
        stats["rejected"] += ing["rejected"]
        if ing["promoted"] > 0:
            print(f"[NEAR_MISS_PROMOTE] Promoted {ing['promoted']} near-miss seed(s) for fine-tuning")

        for r in results:
            if not r.get("success"):
                stats["failed"] += 1
                if _is_rate_or_auth_cooldown(str(r.get("error") or "")):
                    hit_cooldown = True
                continue
            stats["completed"] += 1
            if r.get("stage") == "STAGE2":
                stats["refined"] += 1

            alpha_id = r.get("alpha_id")
            sharpe = r.get("sharpe")
            fitness = r.get("fitness")
            turnover = r.get("turnover") if r.get("turnover") is not None else 0.25
            margin = r.get("margin")
            if not alpha_id or sharpe is None or fitness is None:
                continue

            verdict, _reason = classify_verdict(r, near_miss_sharpe=seed_sharpe_threshold)
            norm_margin = (
                float(margin) / 10000.0
                if (margin is not None and float(margin) > 0.5)
                else (float(margin) if margin is not None else None)
            )
            qualified = (
                verdict == "PASSED"
                and (norm_margin is None or norm_margin >= 0.0010)
                and vault.passes_qualification(float(sharpe), float(fitness), float(turnover))
            )
            if not qualified:
                continue

            # Ignore self-correlation upfront: keep in AlphaVault (app list) and stage into daily Pending pool
            stats["accepted"] += 1
            vault.save_alpha(
                name=f"BRAIN_{alpha_id}",
                expression=str(r.get("expression") or ""),
                sharpe=float(sharpe),
                fitness=float(fitness),
                turnover=float(turnover),
                margin=float(margin or 0.0),
                category="BRAIN_COMPLETE",
                notes=f"alpha_id={alpha_id} ({r.get('stage')}/{r.get('dataset_id')})",
                status_override="BRAIN_COMPLETE",
                brain_alpha_id=str(alpha_id),
                is_check_status="PASS",
            )
            if auto_submit:
                client.stage_candidate_for_daily_selection(
                    str(alpha_id),
                    str(r.get("expression") or ""),
                    float(sharpe),
                    float(fitness),
                    float(turnover),
                    float(margin or 0.0),
                )
            print(f"[PENDING_STAGED] Alpha {alpha_id} ({r.get('stage')}) Sharpe={sharpe:.2f} Fitness={fitness:.2f}")

        return hit_cooldown

    def _flush_pending_if_due() -> None:
        if not auto_submit:
            return
        taipei_now = datetime.now(timezone(timedelta(hours=8)))
        if taipei_now.hour < flush_hour or client.get_daily_submission_count() >= max_daily:
            return
        flushed = client.flush_daily_candidates(max_daily_submissions=max_daily)
        for sr in (flushed or []):
            aid = str(sr.get("alpha_id") or "")
            if sr.get("success"):
                stats["submitted"] += 1
                funnel.mark_submitted(alpha_id=aid)
                _remove_from_app_list(aid)
                print(f"[FLUSH_SUBMITTED] Alpha {aid} submitted & removed from pending list")
            else:
                st = str(sr.get("status") or "SUBMIT_FAILED")
                if st in ("DAILY_QUOTA_REACHED", "OFFLINE"):
                    continue
                reason = str(sr.get("error") or ", ".join(sr.get("failed_checks") or []) or st)
                funnel.mark_submit_failed(alpha_id=aid, reason=f"{st}:{reason}")
                _remove_from_app_list(aid)
                stats["submit_failed_deleted"] += 1
                print(f"[FLUSH_FAILED_REMOVED] Alpha {aid} removed from app list ({reason})")

    in_flight: list[Future[list[dict[str, Any]]]] = []
    cooldown_until = 0.0
    auth_blocked = False
    with ThreadPoolExecutor(max_workers=eff_workers, thread_name_prefix="brain-multisim") as pool:
        try:
            while time.monotonic() < deadline and not auth_blocked:
                still_running: list[Future[list[dict[str, Any]]]] = []
                for fut in in_flight:
                    if fut.done():
                        try:
                            batch_res = fut.result()
                            if any(
                                str(r.get("status") or "").upper() == "AUTH_FAILED"
                                or "captcha" in str(r.get("error") or "").lower()
                                for r in (batch_res or [])
                            ):
                                _handle_batch_results(batch_res)
                                auth_blocked = True
                            elif _handle_batch_results(batch_res):
                                cooldown_until = max(cooldown_until, time.monotonic() + 60.0)
                                print("[COOLDOWN] Hit BRAIN 429/auth throttle; pausing new dispatches for 60s")
                        except Exception as exc:
                            stats["failed"] += 1
                            print(f"[BATCH_ERROR] {exc}")
                            if "captcha" in str(exc).lower():
                                auth_blocked = True
                            elif _is_rate_or_auth_cooldown(str(exc)):
                                cooldown_until = max(cooldown_until, time.monotonic() + 60.0)
                    else:
                        still_running.append(fut)
                in_flight = still_running

                if auth_blocked:
                    print("[AUTH_COOLDOWN] Authentication rate-limited or CAPTCHA required by WorldQuant BRAIN; stopping headless miner to protect account.")
                    break

                exhausted = False
                if time.monotonic() >= cooldown_until:
                    while len(in_flight) < eff_workers and time.monotonic() < deadline:
                        batch = funnel.next_batch(datasets_fields, batch_size=eff_batch)
                        if not batch:
                            exhausted = True
                            break
                        stats["mined"] += len(batch)
                        stats["batches"] += 1
                        stage_tag = batch[0].get("stage", "STAGE1")
                        ds_tag = batch[0].get("dataset_id", "")
                        print(f"[MULTI_SIM] Dispatching {stage_tag} batch #{stats['batches']} ({len(batch)} alphas, ds={ds_tag})")
                        in_flight.append(pool.submit(simulate_multi_alpha, client, batch))

                _flush_pending_if_due()

                if exhausted and not in_flight:
                    print("[DONE] All configured dataset fields and seeds exhausted.")
                    break
                time.sleep(1.5)
        finally:
            for fut in in_flight:
                if auth_blocked:
                    fut.cancel()
                if fut.cancelled():
                    continue
                try:
                    _handle_batch_results(fut.result(timeout=5.0 if auth_blocked else 15.0))
                except Exception:
                    pass
            if not auth_blocked:
                _flush_pending_if_due()

    print(f"[SUMMARY] {stats} | funnel={funnel.summary()}")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone Multi-Simulation WorldQuant BRAIN Alpha Miner")
    parser.add_argument("--duration-minutes", type=float, default=0.0, help="0 = run 24/7 continuously")
    parser.add_argument("--workers", type=int, default=int(os.getenv("BRAIN_MULTI_SIM_WORKERS", "2")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BRAIN_MULTI_SIM_BATCH_SIZE", "5")))
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--seed-sharpe", type=float, default=0.90)
    args = parser.parse_args()
    ds_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    run_headless_miner(
        duration_minutes=args.duration_minutes,
        num_workers=args.workers,
        batch_size=args.batch_size,
        datasets=ds_list,
        seed_sharpe_threshold=args.seed_sharpe,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
