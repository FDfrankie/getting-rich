#!/usr/bin/env python3
"""Standalone High-Throughput Multi-Simulation WorldQuant BRAIN Alpha Miner.

Decoupled from web_ui/server.py and crypto_trading so Web UI or trading bot restarts
never interrupt in-flight BRAIN simulations. Uses:
  1. WorldQuant BRAIN Multi-Simulation API (batch_size 5-10 per slot x workers concurrent slots)
  2. Two-Stage Discovery Funnel (Non-PV Dataset Field Screening -> Seed Parameter Grid Refinement)
  3. Automatic AlphaVault persistence, daily Top-K candidate staging, and 22:00 Taipei flush
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
from worldquant_brain.two_stage_funnel import TwoStageAlphaFunnel

DEFAULT_DATASETS: tuple[str, ...] = (
    "analyst4",
    "fundamental6",
    "model16",
    "option8",
    "news12",
    "sentiment1",
)


def _is_auth_block_message(text: str) -> bool:
    low = (text or "").lower()
    return "captcha" in low or "429" in low or "authentication" in low


def _batch_hit_auth_block(results: list[dict[str, Any]]) -> bool:
    if not results:
        return False
    statuses = [str(r.get("status") or "").upper() for r in results]
    if any(s == "AUTH_FAILED" for s in statuses):
        return True
    if all(s in ("AUTH_FAILED", "SUBMIT_ERROR") for s in statuses):
        return any(_is_auth_block_message(str(r.get("error") or "")) for r in results)
    return False


def run_headless_miner(
    *,
    duration_minutes: float = 38.0,
    target_queue_depth: int = 6,
    num_workers: int = 3,
    batch_size: int = 8,
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
        "submitted": 0,
        "refined": 0,
        "failed": 0,
    }

    if getattr(client, "offline_mode", True):
        print("[ERROR] BRAIN_EMAIL / BRAIN_PASSWORD not configured (offline_mode=True). Exiting.")
        return stats

    if not getattr(client, "is_authenticated", False):
        try:
            auth_ok = bool(client.authenticate())
        except Exception as exc:
            if _is_auth_block_message(str(exc)):
                print(f"[AUTH_COOLDOWN] Authentication rate-limited or CAPTCHA required by WorldQuant BRAIN; stopping headless miner to protect account. ({exc})")
            else:
                print(f"[ERROR] WorldQuant BRAIN authentication error: {exc}")
            return stats
        if not auth_ok:
            print("[ERROR] WorldQuant BRAIN authentication returned False.")
            return stats

    ds_list = list(datasets) if datasets else list(DEFAULT_DATASETS)
    funnel = TwoStageAlphaFunnel(vault.db_path, seed_sharpe_threshold=seed_sharpe_threshold)

    # Load high-coverage fields for each target dataset once at startup
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
    high_bar = float(os.getenv("BRAIN_HIGH_BAR_SHARPE", "1.80"))
    flush_hour = int(os.getenv("BRAIN_DAILY_FLUSH_HOUR", "22") or "22")
    max_daily = int(os.getenv("BRAIN_MAX_DAILY_SUBMISSIONS", "2") or "2")

    def _handle_batch_results(results: list[dict[str, Any]]) -> None:
        ing = funnel.ingest_results(results)
        stats["seeds_promoted"] += ing["promoted"]
        if ing["promoted"] > 0:
            print(f"[STAGE2_PROMOTE] Promoted {ing['promoted']} seed(s) for grid refinement")

        for r in results:
            if not r.get("success"):
                stats["failed"] += 1
                continue
            stats["completed"] += 1
            if r.get("stage") == "STAGE2":
                stats["refined"] += 1

            alpha_id = r.get("alpha_id")
            sharpe = r.get("sharpe")
            fitness = r.get("fitness")
            turnover = r.get("turnover") if r.get("turnover") is not None else 0.25
            margin = r.get("margin")
            checks_passed = bool(r.get("checks_passed", False))
            if not alpha_id or sharpe is None or fitness is None:
                continue

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
                continue

            stats["accepted"] += 1
            submitted_ok = False
            corr_rejected = False
            if auto_submit and float(sharpe) >= high_bar:
                sub = client.submit_alpha(str(alpha_id), max_daily_submissions=max_daily, enforce_checks=True)
                if sub.get("success"):
                    submitted_ok = True
                    stats["submitted"] += 1
                    print(f"[SUBMITTED] Alpha {alpha_id} Sharpe={sharpe:.2f} Fitness={fitness:.2f}")
                elif sub.get("corr_rejected") or "CORR" in str(sub.get("error") or "").upper():
                    corr_rejected = True
                    print(f"[CORR_REJECTED_DELETED] Alpha {alpha_id}: {sub.get('error')}")

            if not submitted_ok and not corr_rejected:
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
                print(f"[SAVED] Alpha {alpha_id} ({r.get('stage')}) Sharpe={sharpe:.2f} Fitness={fitness:.2f}")

    in_flight: list[Future[list[dict[str, Any]]]] = []
    auth_blocked = False
    with ThreadPoolExecutor(max_workers=eff_workers, thread_name_prefix="brain-multisim") as pool:
        try:
            while time.monotonic() < deadline and not auth_blocked:
                # Reap completed futures
                still_running: list[Future[list[dict[str, Any]]]] = []
                for fut in in_flight:
                    if fut.done():
                        try:
                            batch_results = fut.result()
                            _handle_batch_results(batch_results)
                            if _batch_hit_auth_block(batch_results):
                                auth_blocked = True
                        except Exception as exc:
                            stats["failed"] += 1
                            print(f"[BATCH_ERROR] {exc}")
                            if _is_auth_block_message(str(exc)):
                                auth_blocked = True
                    else:
                        still_running.append(fut)
                in_flight = still_running

                if auth_blocked:
                    print("[AUTH_COOLDOWN] Authentication rate-limited or CAPTCHA required by WorldQuant BRAIN; stopping headless miner to protect account.")
                    break

                # Fill up to eff_workers concurrent Multi-Sim batches
                exhausted = False
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

                # Check daily 22:00 Taipei flush
                if auto_submit:
                    taipei_now = datetime.now(timezone(timedelta(hours=8)))
                    if taipei_now.hour >= flush_hour and client.get_daily_submission_count() < max_daily:
                        flushed = client.flush_daily_candidates(max_daily_submissions=max_daily)
                        for sr in flushed:
                            _aid = str(sr.get("alpha_id") or "")
                            if sr.get("success"):
                                stats["submitted"] += 1
                                for _a in vault.list_alphas():
                                    if _a.get("brain_alpha_id") == _aid or _a.get("name") == f"BRAIN_{_aid}":
                                        vault.delete_alpha(int(_a["id"]))
                                print(f"[FLUSH_SUBMITTED] {_aid}")
                            elif sr.get("corr_rejected") or "CORR" in str(sr.get("error") or "").upper():
                                for _a in vault.list_alphas():
                                    if _a.get("brain_alpha_id") == _aid or _a.get("name") == f"BRAIN_{_aid}":
                                        vault.delete_alpha(int(_a["id"]))
                                print(f"[FLUSH_CORR_DELETED] {_aid}: {sr.get('error')}")

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

    print(f"[SUMMARY] {stats} | funnel={funnel.summary()}")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone Multi-Simulation WorldQuant BRAIN Alpha Miner")
    parser.add_argument("--duration-minutes", type=float, default=0.0, help="0 = run 24/7 continuously")
    parser.add_argument("--workers", type=int, default=int(os.getenv("BRAIN_MULTI_SIM_WORKERS", "3")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BRAIN_MULTI_SIM_BATCH_SIZE", "8")))
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
