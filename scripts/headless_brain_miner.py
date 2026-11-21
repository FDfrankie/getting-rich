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
            _stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
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
from worldquant_brain.multi_sim_client import fetch_available_datasets, fetch_dataset_fields, simulate_multi_alpha
from worldquant_brain.local_cs_screener import LocalCrossSectionalScreener, is_price_volume_only
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



def sync_vault_snapshot_to_drive(
    vault_db_path: str | Path,
    *,
    mirror_dir: str | Path | None = None,
    status_json_path: str | Path | None = None,
    miner_stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Safely mirror local SQLite vault to Google Drive via sqlite3.backup() and write atomic pending status JSON."""
    import json
    import sqlite3
    from datetime import datetime, timezone, timedelta

    src = Path(vault_db_path)
    out: dict[str, Any] = {"db_mirrored": False, "status_written": False, "pending_count": 0}
    if str(vault_db_path) == ":memory:" or not src.exists():
        return out

    target_dir = Path(mirror_dir) if mirror_dir is not None else (BASE_DIR / "worldquant_brain")
    target_db = target_dir / "alpha_vault.db"
    try:
        if src.resolve() != target_db.resolve():
            from data_persistence.state_paths import backup_sqlite_to_drive
            mirrored = backup_sqlite_to_drive(src, target_dir, filename="alpha_vault.db")
            out["db_mirrored"] = bool(mirrored)
    except Exception as exc:
        print(f"[DRIVE_SYNC_WARN] SQLite mirror failed: {exc}")

    # Build lightweight atomic JSON snapshot of candidate pool & pending alphas
    counts: dict[str, int] = {}
    pending_alphas: list[dict[str, Any]] = []
    historical_tried_total = 0
    historical_ref_counts: dict[str, int] = {}
    try:
        conn = sqlite3.connect(str(src), timeout=5.0)
        try:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            if "funnel_tried" in tables:
                historical_tried_total = int(conn.execute("SELECT COUNT(*) FROM funnel_tried").fetchone()[0] or 0)
            elif "brain_tried_expressions" in tables:
                historical_tried_total = int(conn.execute("SELECT COUNT(*) FROM brain_tried_expressions").fetchone()[0] or 0)
            if "funnel_reference_db" in tables:
                for oc, cnt in conn.execute("SELECT outcome, COUNT(*) FROM funnel_reference_db GROUP BY outcome").fetchall():
                    historical_ref_counts[str(oc)] = int(cnt)
            if "brain_candidate_pool" in tables:
                for st, cnt in conn.execute(
                    "SELECT status, COUNT(*) FROM brain_candidate_pool GROUP BY status"
                ).fetchall():
                    counts[str(st)] = int(cnt)
                rows = conn.execute(
                    "SELECT alpha_id, sharpe, fitness, turnover, margin, score, candidate_date, created_at "
                    "FROM brain_candidate_pool WHERE UPPER(status)='PENDING' ORDER BY score DESC"
                ).fetchall()
                for r in rows:
                    pending_alphas.append({
                        "alpha_id": r[0],
                        "sharpe": r[1],
                        "fitness": r[2],
                        "turnover": r[3],
                        "margin": r[4],
                        "score": r[5],
                        "candidate_date": r[6],
                        "created_at": r[7],
                    })
        finally:
            conn.close()
    except Exception as exc:
        print(f"[DRIVE_SYNC_WARN] Status query failed: {exc}")

    now_utc = datetime.now(timezone.utc)
    now_tpe = now_utc.astimezone(timezone(timedelta(hours=8)))
    payload = {
        "updated_at_utc": now_utc.isoformat(),
        "updated_at_taipei": now_tpe.strftime("%Y-%m-%d %H:%M:%S+08:00"),
        "source_db": str(src),
        "pending_count": len(pending_alphas),
        "candidate_pool_counts": counts,
        "historical_tried_total": historical_tried_total,
        "historical_reference_counts": historical_ref_counts,
        "pending_alphas": pending_alphas,
        "miner_stats": dict(miner_stats or {}),
    }
    status_path = Path(status_json_path) if status_json_path is not None else (BASE_DIR / "data_persistence" / "brain_pending_status.json")
    try:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_json = status_path.parent / f".{status_path.name}.tmp"
        tmp_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_json, status_path)
        out["status_written"] = True
        out["pending_count"] = len(pending_alphas)
    except Exception as exc:
        print(f"[DRIVE_SYNC_WARN] Status JSON write failed: {exc}")
    return out


def run_headless_miner(
    *,
    duration_minutes: float = 38.0,
    target_queue_depth: int = 6,
    num_workers: int = 2,
    batch_size: int = 5,
    datasets: Sequence[str] | None = None,
    seed_sharpe_threshold: float = 1.15,
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

    if datasets:
        ds_list = list(datasets)
    else:
        discovered = fetch_available_datasets(client)
        if discovered:
            ds_list = [d["id"] for d in discovered]
            print("[AUTO_DATASETS] Discovered from BRAIN /data-sets (by valueScore & uncrowdedness):", ", ".join(f"{d['id']}(vs={d['value_score']:g},alphas={d['alpha_count']})" for d in discovered[:10]))
        else:
            ds_list = list(DEFAULT_DATASETS)
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
    submit_immediately = os.getenv("BRAIN_SUBMIT_IMMEDIATELY", "false").lower() == "true"
    flush_hour = int(os.getenv("BRAIN_DAILY_FLUSH_HOUR", "22") or "22")
    max_daily = int(os.getenv("BRAIN_MAX_DAILY_SUBMISSIONS", "2") or "2")
    enable_local_pv = os.getenv("BRAIN_LOCAL_PV_SCREENING", "true").lower() == "true"
    allow_pv_dl = os.getenv("BRAIN_LOCAL_PV_AUTO_DOWNLOAD", "false").lower() == "true"
    local_screener = LocalCrossSectionalScreener() if enable_local_pv else None
    pv_panel_ready = bool(local_screener and local_screener.has_panel())
    if local_screener and not pv_panel_ready and allow_pv_dl:
        pv_panel_ready = local_screener.load_or_fetch_panel(allow_download=True) is not None
    print(
        f"[MINER_CONFIG] vault_db={vault.db_path} auto_submit={auto_submit} "
        f"submit_immediately={submit_immediately} flush_hour={flush_hour} max_daily={max_daily} "
        f"local_pv_screener={enable_local_pv} pv_panel_ready={pv_panel_ready}"
    )
    _last_flush_wait_hour: dict[str, int | None] = {"hour": None}
    _sync_interval = max(30.0, float(os.getenv("BRAIN_DRIVE_SYNC_INTERVAL_SEC", "180") or "180"))
    _sync_enabled = os.getenv("BRAIN_DRIVE_SYNC_ENABLED", "true").lower() == "true" and vault_db_path is None
    _last_drive_sync: dict[str, float] = {"ts": 0.0}

    def _maybe_sync_to_drive(force: bool = False) -> None:
        if not _sync_enabled:
            return
        now_ts = time.monotonic()
        if not force and (now_ts - _last_drive_sync["ts"]) < _sync_interval:
            return
        _last_drive_sync["ts"] = now_ts
        res = sync_vault_snapshot_to_drive(vault.db_path, miner_stats=stats)
        if res.get("status_written"):
            print(f"[DRIVE_SYNC] Mirrored vault={res.get('db_mirrored')} pending={res.get('pending_count')}")

    _maybe_sync_to_drive(force=True)

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

        print(
            f"[BATCH_RESULT] batch_size={len(results)} | totals: mined={stats['mined']} "
            f"completed={stats['completed']} failed={stats['failed']} "
            f"accepted={stats['accepted']} submitted={stats['submitted']}"
        )
        return hit_cooldown

    def _flush_pending_if_due() -> None:
        if not auto_submit:
            return
        taipei_now = datetime.now(timezone(timedelta(hours=8)))
        if client.get_daily_submission_count() >= max_daily:
            return
        if not submit_immediately and flush_hour >= 0 and taipei_now.hour < flush_hour:
            if _last_flush_wait_hour["hour"] != taipei_now.hour:
                _last_flush_wait_hour["hour"] = taipei_now.hour
                print(
                    f"[FLUSH_WAITING] Taipei time {taipei_now.strftime('%H:%M')} < "
                    f"BRAIN_DAILY_FLUSH_HOUR={flush_hour}; staged alphas remain in candidate pool "
                    f"(set BRAIN_SUBMIT_IMMEDIATELY=true or BRAIN_DAILY_FLUSH_HOUR=-1 to submit right away)"
                )
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
                st = str(sr.get("status") or "SUBMIT_ERROR")
                reason = str(sr.get("error") or ", ".join(sr.get("failed_checks") or []) or st)
                permanent_fail = st in ("SUBMIT_REJECTED", "SUBMISSION_CHECK_FAILED")
                if not permanent_fail:
                    print(f"[FLUSH_TRANSIENT_KEEP] Alpha {aid} kept PENDING ({st}: {reason})")
                    if st in ("DAILY_QUOTA_REACHED", "OFFLINE", "AUTH_FAILED"):
                        break
                    continue
                funnel.mark_submit_failed(alpha_id=aid, reason=f"{st}:{reason}")
                _remove_from_app_list(aid)
                stats["submit_failed_deleted"] += 1
                print(f"[FLUSH_FAILED_REMOVED] Alpha {aid} permanently rejected ({reason})")

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
                        raw_size = eff_batch * 4 if (local_screener and pv_panel_ready) else eff_batch
                        batch = funnel.next_batch(datasets_fields, batch_size=raw_size)
                        if not batch:
                            exhausted = True
                            break
                        if local_screener and pv_panel_ready and any(is_price_volume_only(str(x.get("expression") or "")) for x in batch):
                            screened = local_screener.screen_batch(batch, top_k=eff_batch, allow_download=False)
                            if screened:
                                print(f"[LOCAL_PV_SCREEN] Filtered {len(batch)} -> {len(screened)} top PV/mixed candidates")
                                batch = screened
                            else:
                                batch = batch[:eff_batch]
                        elif len(batch) > eff_batch:
                            batch = batch[:eff_batch]
                        stats["mined"] += len(batch)
                        stats["batches"] += 1
                        stage_tag = batch[0].get("stage", "STAGE1")
                        ds_tag = batch[0].get("dataset_id", "")
                        print(f"[MULTI_SIM] Dispatching {stage_tag} batch #{stats['batches']} ({len(batch)} alphas, ds={ds_tag})")
                        in_flight.append(pool.submit(simulate_multi_alpha, client, batch))

                prev_sub = stats["submitted"]
                _flush_pending_if_due()
                _maybe_sync_to_drive(force=(stats["submitted"] != prev_sub))

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
            _maybe_sync_to_drive(force=True)

    print(f"[SUMMARY] {stats} | funnel={funnel.summary()}")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone Multi-Simulation WorldQuant BRAIN Alpha Miner")
    parser.add_argument("--duration-minutes", type=float, default=0.0, help="0 = run 24/7 continuously")
    _single_mode = os.getenv("BRAIN_DISABLE_MULTI_SIM", "false").lower() == "true"
    _default_workers = "3" if _single_mode else "2"
    _default_batch = "2" if _single_mode else "5"
    parser.add_argument("--workers", type=int, default=int(os.getenv("BRAIN_MULTI_SIM_WORKERS", _default_workers)))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BRAIN_MULTI_SIM_BATCH_SIZE", _default_batch)))
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--seed-sharpe", type=float, default=float(os.getenv("BRAIN_SEED_SHARPE_THRESHOLD", "1.15") or "1.15"))
    args = parser.parse_args()
    ds_list = [d.strip() for d in args.datasets.split(",") if d.strip()] or None
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
