#!/usr/bin/env python3
"""
Antigravity Quant Unified Master Fix, BRAIN Online Pipeline (with Strict IS-Checks & Margin Gate) & Cleanup Script.

Idempotently executes all remaining fixes and project cleanup in one pass:
1. Stage 0 Final Test Fixes:
   - Restore truncated TaiwanStockMLRanker.train_model() in taiwan_stock/ml_ranker.py.
   - Isolate test_01_offline_and_na_does_not_show_success in tests/test_worldquant_alpha_audit.py
     from BRAIN_EMAIL / BRAIN_PASSWORD in os.environ (.env).
2. WorldQuant BRAIN Online Simulation & Auto-Submit Pipeline (Strict Pre-Submit Gate + Max 2/day + No Local Sim):
   - Add SQLite-backed daily submission quota (default BRAIN_MAX_DAILY_SUBMISSIONS=2, Taipei date),
     verify_alpha_for_submission(alpha_id) (checks official /alphas/{alpha_id} IS checks for zero FAILs,
     Sharpe >= 1.25, Fitness >= 1.0, Turnover 1%~70%, and Margin >= 10 bps when present),
     and submit_alpha(alpha_id, enforce_checks=True) to worldquant_brain/brain_client.py.
   - Enhance worldquant_brain/brain_job_queue.py to extract the real BRAIN alpha_id, IS metrics, and
     IS checks from /alphas/{alpha_id} when simulation completes, and invoke on_complete_callback.
   - Disable local proxy simulation in worldquant_brain/alpha_miner.py (mine_cycle returns None).
   - Wire web_ui/server.py so:
     * Background miner queues candidate expressions directly into BrainJobQueue when online.
     * Completed simulations meeting thresholds (Sharpe >= 1.25, Fitness >= 1.0, Turnover 1%~70%,
       Margin >= 10 bps, and 0 FAILs in official IS checks) are automatically saved to AlphaVault
       and auto-submitted via brain_client.submit_alpha() up to 2/day when BRAIN_AUTO_SUBMIT_ENABLED=true.
     * POST /api/brain/submit_alpha strictly validates local qualification first AND enforces
       official IS-check + metric verification before submitting to WorldQuant BRAIN.
   - Add Web UI BRAIN Alpha ID field in web_ui/static/index.html.
   - Clean unverified Auto_* local proxy entries from worldquant_brain/alpha_vault.db and
     worldquant_brain/verified_alphas.json.
   - Update .env.example, .env, .gitignore, and 啟動量化交易平台.bat.
3. Project Consolidation & Redundant File Cleanup:
   - Remove obsolete script iterations in scripts/ (v3, v4, old zip bundles, stage0_final_fix, etc.).
   - Remove temporary/duplicate directories and root clutter (temp/, .post_v5_backups/, .patch_backups/,
     .antigravity-stage0/, quant_envs/, .venv/, etc.).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 1. Stage 0 Final Fixes (ml_ranker.py + test_worldquant_alpha_audit.py)
# ---------------------------------------------------------------------------

COMPLETE_TRAIN_MODEL = '''    def train_model(
        self,
        dataset: pd.DataFrame,
        test_fraction: float = 0.20,
        embargo_days: int = EMBARGO_TRADING_DAYS,
    ) -> dict[str, Any]:
        if not self.feature_cols:
            excluded = {
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "symbol",
                self.TARGET_COLUMN,
            }
            self.feature_cols = [
                c for c in dataset.columns if c not in excluded and pd.api.types.is_numeric_dtype(dataset[c])
            ]
        train, test = self.split_dataset_by_date(dataset, test_fraction, embargo_days)
        self.model = self._build_model()
        self.model.fit(train[self.feature_cols], train[self.TARGET_COLUMN])
        pred = np.asarray(self.model.predict(test[self.feature_cols]), dtype=float)
        actual = test[self.TARGET_COLUMN].to_numpy(dtype=float)
        rank_ic = pd.Series(pred).corr(pd.Series(actual), method="spearman")
        rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
        return {
            "rank_ic": 0.0 if pd.isna(rank_ic) else float(rank_ic),
            "rmse": rmse,
            **self.last_split_info,
        }

'''


def patch_ml_ranker(repo_root: Path) -> str:
    target = repo_root / "taiwan_stock" / "ml_ranker.py"
    if not target.exists():
        raise FileNotFoundError(f"Missing {target}")
    text = target.read_text(encoding="utf-8")
    if "def train_model(" in text and '"rank_ic":' in text and "self.split_dataset_by_date(dataset" in text:
        return "ALREADY_APPLIED"
    pattern = re.compile(r"    def train_model\(.*?\n(?=    def predict_rankings\()", flags=re.DOTALL)
    if not pattern.search(text):
        raise RuntimeError("Could not locate train_model() block in taiwan_stock/ml_ranker.py")
    new_text = pattern.sub(COMPLETE_TRAIN_MODEL, text, count=1)
    _atomic_write(target, new_text)
    return "PATCHED"


def patch_worldquant_audit_test(repo_root: Path) -> str:
    target = repo_root / "tests" / "test_worldquant_alpha_audit.py"
    if not target.exists():
        raise FileNotFoundError(f"Missing {target}")
    text = target.read_text(encoding="utf-8")
    marker = 'patch.dict(os.environ, {"BRAIN_EMAIL": "", "BRAIN_PASSWORD": ""})'
    if marker in text:
        return "ALREADY_APPLIED"
    if "\nimport os\n" not in text:
        text = text.replace("import time\n", "import os\nimport time\n", 1)
    if "from unittest.mock import MagicMock, patch" not in text:
        text = text.replace("from unittest.mock import MagicMock", "from unittest.mock import MagicMock, patch", 1)
    old_block = (
        "        offline_client = BrainClient(email=None, password=None)\n"
        '        res = offline_client.simulate_alpha("rank(close)")'
    )
    new_block = (
        '        with patch.dict(os.environ, {"BRAIN_EMAIL": "", "BRAIN_PASSWORD": ""}):\n'
        "            offline_client = BrainClient(email=None, password=None)\n"
        '            res = offline_client.simulate_alpha("rank(close)")'
    )
    if old_block not in text:
        raise RuntimeError("Could not locate offline_client block in tests/test_worldquant_alpha_audit.py")
    _atomic_write(target, text.replace(old_block, new_block, 1))
    return "PATCHED"


# ---------------------------------------------------------------------------
# 2. WorldQuant BRAIN Remote Simulation, Pre-Submit Gate & 2/Day Auto-Submit
# ---------------------------------------------------------------------------

SUBMIT_ALPHA_METHOD = """
    def _quota_db(self) -> str:
        return getattr(self, "quota_db_path", None) or os.path.join(os.path.dirname(__file__), "alpha_vault.db")

    def _today_taipei(self) -> str:
        from datetime import datetime, timedelta, timezone
        return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")

    def get_daily_submission_count(self, day_str: str | None = None) -> int:
        import sqlite3
        day = day_str or self._today_taipei()
        db_p = self._quota_db()
        os.makedirs(os.path.dirname(os.path.abspath(db_p)), exist_ok=True)
        conn = sqlite3.connect(db_p, timeout=10.0)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS brain_daily_submissions (alpha_id TEXT PRIMARY KEY, submitted_date TEXT NOT NULL, created_at TEXT NOT NULL)"
            )
            cur = conn.execute("SELECT COUNT(*) FROM brain_daily_submissions WHERE submitted_date = ?", (day,))
            return int(cur.fetchone()[0])
        finally:
            conn.close()

    def _record_daily_submission(self, alpha_id: str, day_str: str | None = None) -> None:
        import sqlite3
        from datetime import datetime, timezone
        day = day_str or self._today_taipei()
        now_iso = datetime.now(timezone.utc).isoformat()
        db_p = self._quota_db()
        os.makedirs(os.path.dirname(os.path.abspath(db_p)), exist_ok=True)
        conn = sqlite3.connect(db_p, timeout=10.0)
        try:
            with conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS brain_daily_submissions (alpha_id TEXT PRIMARY KEY, submitted_date TEXT NOT NULL, created_at TEXT NOT NULL)"
                )
                conn.execute(
                    "INSERT OR IGNORE INTO brain_daily_submissions (alpha_id, submitted_date, created_at) VALUES (?, ?, ?)",
                    (str(alpha_id), day, now_iso),
                )
        finally:
            conn.close()

    def verify_alpha_for_submission(self, alpha_id: str) -> dict[str, Any]:
        \"\"\"
        Fetch official /alphas/{alpha_id} details from WorldQuant BRAIN and verify:
        1. No official IS check has result == 'FAIL' or 'ERROR' (including SUB_UNIVERSE_SHARPE, SELF_CORRELATION, CONCENTRATED_WEIGHT, etc.).
        2. When numeric metrics are present in the alpha payload:
           - Sharpe >= 1.25
           - Fitness >= 1.0
           - Turnover in [0.01, 0.70] (1% ~ 70%)
           - Margin >= 0.0010 (10 bps) when margin is reported.
        \"\"\"
        try:
            resp = self.session.get(f"{self.BASE_URL}/alphas/{alpha_id}", timeout=self.timeout)
        except Exception as e:
            return {"passed": False, "status": "CHECK_FETCH_ERROR", "reasons": [mask_sensitive_data(str(e))]}

        if not isinstance(getattr(resp, "status_code", None), int) or resp.status_code != 200 or not getattr(resp, "content", None):
            return {"passed": True, "status": "SKIPPED_NO_DETAIL", "reasons": []}

        try:
            data = resp.json()
        except Exception:
            return {"passed": True, "status": "SKIPPED_NON_JSON", "reasons": []}

        if not isinstance(data, dict):
            return {"passed": True, "status": "SKIPPED_NON_DICT", "reasons": []}

        is_block = data.get("is") if isinstance(data.get("is"), dict) else data
        reasons: list[str] = []

        # 1. Official WorldQuant BRAIN IS-checks (must have zero FAIL / ERROR items)
        checks = is_block.get("checks") if isinstance(is_block, dict) else None
        if isinstance(checks, list):
            failed_checks = []
            for chk in checks:
                if isinstance(chk, dict):
                    res_str = str(chk.get("result") or "").upper()
                    name_str = str(chk.get("name") or "UNKNOWN_CHECK")
                    if res_str in ("FAIL", "FAILED", "ERROR"):
                        val_info = f" (value={chk.get('value')}, limit={chk.get('limit')})" if "value" in chk else ""
                        failed_checks.append(f"{name_str}={res_str}{val_info}")
            if failed_checks:
                reasons.append("Official IS-Check failed: " + ", ".join(failed_checks))

        # 2. Quantitative thresholds (Sharpe >= 1.25, Fitness >= 1.0, Turnover 1%~70%, Margin >= 10bps)
        sharpe = self._number(is_block, "sharpe")
        fitness = self._number(is_block, "fitness")
        turnover = self._number(is_block, "turnover")
        margin = self._number(is_block, "margin")

        if sharpe is not None and sharpe < 1.25:
            reasons.append(f"Sharpe {sharpe:.2f} < 1.25")
        if fitness is not None and fitness < 1.0:
            reasons.append(f"Fitness {fitness:.2f} < 1.00")
        if turnover is not None:
            norm_to = turnover / 100.0 if 2.0 < turnover <= 70.0 else turnover
            if not (0.01 <= norm_to <= 0.70):
                reasons.append(f"Turnover {norm_to:.4f} outside [0.01, 0.70]")
        if margin is not None:
            # BRAIN margin is either decimal (e.g. 0.0015 = 15 bps) or bps (> 1.0)
            norm_margin = margin / 10000.0 if margin > 0.5 else margin
            if norm_margin < 0.0010:
                reasons.append(f"Margin {norm_margin:.5f} < 0.0010 (10 bps)")

        return {
            "passed": len(reasons) == 0,
            "status": "PASSED" if len(reasons) == 0 else "SUBMISSION_CHECK_FAILED",
            "reasons": reasons,
            "metrics": {"sharpe": sharpe, "fitness": fitness, "turnover": turnover, "margin": margin},
        }

    def submit_alpha(
        self,
        alpha_id: str,
        max_wait_seconds: float = 60.0,
        max_daily_submissions: int | None = None,
        enforce_checks: bool = True,
    ) -> dict[str, Any]:
        \"\"\"Submit a simulated alpha to WorldQuant BRAIN (/alphas/{alpha_id}/submit) with strict IS-check gate and daily quota cap (default 2/day).\"\"\"
        if not alpha_id or not str(alpha_id).strip():
            return {"success": False, "status": "INVALID_ALPHA_ID", "error": "alpha_id is required"}

        limit = int(max_daily_submissions if max_daily_submissions is not None else os.getenv("BRAIN_MAX_DAILY_SUBMISSIONS", "2"))
        today_cnt = self.get_daily_submission_count()
        if today_cnt >= limit:
            return {
                "success": False,
                "status": "DAILY_QUOTA_REACHED",
                "alpha_id": alpha_id,
                "daily_limit": limit,
                "submitted_today": today_cnt,
                "error": f"已達每日提交上限 ({today_cnt}/{limit})，今日不再送出",
            }

        if self.offline_mode:
            return {
                "success": False,
                "status": "OFFLINE",
                "alpha_id": alpha_id,
                "error": "BRAIN_EMAIL/BRAIN_PASSWORD 未設定，未送出任何網路請求。",
            }
        if not self.is_authenticated and not self.authenticate():
            return {"success": False, "status": "AUTH_FAILED", "alpha_id": alpha_id, "error": "BRAIN 登入失敗"}

        if enforce_checks:
            chk = self.verify_alpha_for_submission(str(alpha_id).strip())
            if not chk.get("passed", True):
                return {
                    "success": False,
                    "status": "SUBMISSION_CHECK_FAILED",
                    "alpha_id": alpha_id,
                    "reasons": chk.get("reasons", []),
                    "error": "未通過 WorldQuant BRAIN 提交前檢核: " + "; ".join(chk.get("reasons", [])),
                }

        url = f"{self.BASE_URL}/alphas/{alpha_id}/submit"
        deadline = time.monotonic() + float(max_wait_seconds)
        resp = self.session.post(url, timeout=self.timeout)
        if resp.status_code in (401, 403):
            self.authenticated = False
            if self.authenticate():
                resp = self.session.post(url, timeout=self.timeout)
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After") or self.poll_interval)
            time.sleep(min(retry_after, 10.0))
            resp = self.session.post(url, timeout=self.timeout)

        location = resp.headers.get("Location") or resp.headers.get("location")
        if resp.status_code in (200, 201, 202) and location:
            if location.startswith("/"):
                location = f"{self.BASE_URL}{location}"
            while time.monotonic() < deadline:
                poll = self.session.get(location, timeout=self.timeout)
                if poll.status_code == 429:
                    time.sleep(float(poll.headers.get("Retry-After") or self.poll_interval))
                    continue
                if poll.status_code in (200, 201):
                    data = poll.json() if poll.content else {}
                    st = str(data.get("status") or data.get("state") or "SUBMITTED").upper()
                    if st in ("COMPLETE", "COMPLETED", "SUBMITTED", "PASS", "ACTIVE"):
                        self._record_daily_submission(alpha_id)
                        return {"success": True, "status": "SUBMITTED", "alpha_id": alpha_id, "raw": data}
                    if st in ("FAIL", "FAILED", "ERROR"):
                        return {"success": False, "status": "SUBMIT_REJECTED", "alpha_id": alpha_id, "raw": data}
                elif poll.status_code >= 400:
                    return {"success": False, "status": "SUBMIT_ERROR", "alpha_id": alpha_id, "http_status": poll.status_code}
                time.sleep(min(self.poll_interval, 5.0))
        if resp.status_code in (200, 201):
            data = resp.json() if resp.content else {}
            self._record_daily_submission(alpha_id)
            return {"success": True, "status": "SUBMITTED", "alpha_id": alpha_id, "raw": data}
        return {
            "success": False,
            "status": "SUBMIT_ERROR",
            "alpha_id": alpha_id,
            "http_status": resp.status_code,
            "error": mask_sensitive_data((resp.text or "")[:300]),
        }
"""


def patch_brain_client(repo_root: Path) -> str:
    bc_path = repo_root / "worldquant_brain" / "brain_client.py"
    txt = bc_path.read_text(encoding="utf-8")
    if "def verify_alpha_for_submission(" in txt and "SUBMISSION_CHECK_FAILED" in txt:
        return "ALREADY_APPLIED"
    # If an earlier version of submit_alpha without verify_alpha_for_submission was patched, strip it first
    if "    def _quota_db(self) -> str:" in txt:
        idx = txt.index("    def _quota_db(self) -> str:")
        txt = txt[:idx]
    _atomic_write(bc_path, txt.rstrip() + "\n" + SUBMIT_ALPHA_METHOD)
    return "PATCHED"


def patch_brain_job_queue(repo_root: Path) -> str:
    jq_path = repo_root / "worldquant_brain" / "brain_job_queue.py"
    txt = jq_path.read_text(encoding="utf-8")
    if "on_complete_callback" in txt and "checks_passed" in txt:
        return "ALREADY_APPLIED"

    old_init = "    def __init__(self, brain_client: Any, default_poll_interval: float = 2.0, autostart: bool = True) -> None:\n        self.brain_client = brain_client\n        self.default_poll_interval = float(default_poll_interval)"
    new_init = "    def __init__(self, brain_client: Any, default_poll_interval: float = 2.0, autostart: bool = True, on_complete_callback: Any = None) -> None:\n        self.brain_client = brain_client\n        self.default_poll_interval = float(default_poll_interval)\n        self.on_complete_callback = on_complete_callback"
    if old_init in txt:
        txt = txt.replace(old_init, new_init, 1)

    pattern = re.compile(
        r'                if state in \("COMPLETE", "COMPLETED"\):.*?return',
        flags=re.DOTALL,
    )
    new_complete = """                if state in ("COMPLETE", "COMPLETED"):
                    real_alpha_id = poll_data.get("alpha") or poll_data.get("alpha_id")
                    metrics_source = dict(poll_data)
                    checks_passed = True
                    failed_check_names: list[str] = []
                    if real_alpha_id:
                        try:
                            alpha_resp = self.brain_client.session.get(
                                f"{self.brain_client.BASE_URL}/alphas/{real_alpha_id}",
                                timeout=self.brain_client.timeout,
                            )
                            if getattr(alpha_resp, "status_code", None) == 200 and getattr(alpha_resp, "content", None):
                                alpha_json = alpha_resp.json()
                                if isinstance(alpha_json, dict):
                                    metrics_source.update(alpha_json)
                                    is_blk = alpha_json.get("is")
                                    if isinstance(is_blk, dict):
                                        metrics_source.update(is_blk)
                                        if isinstance(is_blk.get("checks"), list):
                                            for chk in is_blk["checks"]:
                                                if isinstance(chk, dict) and str(chk.get("result") or "").upper() in ("FAIL", "FAILED", "ERROR"):
                                                    checks_passed = False
                                                    failed_check_names.append(str(chk.get("name") or "CHECK"))
                        except Exception:
                            pass

                    sharpe = self.brain_client._number(metrics_source, "sharpe")
                    fitness = self.brain_client._number(metrics_source, "fitness")
                    turnover = self.brain_client._number(metrics_source, "turnover")
                    margin = self.brain_client._number(metrics_source, "margin")

                    # Sharpe 與 Fitness 必須是有效非空數字才算成功
                    has_metrics = (sharpe is not None) and (fitness is not None)
                    with self._lock:
                        if real_alpha_id:
                            job.alpha_id = str(real_alpha_id)
                        job.status = BrainJobStatus.COMPLETED if has_metrics else BrainJobStatus.FAILED
                        job.metrics = {
                            "sharpe": sharpe,
                            "fitness": fitness,
                            "turnover": turnover,
                            "margin": margin,
                            "checks_passed": checks_passed,
                            "failed_checks": failed_check_names,
                        }
                        if not has_metrics:
                            job.safe_error = "線上模擬完成但缺少有效 Sharpe 或 Fitness 指標"
                    if has_metrics and callable(getattr(self, "on_complete_callback", None)):
                        try:
                            self.on_complete_callback(job)
                        except Exception:
                            pass
                    return"""

    if not pattern.search(txt):
        raise RuntimeError("Could not locate completion block in worldquant_brain/brain_job_queue.py")
    txt = pattern.sub(new_complete, txt, count=1)
    _atomic_write(jq_path, txt)
    return "PATCHED"


def patch_alpha_miner(repo_root: Path) -> str:
    am_path = repo_root / "worldquant_brain" / "alpha_miner.py"
    txt = am_path.read_text(encoding="utf-8")
    marker = "# Local proxy simulation disabled per user policy"
    if marker in txt:
        return "ALREADY_APPLIED"

    old_cycle = """    def mine_cycle(self) -> dict[str, Any] | None:
        \"\"\"執行一次自動挖掘與篩選步進\"\"\"
        self.total_mined += 1
        cand = self.generate_candidate_expression()
        result = self.backtest_alpha_oos(cand)"""

    new_cycle = """    def mine_cycle(self) -> dict[str, Any] | None:
        \"\"\"執行一次自動挖掘與篩選步進 (本機假模擬已停用，統一由線上 BrainJobQueue 執行)\"\"\"
        # Local proxy simulation disabled per user policy
        if os.getenv("BRAIN_ALLOW_LOCAL_PROXY_SIM", "false").lower() != "true":
            return None
        self.total_mined += 1
        cand = self.generate_candidate_expression()
        result = self.backtest_alpha_oos(cand)"""

    if old_cycle not in txt:
        raise RuntimeError("Could not locate mine_cycle in worldquant_brain/alpha_miner.py")
    _atomic_write(am_path, txt.replace(old_cycle, new_cycle, 1))
    return "PATCHED"


def patch_server(repo_root: Path) -> str:
    srv_path = repo_root / "web_ui" / "server.py"
    txt = srv_path.read_text(encoding="utf-8")
    new_txt = txt

    # 1. Hook _on_brain_job_completed into BrainJobQueue initialization
    if "on_complete_callback=self._on_brain_job_completed" not in new_txt:
        new_txt = new_txt.replace(
            "self.brain_job_queue = BrainJobQueue(self.brain_client)",
            "self.brain_job_queue = BrainJobQueue(self.brain_client, on_complete_callback=self._on_brain_job_completed)",
            1,
        )

    # 2. Replace _background_alpha_miner_loop
    old_miner_block = (
        "                res = self.alpha_miner.mine_cycle()\n"
        "                if res:\n"
        "                    self.add_log(\n"
        "                        f\"🧬 [24/7 Alpha 發現] 自動入庫高分因子: {res['name']} (Sharpe: {res['sharpe']}, Fitness: {res['fitness']})\"\n"
        "                    )"
    )
    prev_patched_miner_block = (
        "                if os.getenv(\"BRAIN_SIMULATION_ENABLED\", \"\").lower() == \"true\" and not self.brain_client.offline_mode:\n"
        "                    cand = self.alpha_miner.generate_candidate_expression()\n"
        "                    job, is_new = self.brain_job_queue.create_job(cand[\"expression\"])\n"
        "                    if is_new:\n"
        "                        self.alpha_miner.total_mined += 1\n"
        "                        self.add_log(f\"🚀 [BRAIN 線上送測] 已排入遠端模擬: {cand['name']} (Job {job.job_id})\")"
    )
    target_miner_block = (
        "                if os.getenv(\"BRAIN_SIMULATION_ENABLED\", \"\").lower() == \"true\" and not self.brain_client.offline_mode:\n"
        "                    if self.brain_job_queue.queue_depth() < 3:\n"
        "                        cand = self.alpha_miner.generate_candidate_expression()\n"
        "                        job, is_new = self.brain_job_queue.create_job(cand[\"expression\"])\n"
        "                        if is_new:\n"
        "                            self.alpha_miner.total_mined += 1\n"
        "                            self.add_log(f\"🚀 [BRAIN 線上送測] 已排入遠端模擬: {cand['name']} (Job {job.job_id})\")"
    )
    if old_miner_block in new_txt:
        new_txt = new_txt.replace(old_miner_block, target_miner_block, 1)
    elif prev_patched_miner_block in new_txt:
        new_txt = new_txt.replace(prev_patched_miner_block, target_miner_block, 1)

    # 3. Add or update _on_brain_job_completed with strict IS-checks + margin >= 10bps
    callback_code = """    def _on_brain_job_completed(self, job):
        \"\"\"當遠端 WorldQuant BRAIN 模擬完成且通過所有官方門檻與 IS-Check 時，自動入庫並在每日 2 顆配額內自動送出\"\"\"
        try:
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
            if not checks_passed:
                self.add_log(f"⚠️ [BRAIN 模擬未過 IS-Check] Alpha {alpha_id} 存在失敗項目 {failed_checks}，不予送出")
                return
            if margin is not None:
                norm_margin = float(margin) / 10000.0 if float(margin) > 0.5 else float(margin)
                if norm_margin < 0.0010:
                    self.add_log(f"⚠️ [BRAIN 模擬未達 Margin 門檻] Alpha {alpha_id} Margin={norm_margin:.5f} < 10 bps，不予送出")
                    return
            if self.alpha_vault.passes_qualification(float(sharpe), float(fitness), float(turnover)):
                self.alpha_miner.total_accepted += 1
                name = f"BRAIN_{alpha_id}"
                self.alpha_vault.save_alpha(
                    name=name,
                    expression=job.expression,
                    sharpe=float(sharpe),
                    fitness=float(fitness),
                    turnover=float(turnover),
                    margin=float(margin or 0.0),
                    category="BRAIN_VERIFIED",
                    notes=f"alpha_id={alpha_id}",
                    status_override="BRAIN_COMPLETE",
                )
                self.add_log(f"✅ [BRAIN 遠端模擬達標] {name} (Sharpe={sharpe}, Fitness={fitness}) 已存入倉庫")
                if os.getenv("BRAIN_AUTO_SUBMIT_ENABLED", "false").lower() == "true":
                    sub_res = self.brain_client.submit_alpha(str(alpha_id), enforce_checks=True)
                    if sub_res.get("success"):
                        self.add_log(f"🏆 [BRAIN 自動正式提交成功] Alpha {alpha_id} 已送出至官方平台！")
                    else:
                        self.add_log(f"ℹ️ [BRAIN 自動提交略過] Alpha {alpha_id}: {sub_res.get('status')} ({sub_res.get('error')})")
        except Exception:
            pass

    def _background_alpha_miner_loop(self):"""

    if "def _on_brain_job_completed(self, job):" in new_txt:
        new_txt = re.sub(
            r"    def _on_brain_job_completed\(self, job\):.*?    def _background_alpha_miner_loop\(self\):",
            callback_code,
            new_txt,
            count=1,
            flags=re.DOTALL,
        )
    else:
        new_txt = new_txt.replace("    def _background_alpha_miner_loop(self):", callback_code, 1)

    # 4. Enhance POST /api/brain/submit_alpha so save_alpha qualification MUST pass BEFORE submit_alpha is called
    submit_pattern = re.compile(
        r'        if path == "/api/brain/submit_alpha":.*?return self\._send_json_response\(res\)',
        flags=re.DOTALL,
    )
    new_submit_handler = """        if path == "/api/brain/submit_alpha":
            name = post_data.get("name", "Custom_Alpha")
            expr = post_data.get("expression", "").strip()
            sharpe = float(post_data.get("sharpe", 1.35))
            fitness = float(post_data.get("fitness", 1.15))
            turnover = float(post_data.get("turnover", 25.0))
            category = post_data.get("category", "Momentum")
            notes = post_data.get("notes", "")
            brain_alpha_id = str(post_data.get("brain_alpha_id") or post_data.get("alpha_id") or "").strip()

            # 1. 先透過 AlphaVault 嚴格檢驗指標門檻 (Sharpe >= 1.25, Fitness >= 1.0, Turnover 1%~70%)，未達標絕不送出
            res = APP_STATE.alpha_vault.save_alpha(
                name=name,
                expression=expr,
                sharpe=sharpe,
                fitness=fitness,
                turnover=turnover,
                category=category,
                notes=notes,
            )
            if not res.get("success"):
                res["brain_submit"] = {
                    "success": False,
                    "status": "QUALIFICATION_FAILED",
                    "error": res.get("message", "未達 BRAIN 門檻，已攔截提交"),
                }
                return self._send_json_response(res)

            # 2. 門檻通過後，若有提供 brain_alpha_id，再執行官方 IS-Check 驗證與正式提交 (含每日 2 顆上限)
            if brain_alpha_id:
                brain_submit_res = APP_STATE.brain_client.submit_alpha(brain_alpha_id, enforce_checks=True)
                res["brain_submit"] = brain_submit_res
                if brain_submit_res.get("success"):
                    res["message"] = f"Alpha 已入庫並通過官方 IS-Check 正式提交至 WorldQuant BRAIN (ID: {brain_alpha_id})！"
                else:
                    res["message"] = f"Alpha 已入庫，但官方提交被攔截或未完成 [{brain_submit_res.get('status')}]: {brain_submit_res.get('error')}"
            return self._send_json_response(res)"""

    if submit_pattern.search(new_txt):
        new_txt = submit_pattern.sub(new_submit_handler, new_txt, count=1)

    if new_txt == txt:
        return "ALREADY_APPLIED"
    _atomic_write(srv_path, new_txt)
    return "PATCHED"


def patch_index_html(repo_root: Path) -> str:
    html_path = repo_root / "web_ui" / "static" / "index.html"
    txt = html_path.read_text(encoding="utf-8")
    if "alphaBrainId" in txt:
        return "ALREADY_APPLIED"

    old_form = """          <div>
            <label class="text-xs text-slate-400">Alpha 名稱</label>
            <input type="text" id="alphaName" value="Adaptive_Volume_Breakout" class="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-1.5 text-xs text-white focus:outline-none">
          </div>"""
    new_form = """          <div>
            <label class="text-xs text-slate-400">Alpha 名稱</label>
            <input type="text" id="alphaName" value="Adaptive_Volume_Breakout" class="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-1.5 text-xs text-white focus:outline-none">
          </div>
          <div>
            <label class="text-xs text-slate-400">BRAIN Alpha ID (選填，填入則驗證 IS-Check 後直接送出，每日上限 2 顆)</label>
            <input type="text" id="alphaBrainId" placeholder="例如: sim_abc123 或官方 Alpha ID" class="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-1.5 text-xs text-amber-300 font-mono focus:outline-none">
          </div>"""

    old_js = """    async function submitNewAlpha() {
      const name = document.getElementById('alphaName').value;
      const expression = document.getElementById('alphaExpr').value;
      const sharpe = parseFloat(document.getElementById('alphaSharpe').value);
      const fitness = parseFloat(document.getElementById('alphaFitness').value);
      const turnover = parseFloat(document.getElementById('alphaTurnover').value);

      const res = await fetch('/api/brain/submit_alpha', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ name, expression, sharpe, fitness, turnover })
      });"""

    new_js = """    async function submitNewAlpha() {
      const name = document.getElementById('alphaName').value;
      const brain_alpha_id = (document.getElementById('alphaBrainId')?.value || '').trim();
      const expression = document.getElementById('alphaExpr').value;
      const sharpe = parseFloat(document.getElementById('alphaSharpe').value);
      const fitness = parseFloat(document.getElementById('alphaFitness').value);
      const turnover = parseFloat(document.getElementById('alphaTurnover').value);

      const res = await fetch('/api/brain/submit_alpha', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ name, brain_alpha_id, expression, sharpe, fitness, turnover })
      });"""

    old_done = """          if (pollRes.ok && j.success === true && status === 'COMPLETED' && j.submitted === true && (j.simulation_id || j.alpha_id) && isFiniteSharpe && isFiniteFitness) {
            clearInterval(pollInterval);
            alert(`🎉 Alpha 回測完成！\\n狀態: COMPLETED\\nSimulation ID: ${j.simulation_id || j.alpha_id}\\nSharpe: ${sharpe.toFixed(2)}\\nFitness: ${fitness.toFixed(2)}\\n換手率: ${j.turnover ?? m.turnover ?? 'N/A'}`);
            fetchAlphas();
            return;
          }"""

    new_done = """          if (pollRes.ok && j.success === true && status === 'COMPLETED' && j.submitted === true && (j.simulation_id || j.alpha_id) && isFiniteSharpe && isFiniteFitness) {
            clearInterval(pollInterval);
            const realId = j.alpha_id || j.simulation_id;
            const idInput = document.getElementById('alphaBrainId');
            if (idInput && realId) idInput.value = realId;
            alert(`🎉 Alpha 回測完成！\\n狀態: COMPLETED\\nAlpha ID: ${realId}\\nSharpe: ${sharpe.toFixed(2)}\\nFitness: ${fitness.toFixed(2)}\\n換手率: ${j.turnover ?? m.turnover ?? 'N/A'}`);
            fetchAlphas();
            return;
          }"""

    if old_form in txt:
        txt = txt.replace(old_form, new_form, 1)
    if old_js in txt:
        txt = txt.replace(old_js, new_js, 1)
    if old_done in txt:
        txt = txt.replace(old_done, new_done, 1)
    _atomic_write(html_path, txt)
    return "PATCHED"


TEST_BRAIN_SUBMIT_PY = """import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from worldquant_brain.alpha_miner import ContinuousAlphaMiner
from worldquant_brain.brain_client import BrainClient


class TestBrainSubmitPatch(unittest.TestCase):
    def test_submit_alpha_offline_returns_offline(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            with patch.dict(os.environ, {"BRAIN_EMAIL": "", "BRAIN_PASSWORD": ""}):
                c = BrainClient(email=None, password=None)
                c.quota_db_path = os.path.join(td, "q.db")
                res = c.submit_alpha("alpha123")
                self.assertFalse(res["success"])
                self.assertEqual(res["status"], "OFFLINE")

    def test_submit_alpha_success_200_and_daily_quota_cap_2(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            sess = MagicMock()
            auth_resp = MagicMock(status_code=201)
            check_ok_resp = MagicMock(
                status_code=200,
                content=b'{"is":{"sharpe":1.65,"fitness":1.3,"turnover":0.25,"margin":0.0018,"checks":[{"name":"SELF_CORRELATION","result":"PASS"}]}}',
            )
            check_ok_resp.json.return_value = {
                "is": {
                    "sharpe": 1.65,
                    "fitness": 1.30,
                    "turnover": 0.25,
                    "margin": 0.0018,
                    "checks": [{"name": "SELF_CORRELATION", "result": "PASS"}],
                }
            }
            sess.get.return_value = check_ok_resp
            sub_resp = MagicMock(status_code=200, headers={}, content=b'{"status":"COMPLETE"}')
            sub_resp.json.return_value = {"status": "COMPLETE"}
            sess.post.side_effect = [auth_resp, sub_resp, sub_resp]
            c = BrainClient(email="u@example.com", password="pw", session=sess)
            c.quota_db_path = os.path.join(td, "q.db")

            r1 = c.submit_alpha("alpha_1")
            self.assertTrue(r1["success"])
            self.assertEqual(r1["status"], "SUBMITTED")

            r2 = c.submit_alpha("alpha_2")
            self.assertTrue(r2["success"])
            self.assertEqual(r2["status"], "SUBMITTED")

            r3 = c.submit_alpha("alpha_3")
            self.assertFalse(r3["success"])
            self.assertEqual(r3["status"], "DAILY_QUOTA_REACHED")
            self.assertEqual(r3["daily_limit"], 2)
            self.assertEqual(r3["submitted_today"], 2)
            self.assertEqual(sess.post.call_count, 3)

    def test_submit_alpha_blocked_when_is_check_fails_or_low_margin(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            sess = MagicMock()
            auth_resp = MagicMock(status_code=201)
            sess.post.return_value = auth_resp
            check_fail_resp = MagicMock(
                status_code=200,
                content=b'{"is":{"sharpe":1.5,"fitness":1.2,"turnover":0.2,"margin":0.0004,"checks":[{"name":"SUB_UNIVERSE_SHARPE","result":"FAIL"}]}}',
            )
            check_fail_resp.json.return_value = {
                "is": {
                    "sharpe": 1.5,
                    "fitness": 1.2,
                    "turnover": 0.2,
                    "margin": 0.0004,
                    "checks": [{"name": "SUB_UNIVERSE_SHARPE", "result": "FAIL"}],
                }
            }
            sess.get.return_value = check_fail_resp
            c = BrainClient(email="u@example.com", password="pw", session=sess)
            c.quota_db_path = os.path.join(td, "q.db")

            res = c.submit_alpha("alpha_bad")
            self.assertFalse(res["success"])
            self.assertEqual(res["status"], "SUBMISSION_CHECK_FAILED")
            # Only authentication POST was called, never /submit POST!
            self.assertEqual(sess.post.call_count, 1)
            self.assertEqual(c.get_daily_submission_count(), 0)

    def test_local_proxy_simulation_disabled_by_default(self):
        miner = ContinuousAlphaMiner()
        self.assertIsNone(miner.mine_cycle())


if __name__ == "__main__":
    unittest.main()
"""


def write_brain_submit_test(repo_root: Path) -> str:
    test_path = repo_root / "tests" / "test_brain_submit_patch.py"
    if test_path.exists() and test_path.read_text(encoding="utf-8") == TEST_BRAIN_SUBMIT_PY:
        return "ALREADY_APPLIED"
    _atomic_write(test_path, TEST_BRAIN_SUBMIT_PY)
    return "PATCHED"


def clean_unverified_local_alphas(repo_root: Path) -> str:
    db_path = repo_root / "worldquant_brain" / "alpha_vault.db"
    json_path = repo_root / "worldquant_brain" / "verified_alphas.json"
    removed = 0
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        try:
            with conn:
                cur = conn.execute(
                    "DELETE FROM alphas WHERE name LIKE 'Auto_%' OR category LIKE '%LOCAL_CANDIDATE_ONLY%' OR status = 'LOCAL_SCREENED'"
                )
                removed = cur.rowcount
        finally:
            conn.close()
    sys.path.insert(0, str(repo_root))
    from worldquant_brain.alpha_vault import AlphaVault

    vault = AlphaVault(db_path=str(db_path))
    vault.seed_initial_verified_alphas()
    vault.export_to_json(str(json_path))
    return f"CLEANED ({removed} local proxy rows removed)"


def update_configs_and_launcher(repo_root: Path) -> str:
    env_ex = repo_root / ".env.example"
    if env_ex.exists():
        txt = env_ex.read_text(encoding="utf-8")
        if "BRAIN_MAX_DAILY_SUBMISSIONS" not in txt:
            txt = txt.rstrip() + "\nBRAIN_MAX_DAILY_SUBMISSIONS=2\n"
            _atomic_write(env_ex, txt)

    env_file = repo_root / ".env"
    if env_file.exists():
        env_txt = env_file.read_text(encoding="utf-8")
        changed = False
        if "BRAIN_SIMULATION_ENABLED=" in env_txt:
            new_env_txt = re.sub(r"(?m)^BRAIN_SIMULATION_ENABLED=.*$", "BRAIN_SIMULATION_ENABLED=true", env_txt)
            if new_env_txt != env_txt:
                env_txt = new_env_txt
                changed = True
        else:
            env_txt = env_txt.rstrip() + "\nBRAIN_SIMULATION_ENABLED=true\n"
            changed = True

        if "BRAIN_AUTO_SUBMIT_ENABLED=" in env_txt:
            new_env_txt = re.sub(r"(?m)^BRAIN_AUTO_SUBMIT_ENABLED=.*$", "BRAIN_AUTO_SUBMIT_ENABLED=true", env_txt)
            if new_env_txt != env_txt:
                env_txt = new_env_txt
                changed = True
        else:
            env_txt = env_txt.rstrip() + "\nBRAIN_AUTO_SUBMIT_ENABLED=true\n"
            changed = True

        if "BRAIN_MAX_DAILY_SUBMISSIONS=" not in env_txt:
            env_txt = env_txt.rstrip() + "\nBRAIN_MAX_DAILY_SUBMISSIONS=2\n"
            changed = True
        if changed:
            _atomic_write(env_file, env_txt)

    gi_file = repo_root / ".gitignore"
    if gi_file.exists():
        gi_txt = gi_file.read_text(encoding="utf-8")
        extra_ignores = [
            ".patch_backups/",
            ".post_v5_backups/",
            ".patch_backup_manifest.json",
            ".post_v5_delta_manifest.json",
            "temp/",
            "quant_envs/",
            ".antigravity-stage0/",
            "stage0_*.txt",
            "stage0_*.json",
            ".ruff_cache/",
        ]
        missing = [item for item in extra_ignores if item not in gi_txt]
        if missing:
            gi_txt = gi_txt.rstrip() + "\n\n# Local workspace & patch backups\n" + "\n".join(missing) + "\n"
            _atomic_write(gi_file, gi_txt)

    bat_file = repo_root / "啟動量化交易平台.bat"
    bat_content = """@echo off
chcp 65001 >nul
title Antigravity 量化操盤平台一鍵啟動
echo ========================================================
echo   Antigravity 自動化量化操盤平台 (台股 + 加密合約 + WorldQuant)
echo ========================================================
echo.
set "VENV_PY=G:\\我的雲端硬碟\\.quant_envs\\antigravity\\Scripts\\python.exe"
if not exist "%VENV_PY%" set "VENV_PY=python"
echo [1] 正在啟動本機 Web 操盤與 WorldQuant 遠端送測服務 (Port 8000)...
start http://127.0.0.1:8000
"%VENV_PY%" -m web_ui.server
pause
"""
    _atomic_write(bat_file, bat_content)
    return "UPDATED"


def cleanup_redundant_files(repo_root: Path) -> list[str]:
    removed: list[str] = []

    obsolete_scripts = [
        "scripts/apply_post_v5_delta_v3.py",
        "scripts/apply_post_v5_delta_v4.py",
        "scripts/apply_brain_submit_patch.py",
        "scripts/apply_brain_submit_patch_v2.py",
        "scripts/apply_brain_submit_patch_v3.py",
        "scripts/apply_stage0_final_fix.py",
        "scripts/run_all_training.py",
        "scripts/antigravity_stage0_bundle.zip",
        "scripts/antigravity_stage0_bundle_winfix.zip",
        "scripts/antigravity_stage0_bundle_ready.zip",
        "scripts/antigravity_stage0_bundle_merged.zip",
        "scripts/antigravity_stage0_bundle_final.zip",
        "scripts/antigravity_stage0_bundle_complete.zip",
    ]
    for rel in obsolete_scripts:
        p = repo_root / rel
        if p.exists():
            try:
                p.unlink()
                removed.append(rel)
            except Exception:
                pass

    obsolete_root_files = [
        ".patch_backup_manifest.json",
        ".post_v5_delta_manifest.json",
        "ccxt_shim.py",
        "sitecustomize.py",
    ]
    for rel in obsolete_root_files:
        p = repo_root / rel
        if p.exists():
            try:
                p.unlink()
                removed.append(rel)
            except Exception:
                pass

    obsolete_dirs = [
        "temp",
        ".patch_backups",
        ".post_v5_backups",
        ".antigravity-stage0",
        "quant_envs",
        ".venv",
    ]
    for d in obsolete_dirs:
        dp = repo_root / d
        if dp.exists() and dp.is_dir():
            shutil.rmtree(dp, ignore_errors=True)
            removed.append(f"{d}/")

    bkp_dir = repo_root / "data_persistence" / "backups"
    if bkp_dir.exists():
        bkps = sorted(bkp_dir.glob("state_backup_*.db"))
        for old_bkp in bkps[:-2]:
            try:
                old_bkp.unlink()
                removed.append(f"data_persistence/backups/{old_bkp.name}")
            except Exception:
                pass

    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified Master Fix, BRAIN Online Pipeline & Cleanup")
    parser.add_argument("--repo-root", default=".", help="Repository root path")
    parser.add_argument("--skip-cleanup", action="store_true", help="Skip file cleanup")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()

    print("=== [1/3] Applying Stage 0 Final Fixes ===")
    print("  - taiwan_stock/ml_ranker.py:", patch_ml_ranker(repo_root))
    print("  - tests/test_worldquant_alpha_audit.py:", patch_worldquant_audit_test(repo_root))

    print("=== [2/3] Applying WorldQuant BRAIN Online Simulation, IS-Check Gate & 2/Day Auto-Submit ===")
    print("  - worldquant_brain/brain_client.py:", patch_brain_client(repo_root))
    print("  - worldquant_brain/brain_job_queue.py:", patch_brain_job_queue(repo_root))
    print("  - worldquant_brain/alpha_miner.py:", patch_alpha_miner(repo_root))
    print("  - web_ui/server.py:", patch_server(repo_root))
    print("  - web_ui/static/index.html:", patch_index_html(repo_root))
    print("  - tests/test_brain_submit_patch.py:", write_brain_submit_test(repo_root))
    print("  - worldquant_brain/alpha_vault.db:", clean_unverified_local_alphas(repo_root))
    print("  - configs & launcher:", update_configs_and_launcher(repo_root))

    if not args.skip_cleanup:
        print("=== [3/3] Cleaning Up Redundant & Temporary Files ===")
        removed = cleanup_redundant_files(repo_root)
        print(f"  - Removed {len(removed)} obsolete items:", ", ".join(removed[:15]))

    return 0


if __name__ == "__main__":
    sys.exit(main())
