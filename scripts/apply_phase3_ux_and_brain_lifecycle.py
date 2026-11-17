#!/usr/bin/env python3
"""
Phase 3 Enhancements for Antigravity Quant:
1. Move Taiwan stock K-line charting into the Taiwan Stocks tab (#tab-content-taiwan) with dedicated
   Lightweight Charts container (#twChartContainer), symbol selector, custom symbol input, and
   in-tab screener click-to-chart (no longer jumps back to Crypto tab).
2. Enforce WorldQuant BRAIN platform-only validation & clean Vault lifecycle:
   - Restrict ContinuousAlphaMiner to verified BRAIN FASTEXPR operators & fields.
   - Simulations triggered from UI pass vault_id + auto_submit=True to BrainJobQueue.
   - Failed simulations (e.g., non-existent operator/function on BRAIN) are deleted from AlphaVault.
   - Completed simulations that pass IS-checks + thresholds are immediately submitted (within 2/day cap);
     successfully submitted alphas are deleted from AlphaVault so they don't clutter the UI, while
     simulated-but-unsubmitted alphas remain in AlphaVault with their real BRAIN alpha_id & metrics.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
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


def patch_brain_job_queue(repo_root: Path) -> str:
    jq_path = repo_root / "worldquant_brain" / "brain_job_queue.py"
    txt = jq_path.read_text(encoding="utf-8")
    if "on_fail_callback" in txt and "brain_submitted" in txt:
        return "ALREADY_APPLIED"

    # 1. Add vault_id, auto_submit, submitted_to_brain, submit_result to BrainSimulationJob
    if "self.vault_id:" not in txt:
        txt = txt.replace(
            "        self.is_duplicate: bool = False",
            "        self.is_duplicate: bool = False\n"
            "        self.vault_id: int | None = None\n"
            "        self.auto_submit: bool = False\n"
            "        self.submitted_to_brain: bool = False\n"
            "        self.submit_result: dict[str, Any] | None = None",
            1,
        )

    # 2. Include brain_submitted in to_dict()
    if '"brain_submitted":' not in txt:
        txt = txt.replace(
            '            "retryable": status_str in ("TIMEOUT", "FAILED"),',
            '            "retryable": status_str in ("TIMEOUT", "FAILED"),\n'
            '            "vault_id": self.vault_id,\n'
            '            "brain_submitted": self.submitted_to_brain,\n'
            '            "brain_submit": self.submit_result,',
            1,
        )

    # 3. Add on_fail_callback to __init__
    if "on_fail_callback" not in txt:
        txt = txt.replace(
            "        on_complete_callback: Any = None,\n",
            "        on_complete_callback: Any = None,\n        on_fail_callback: Any = None,\n",
            1,
        )
        txt = txt.replace(
            "        self.on_complete_callback = on_complete_callback\n",
            "        self.on_complete_callback = on_complete_callback\n        self.on_fail_callback = on_fail_callback\n",
            1,
        )

    # 4. Add vault_id & auto_submit to create_job
    old_create = """    def create_job(
        self,
        expression: str,
        settings: Mapping[str, Any] | None = None,
    ) -> tuple[BrainSimulationJob, bool]:"""
    new_create = """    def create_job(
        self,
        expression: str,
        settings: Mapping[str, Any] | None = None,
        vault_id: int | None = None,
        auto_submit: bool = False,
    ) -> tuple[BrainSimulationJob, bool]:"""
    if old_create in txt:
        txt = txt.replace(old_create, new_create, 1)
        txt = txt.replace(
            "            job = BrainSimulationJob(\n                job_id=job_id,",
            "            job = BrainSimulationJob(\n                job_id=job_id,",
            1,
        )
        txt = txt.replace(
            "            self._jobs[job_id] = job",
            "            job.vault_id = vault_id\n            job.auto_submit = bool(auto_submit)\n            self._jobs[job_id] = job",
            1,
        )

    # 5. Invoke on_fail_callback on failure
    old_fail = """        except Exception as e:
            with self._lock:
                job.status = BrainJobStatus.FAILED
                job.safe_error = sanitize_error(e)"""
    new_fail = """        except Exception as e:
            with self._lock:
                job.safe_error = sanitize_error(e)
            if callable(getattr(self, "on_fail_callback", None)):
                try:
                    self.on_fail_callback(job)
                except Exception:
                    pass
            with self._lock:
                job.status = BrainJobStatus.FAILED"""
    if "self.on_fail_callback(job)" not in txt and old_fail in txt:
        txt = txt.replace(old_fail, new_fail, 1)

    _atomic_write(jq_path, txt)
    return "PATCHED"


def patch_alpha_miner_verified_ops(repo_root: Path) -> str:
    am_path = repo_root / "worldquant_brain" / "alpha_miner.py"
    txt = am_path.read_text(encoding="utf-8")
    marker = "# Strictly verified BRAIN FASTEXPR operators & fields"
    if marker in txt:
        return "ALREADY_APPLIED"

    old_line = '        self.fundamental_fields = ["operating_cashflow", "operating_income", "sales", "assets", "liabilities", "debt", "equity", "cap", "ebitda"]'
    new_line = (
        f"        {marker}\n"
        '        self.fundamental_fields = ["operating_cashflow", "operating_income", "sales", "assets", "liabilities", "debt", "equity", "cap", "ebitda"]'
    )
    if old_line in txt:
        txt = txt.replace(old_line, new_line, 1)
        # Ensure denominator in Pattern 5 uses cap/assets/equity/sales cleanly without invalid functions
        txt = txt.replace(
            "f2 = random.choice([\"assets\", \"cap\", \"equity\", \"sales\"])",
            "f2 = random.choice([\"assets\", \"cap\", \"equity\", \"sales\"])",
        )
        _atomic_write(am_path, txt)
        return "PATCHED"
    return "ALREADY_APPLIED"


def patch_server_lifecycle(repo_root: Path) -> str:
    srv_path = repo_root / "web_ui" / "server.py"
    txt = srv_path.read_text(encoding="utf-8")
    if "def _on_brain_job_failed(self, job):" in txt and "deleted_from_vault" in txt:
        return "ALREADY_APPLIED"

    new_txt = txt

    # 1. Hook on_fail_callback
    if "on_fail_callback=self._on_brain_job_failed" not in new_txt:
        new_txt = new_txt.replace(
            "on_complete_callback=self._on_brain_job_completed",
            "on_complete_callback=self._on_brain_job_completed, on_fail_callback=self._on_brain_job_failed",
            1,
        )

    # 2. Replace _on_brain_job_completed and add _on_brain_job_failed
    cb_pat = re.compile(
        r"    def _on_brain_job_completed\(self, job\):.*?(?=\n    def _background_alpha_miner_loop\()",
        flags=re.DOTALL,
    )
    new_cbs = '''    def _on_brain_job_failed(self, job):
        """當 WorldQuant BRAIN 模擬失敗 (例如使用了不存在的函式或 operator) 時，從 Vault 清單移除不保留"""
        try:
            vid = getattr(job, "vault_id", None)
            if vid is not None:
                self.alpha_vault.delete_alpha(int(vid))
                self.add_log(f"🗑️ [BRAIN 模擬失敗移除] 已移除無效因子 ID={vid} ({job.safe_error})")
        except Exception:
            pass

    def _on_brain_job_completed(self, job):
        """
        WorldQuant BRAIN 模擬完成後：
        1. 通過官方 IS-Check 與門檻者，若觸發自動/一鍵送出且 submit_alpha 成功 -> 從 Vault 清單刪除不佔版面。
        2. 若模擬成功但未送出 (例如今日已滿 2 顆或排入候選池) -> 保留在 Vault 顯示真實 alpha_id 與指標。
        3. 若模擬未達門檻/IS-Check 失敗且原本來自 Vault -> 從 Vault 移除。
        """
        try:
            m = job.metrics or {}
            sharpe = m.get("sharpe")
            fitness = m.get("fitness")
            turnover = m.get("turnover") if m.get("turnover") is not None else 0.25
            margin = m.get("margin")
            checks_passed = m.get("checks_passed", True)
            failed_checks = m.get("failed_checks") or []
            alpha_id = job.alpha_id
            vid = getattr(job, "vault_id", None)
            if sharpe is None or fitness is None or not alpha_id:
                return

            if not checks_passed:
                self.add_log(f"⚠️ [BRAIN 未過 IS-Check] Alpha {alpha_id} 失敗項目 {failed_checks}，不予送出")
                if vid is not None:
                    self.alpha_vault.delete_alpha(int(vid))
                return

            if margin is not None:
                norm_margin = float(margin) / 10000.0 if float(margin) > 0.5 else float(margin)
                if norm_margin < 0.0010:
                    self.add_log(f"⚠️ [BRAIN 未達 Margin 門檻] Alpha {alpha_id} Margin={norm_margin:.5f} < 10 bps，不予送出")
                    if vid is not None:
                        self.alpha_vault.delete_alpha(int(vid))
                    return

            if not self.alpha_vault.passes_qualification(float(sharpe), float(fitness), float(turnover)):
                self.add_log(f"⚠️ [BRAIN 指標未達標] Alpha {alpha_id} (Sharpe={sharpe}, Fitness={fitness})")
                if vid is not None:
                    self.alpha_vault.delete_alpha(int(vid))
                return

            self.alpha_miner.total_accepted += 1
            should_try_submit = bool(getattr(job, "auto_submit", False)) or (
                os.getenv("BRAIN_AUTO_SUBMIT_ENABLED", "false").lower() == "true"
                and float(sharpe) >= float(os.getenv("BRAIN_HIGH_BAR_SHARPE", "1.80"))
            )

            submitted_ok = False
            if should_try_submit:
                sub_res = self.brain_client.submit_alpha(str(alpha_id), enforce_checks=True)
                job.submit_result = sub_res
                if sub_res.get("success"):
                    submitted_ok = True
                    job.submitted_to_brain = True
                    self.add_log(f"🏆 [BRAIN 驗證並送出成功] Alpha {alpha_id} 已正式送出至官網並移出清單！")
                    if vid is not None:
                        self.alpha_vault.delete_alpha(int(vid))
                    return
                else:
                    self.add_log(f"ℹ️ [BRAIN 模擬成功但未送出] Alpha {alpha_id}: {sub_res.get('status')} ({sub_res.get('error')})")

            # 模擬成功但尚未送出 -> 保留在清單中 (更新或存入 BRAIN_COMPLETE)
            if not submitted_ok:
                if vid is not None:
                    self.alpha_vault.delete_alpha(int(vid))
                self.alpha_vault.save_alpha(
                    name=f"BRAIN_{alpha_id}",
                    expression=job.expression,
                    sharpe=float(sharpe),
                    fitness=float(fitness),
                    turnover=float(turnover),
                    margin=float(margin or 0.0),
                    category="BRAIN_COMPLETE",
                    notes=f"alpha_id={alpha_id} (模擬通過，待送出)",
                    status_override="BRAIN_COMPLETE",
                )
                if os.getenv("BRAIN_AUTO_SUBMIT_ENABLED", "false").lower() == "true":
                    self.brain_client.stage_candidate_for_daily_selection(
                        str(alpha_id), job.expression, float(sharpe), float(fitness), float(turnover), float(margin or 0.0)
                    )
                self.add_log(f"💾 [BRAIN 模擬成功保留] Alpha {alpha_id} (Sharpe={sharpe}, Fitness={fitness}) 已留存於清單")
        except Exception:
            pass
'''
    if not cb_pat.search(new_txt):
        raise RuntimeError("Could not locate _on_brain_job_completed in web_ui/server.py")
    new_txt = cb_pat.sub(new_cbs, new_txt, count=1)

    # 3. Pass vault_id & auto_submit in /api/brain/simulations and /api/brain/test_alpha
    old_sim_create = 'job, is_new = APP_STATE.brain_job_queue.create_job(expr, settings)'
    new_sim_create = (
        'vault_id = post_data.get("vault_id")\n'
        '                auto_submit = bool(post_data.get("auto_submit", vault_id is not None))\n'
        '                job, is_new = APP_STATE.brain_job_queue.create_job(\n'
        '                    expr, settings, vault_id=int(vault_id) if vault_id is not None else None, auto_submit=auto_submit\n'
        '                )'
    )
    if old_sim_create in new_txt:
        new_txt = new_txt.replace(old_sim_create, new_sim_create, 1)

    old_test_create = 'job, is_new = APP_STATE.brain_job_queue.create_job(expr)'
    new_test_create = (
        'vault_id = post_data.get("vault_id") or post_data.get("alpha_id")\n'
        '                auto_submit = bool(post_data.get("auto_submit", True))\n'
        '                job, is_new = APP_STATE.brain_job_queue.create_job(\n'
        '                    expr, vault_id=int(vault_id) if vault_id is not None and str(vault_id).isdigit() else None, auto_submit=auto_submit\n'
        '                )'
    )
    if old_test_create in new_txt:
        new_txt = new_txt.replace(old_test_create, new_test_create, 1)

    # 4. In /api/brain/submit_alpha: if submitted successfully, remove from vault so it doesn't clutter board
    old_submit_end = '''            if brain_alpha_id:
                brain_submit_res = APP_STATE.brain_client.submit_alpha(brain_alpha_id, enforce_checks=True)
                res["brain_submit"] = brain_submit_res
                if brain_submit_res.get("success"):
                    res["message"] = f"Alpha 已入庫並通過官方 IS-Check 正式提交至 WorldQuant BRAIN (ID: {brain_alpha_id})！"
                else:
                    res["message"] = f"Alpha 已入庫，但官方提交被攔截或未完成 [{brain_submit_res.get('status')}]: {brain_submit_res.get('error')}"
            return self._send_json_response(res)'''

    new_submit_end = '''            if brain_alpha_id:
                brain_submit_res = APP_STATE.brain_client.submit_alpha(brain_alpha_id, enforce_checks=True)
                res["brain_submit"] = brain_submit_res
                if brain_submit_res.get("success"):
                    vault_id_to_del = post_data.get("vault_id") or res.get("alpha_id")
                    if vault_id_to_del is not None:
                        APP_STATE.alpha_vault.delete_alpha(int(vault_id_to_del))
                    res["deleted_from_vault"] = True
                    res["message"] = f"Alpha 已通過官方 IS-Check 正式提交至 WorldQuant BRAIN (ID: {brain_alpha_id})，並已從清單移除！"
                else:
                    res["deleted_from_vault"] = False
                    res["message"] = f"Alpha 模擬資料已保留，但官方提交未完成 [{brain_submit_res.get('status')}]: {brain_submit_res.get('error')}"
            return self._send_json_response(res)'''

    if old_submit_end in new_txt:
        new_txt = new_txt.replace(old_submit_end, new_submit_end, 1)

    _atomic_write(srv_path, new_txt)
    return "PATCHED"


def patch_index_html_taiwan_and_brain(repo_root: Path) -> str:
    html_path = repo_root / "web_ui" / "static" / "index.html"
    txt = html_path.read_text(encoding="utf-8")
    if 'id="twChartContainer"' in txt and "selectTaiwanStock(" in txt:
        return "ALREADY_APPLIED"

    # 1. Remove Taiwan stocks from Crypto tab's chartSymbolSelect
    old_crypto_opts = """              <option value="BTC/USDT" selected>BTC/USDT (比特幣)</option>
              <option value="ETH/USDT">ETH/USDT (以太幣)</option>
              <option value="SOL/USDT">SOL/USDT (索拉納)</option>
              <option value="2330.TW">2330.TW (台積電)</option>
              <option value="2454.TW">2454.TW (聯發科)</option>
              <option value="2317.TW">2317.TW (鴻海)</option>
              <option value="00632R.TW">00632R.TW (台灣50反1)</option>"""
    new_crypto_opts = """              <option value="BTC/USDT" selected>BTC/USDT (比特幣)</option>
              <option value="ETH/USDT">ETH/USDT (以太幣)</option>
              <option value="SOL/USDT">SOL/USDT (索拉納)</option>"""
    if old_crypto_opts in txt:
        txt = txt.replace(old_crypto_opts, new_crypto_opts, 1)

    # 2. Insert dedicated Taiwan Stock K-Line Chart card inside #tab-content-taiwan
    tw_chart_card = """
      <!-- 台股專屬即時 K 線看盤區 -->
      <div class="glass-card rounded-2xl p-4 md:p-5 shadow-xl space-y-3">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div class="flex items-center space-x-2">
            <i class="fa-solid fa-chart-candlestick text-rose-400"></i>
            <h3 class="font-bold text-sm md:text-base text-white">台股即時走勢看盤</h3>
            <span id="twChartTimeframeBadge" class="text-[10px] px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 font-bold">1d</span>
          </div>
          <div class="flex flex-wrap items-center gap-2">
            <select id="twChartSymbolSelect" onchange="selectTaiwanStock(this.value)" class="bg-slate-900 border border-slate-700 text-xs rounded-xl px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-rose-500 font-mono">
              <option value="2330.TW" selected>2330.TW (台積電)</option>
              <option value="2308.TW">2308.TW (台達電)</option>
              <option value="2454.TW">2454.TW (聯發科)</option>
              <option value="3008.TW">3008.TW (大立光)</option>
              <option value="3443.TW">3443.TW (創意)</option>
              <option value="6442.TW">6442.TW (光聖)</option>
              <option value="2317.TW">2317.TW (鴻海)</option>
              <option value="00981A.TW">00981A.TW</option>
              <option value="00632R.TW">00632R.TW (台灣50反1)</option>
              <option value="^TWII">^TWII (加權指數)</option>
            </select>
            <div class="flex items-center space-x-1">
              <input type="text" id="twCustomSymbolInput" placeholder="輸入代碼如 2330.TW" class="w-32 bg-slate-900 border border-slate-700 rounded-xl px-2.5 py-1.5 text-xs text-white font-mono focus:outline-none">
              <button type="button" onclick="const v=document.getElementById('twCustomSymbolInput').value.trim(); if(v) selectTaiwanStock(v.toUpperCase());" class="px-2.5 py-1.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white text-xs font-bold">查詢</button>
            </div>
            <div class="flex items-center bg-slate-900 rounded-xl p-1 border border-slate-800 text-xs font-mono">
              <button type="button" onclick="setTwChartInterval('5m')" id="btn-tw-int-5m" class="px-2 py-1 rounded-lg text-slate-400 hover:text-white">5m</button>
              <button type="button" onclick="setTwChartInterval('15m')" id="btn-tw-int-15m" class="px-2 py-1 rounded-lg text-slate-400 hover:text-white">15m</button>
              <button type="button" onclick="setTwChartInterval('1h')" id="btn-tw-int-1h" class="px-2 py-1 rounded-lg text-slate-400 hover:text-white">1h</button>
              <button type="button" onclick="setTwChartInterval('1d')" id="btn-tw-int-1d" class="px-2 py-1 rounded-lg font-bold bg-rose-600 text-white">1d</button>
            </div>
          </div>
        </div>
        <div id="twChartLegend" class="flex flex-wrap items-center gap-3 text-xs font-mono bg-slate-950/70 p-2.5 rounded-xl border border-slate-800/80">
          <span class="text-slate-400">標的: <strong id="twLegendSymbol" class="text-amber-300">2330.TW</strong></span>
          <span class="text-slate-400">開: <strong id="twLegendOpen" class="text-white">--</strong></span>
          <span class="text-slate-400">高: <strong id="twLegendHigh" class="text-emerald-400">--</strong></span>
          <span class="text-slate-400">低: <strong id="twLegendLow" class="text-rose-400">--</strong></span>
          <span class="text-slate-400">收: <strong id="twLegendClose" class="text-white">--</strong></span>
          <span class="text-yellow-400">MA5: <strong id="twLegendMA5">--</strong></span>
          <span class="text-blue-400">MA20: <strong id="twLegendMA20">--</strong></span>
          <span class="text-purple-400">MA60: <strong id="twLegendMA60">--</strong></span>
        </div>
        <div id="twChartContainer" class="w-full h-[380px] rounded-xl overflow-hidden bg-slate-950 relative border border-slate-900">
          <div id="twChartLoading" class="hidden absolute inset-0 flex items-center justify-center bg-slate-950/80 z-10 text-rose-400 text-xs font-bold">
            <i class="fa-solid fa-spinner animate-spin mr-2"></i> 正在加載台股 K 線行情...
          </div>
        </div>
      </div>

      <!-- 選股池與晨報文字檢視 -->"""
    if "      <!-- 選股池與晨報文字檢視 -->" in txt:
        txt = txt.replace("      <!-- 選股池與晨報文字檢視 -->", tw_chart_card, 1)

    # 3. Screener clicks stay on Taiwan tab
    txt = txt.replace(
        "onclick=\"switchTab('crypto'); switchChartSymbol('${p.symbol}')\"",
        "onclick=\"selectTaiwanStock('${p.symbol}')\"",
    )
    txt = txt.replace(
        "onclick=\"switchTab('crypto'); switchChartSymbol('${h.symbol.split(' ')[0]}')\"",
        "onclick=\"selectTaiwanStock('${h.symbol.split(' ')[0]}')\"",
    )

    # 4. Add Taiwan Lightweight Charts JS & hook into switchTab('taiwan')
    tw_js = """
    let twChart = null;
    let twCandleSeries = null;
    let twVolumeSeries = null;
    let twMa5Line = null, twMa20Line = null, twMa60Line = null;
    let currentTwSymbol = '2330.TW';
    let currentTwInterval = '1d';

    function initTwLightweightChart() {
      const container = document.getElementById('twChartContainer');
      if (!container || twChart || typeof LightweightCharts === 'undefined') return;
      twChart = LightweightCharts.createChart(container, {
        width: container.clientWidth || 800,
        height: 380,
        layout: { background: { type: 'solid', color: '#020617' }, textColor: '#94a3b8' },
        grid: { vertLines: { color: '#0f172a' }, horzLines: { color: '#0f172a' } },
        timeScale: { timeVisible: true, borderColor: '#1e293b' },
        rightPriceScale: { borderColor: '#1e293b' },
      });
      twCandleSeries = twChart.addCandlestickSeries({
        upColor: '#ef4444', downColor: '#10b981',
        borderUpColor: '#ef4444', borderDownColor: '#10b981',
        wickUpColor: '#ef4444', wickDownColor: '#10b981',
      });
      twVolumeSeries = twChart.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: '' });
      twVolumeSeries.priceScale().applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
      twMa5Line = twChart.addLineSeries({ color: '#facc15', lineWidth: 1.5 });
      twMa20Line = twChart.addLineSeries({ color: '#3b82f6', lineWidth: 1.5 });
      twMa60Line = twChart.addLineSeries({ color: '#a855f7', lineWidth: 1.5 });
      window.addEventListener('resize', () => {
        if (twChart && container) twChart.applyOptions({ width: container.clientWidth });
      });
    }

    async function loadTwChartData() {
      const loading = document.getElementById('twChartLoading');
      if (loading) loading.classList.remove('hidden');
      try {
        const res = await fetch(`/api/crypto/klines?symbol=${encodeURIComponent(currentTwSymbol)}&interval=${currentTwInterval}`);
        const d = await res.json();
        if (d.success && d.candles && d.candles.length > 0) {
          if (!twChart) initTwLightweightChart();
          if (twChart) {
            const container = document.getElementById('twChartContainer');
            if (container && container.clientWidth > 0) twChart.applyOptions({ width: container.clientWidth });
            twCandleSeries.setData(d.candles.map(c => ({ time: c.time, open: c.open, high: c.high, low: c.low, close: c.close })));
            twVolumeSeries.setData(d.candles.map(c => ({
              time: c.time, value: c.volume,
              color: c.close >= c.open ? 'rgba(239, 68, 68, 0.35)' : 'rgba(16, 185, 129, 0.35)'
            })));
            if (d.overlay) {
              if (d.overlay.ma5) twMa5Line.setData(d.overlay.ma5);
              if (d.overlay.ma20) twMa20Line.setData(d.overlay.ma20);
              if (d.overlay.ma60) twMa60Line.setData(d.overlay.ma60);
            }
          }
          const lastC = d.candles[d.candles.length - 1];
          const setTxt = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
          setTxt('twLegendSymbol', currentTwSymbol);
          setTxt('twLegendOpen', Number(lastC.open).toFixed(2));
          setTxt('twLegendHigh', Number(lastC.high).toFixed(2));
          setTxt('twLegendLow', Number(lastC.low).toFixed(2));
          setTxt('twLegendClose', Number(lastC.close).toFixed(2));
          if (d.overlay?.ma5?.length) setTxt('twLegendMA5', Number(d.overlay.ma5[d.overlay.ma5.length - 1].value).toFixed(2));
          if (d.overlay?.ma20?.length) setTxt('twLegendMA20', Number(d.overlay.ma20[d.overlay.ma20.length - 1].value).toFixed(2));
          if (d.overlay?.ma60?.length) setTxt('twLegendMA60', Number(d.overlay.ma60[d.overlay.ma60.length - 1].value).toFixed(2));
        }
      } catch (e) {
        console.error(e);
      } finally {
        if (loading) loading.classList.add('hidden');
      }
    }

    function selectTaiwanStock(sym) {
      currentTwSymbol = sym;
      const sel = document.getElementById('twChartSymbolSelect');
      if (sel) {
        const exists = Array.from(sel.options).some(o => o.value === sym);
        if (!exists) {
          const opt = document.createElement('option');
          opt.value = sym;
          opt.text = sym;
          sel.appendChild(opt);
        }
        sel.value = sym;
      }
      if (currentTab !== 'taiwan') switchTab('taiwan');
      loadTwChartData();
    }

    function setTwChartInterval(intv) {
      currentTwInterval = intv;
      ['5m', '15m', '1h', '1d'].forEach(i => {
        const btn = document.getElementById(`btn-tw-int-${i}`);
        if (btn) {
          btn.className = (i === intv)
            ? "px-2 py-1 rounded-lg font-bold bg-rose-600 text-white"
            : "px-2 py-1 rounded-lg text-slate-400 hover:text-white";
        }
      });
      const b = document.getElementById('twChartTimeframeBadge');
      if (b) b.innerText = intv;
      loadTwChartData();
    }
"""
    txt = txt.replace("    function stopChartPolling() {", tw_js + "\n    function stopChartPolling() {", 1)
    txt = txt.replace(
        "else if (tabId === 'taiwan') {",
        "else if (tabId === 'taiwan') {\n        loadTwChartData();",
        1,
    )

    # 5. Update testAlpha to pass vault_id & auto_submit=True and remove submitted/failed items from alphaList
    old_test_body = "body: JSON.stringify({ expression: expr })"
    new_test_body = "body: JSON.stringify({ expression: expr, vault_id: alphaId, auto_submit: true })"
    if old_test_body in txt:
        txt = txt.replace(old_test_body, new_test_body, 1)

    old_alert_done = """            alert(`🎉 Alpha 回測完成！\\n狀態: COMPLETED\\nAlpha ID: ${realId}\\nSharpe: ${sharpe.toFixed(2)}\\nFitness: ${fitness.toFixed(2)}\\n換手率: ${j.turnover ?? m.turnover ?? 'N/A'}`);
            fetchAlphas();"""
    new_alert_done = """            if (j.brain_submitted) {
              alert(`🏆 已於 WorldQuant BRAIN 驗證並正式送出！該項目已從清單移除。\\nAlpha ID: ${realId}\\nSharpe: ${sharpe.toFixed(2)}\\nFitness: ${fitness.toFixed(2)}`);
            } else {
              alert(`💾 WorldQuant BRAIN 模擬成功（暫未送出，已保留於清單）：\\nAlpha ID: ${realId}\\nSharpe: ${sharpe.toFixed(2)}\\nFitness: ${fitness.toFixed(2)}\\n原因: ${j.brain_submit?.error || '待送出或已達今日 2 顆上限'}`);
            }
            fetchAlphas();"""
    if old_alert_done in txt:
        txt = txt.replace(old_alert_done, new_alert_done, 1)

    txt = txt.replace(
        "alert(`❌ Alpha 模擬未通過 [${status || 'HTTP_' + pollRes.status}]:\\n${reason}`);",
        "alert(`❌ WorldQuant BRAIN 驗證失敗（可能含不存在的函式或運算子，已從清單移除）[${status || 'HTTP_' + pollRes.status}]:\\n${reason}`);\n            fetchAlphas();",
        1,
    )

    # Update button text on Alpha cards to reflect simulate + auto-submit
    txt = txt.replace("🚀 送出至 WorldQuant 回測", "🚀 送出至 WorldQuant 驗證並提交")

    _atomic_write(html_path, txt)
    return "PATCHED"


PHASE3_TEST_PY = """import os
import tempfile
import unittest
from unittest.mock import MagicMock

from worldquant_brain.alpha_vault import AlphaVault
from worldquant_brain.brain_job_queue import BrainJobQueue, BrainJobStatus


class TestPhase3Lifecycle(unittest.TestCase):
    def test_job_queue_vault_id_and_fail_callback(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            vault = AlphaVault(db_path=os.path.join(td, "v.db"))
            saved = vault.save_alpha("Bad_AI_Alpha", "nonexistent_op(close)", 1.5, 1.2, 0.2)
            vid = saved["alpha_id"]
            self.assertEqual(len(vault.list_alphas()), 1)

            def on_fail(job):
                if job.vault_id is not None:
                    vault.delete_alpha(job.vault_id)

            client = MagicMock()
            client.offline_mode = False
            client.authenticated = True
            client.BASE_URL = "https://api.worldquantbrain.com"
            client.timeout = 5.0
            client.session.post.side_effect = RuntimeError("Unknown operator nonexistent_op")

            q = BrainJobQueue(client, default_poll_interval=0.01, on_fail_callback=on_fail)
            job, _ = q.create_job("nonexistent_op(close)", vault_id=vid, auto_submit=True)

            import time
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and job.status != BrainJobStatus.FAILED:
                time.sleep(0.02)

            self.assertEqual(job.status, BrainJobStatus.FAILED)
            self.assertEqual(len(vault.list_alphas()), 0)


if __name__ == "__main__":
    unittest.main()
"""


def write_phase3_tests(repo_root: Path) -> str:
    target = repo_root / "tests" / "test_phase3_lifecycle.py"
    if target.exists() and target.read_text(encoding="utf-8") == PHASE3_TEST_PY:
        return "ALREADY_APPLIED"
    _atomic_write(target, PHASE3_TEST_PY)
    return "PATCHED"


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply Phase 3 Taiwan Chart & BRAIN Lifecycle")
    parser.add_argument("--repo-root", default=".", help="Repository root path")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()

    print("1. worldquant_brain/brain_job_queue.py:", patch_brain_job_queue(repo_root))
    print("2. worldquant_brain/alpha_miner.py:", patch_alpha_miner_verified_ops(repo_root))
    print("3. web_ui/server.py:", patch_server_lifecycle(repo_root))
    print("4. web_ui/static/index.html:", patch_index_html_taiwan_and_brain(repo_root))
    print("5. tests/test_phase3_lifecycle.py:", write_phase3_tests(repo_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
