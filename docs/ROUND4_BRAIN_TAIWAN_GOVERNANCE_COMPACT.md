/goal /browser /plan /teamwork /learn /boost /migrate-workflows /swarm

# ROUND4_BRAIN_TAIWAN_GOVERNANCE_COMPACT.md

> **Execution Scope**: P9（WorldQuant BRAIN 正式流程與受控 Auto-Submit）+ P10（台股研究、雙時段排程與反向 ETF 隔離）+ P11（Notebook 去漂移與模型治理）。
> **嚴禁**：執行真實 BRAIN Production Submit、在 `pytest` 連外、啟動 `BITUNIX_LIVE`、發送真實資金委託。

## 0. Token 紀律
1. 不要重述規格、不要建新規格檔、不要先輸出長篇計畫，直接改程式與測試。
2. 每完成一個 Phase 即跑 `python -m compileall -q .` 與 `pytest -m "not live" -q` 並獨立 commit，不得把 `NOT RUN` 寫成 `PASS`。

## 1. P9：WorldQuant BRAIN 正式流程與受控 Auto-Submit
- 嚴禁以 BTC proxy 冒充股票 FASTEXPR 的 BRAIN 指標；實作完整狀態機（`GENERATED ... -> BRAIN_COMPLETE / BRAIN_REJECTED -> SUBMISSION_CHECK_PENDING -> ... -> SUBMITTED`）。
- 正式流程：auth → POST simulations → 持久化 `Location` → poll → `alpha_id` → GET official details → submission checks/correlation/sub-universe → hard gate → optional auto-submit；queue 與 raw response 落 DB 支援 restart recovery。
- 缺少 `alpha_id`/`Location` 的舊 `VERIFIED` 遷移為 `LEGACY_UNVERIFIED`；local proxy 改 `LOCAL_SCREENED`；seed 改 `DEMO_SAMPLE`。
- Auto-Submit 預設關閉，需 `BRAIN_AUTO_SUBMIT_ACK=I_UNDERSTAND_SUBMISSION_IS_IRREVERSIBLE`、quota、cooldown、single-worker claim、SHA-256 冪等、kill switch 與 audit log。
- 新增 8 項 mock 測試：`test_brain_official_metrics.py`、`test_brain_restart_recovery.py`、`test_alpha_status_migration.py`、`test_brain_auto_submit_gate.py`、`test_brain_submission_idempotency.py`、`test_brain_submission_recovery.py`、`test_brain_submission_quota.py`、`test_brain_submission_kill_switch.py`。

## 2. P10：台股研究、雙時段排程與反向 ETF 隔離
- Target 未扣 `^TWII` 或 `0050.TW` 一律標示 `RAW_RETURN`；Rank IC 採每日 cross-sectional Spearman（`< 10` 檔不納入正式 IC；僅 8 檔時 UI 顯示 `Watchlist Ranking, Universe Size=8`）。
- `ReportGenerator` 改從 Model Registry 載入已驗證 artifact（無合格模型標 `RULE_BASED` fallback），禁止在 Web request 現場下載重訓。
- Scheduler 改用 `ZoneInfo("Asia/Taipei")`（08:00 前一交易日、16:30 當日資料完成否則延遲重試）；移除「無紀錄紀律率 100%」；金額採 `NT$`/`TWD`；HTML/Markdown escape；隔離反向 ETF（`00632R`）。
- 新增：`tests/test_taiwan_cross_sectional_ic.py`、`tests/test_scheduler_timezone.py`、`tests/test_report_target_mode.py`、`tests/test_report_escaping.py`。

## 3. P11：Notebook 去漂移與模型治理
- `kaggle_notebooks/` 改用 `pip install -e .` import 正式模組；隔離無 manifest 的 `.joblib`/`.pt` 至 `artifacts/legacy_unverified/`；PyTorch 優先 `state_dict` 與 `weights_only=True`，並驗證 manifest SHA-256 原子熱載入。
- 新增：`tests/test_notebook_drift.py`、`tests/test_model_manifest.py`、`tests/test_artifact_trust.py`、`tests/test_ppo_action_distribution.py`。
最終狀態只能是 `PHASE_COMPLETE`、`PARTIAL` 或 `BLOCKED`。
