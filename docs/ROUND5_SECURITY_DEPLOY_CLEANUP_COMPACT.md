/goal /browser /plan /teamwork /learn /boost /migrate-workflows /swarm

# ROUND5_SECURITY_DEPLOY_CLEANUP_COMPACT.md

> **Execution Scope**: P12（Web 權限、UI 安全中心、Dockerfile 與部署）+ P13（`pyproject.toml` 封裝、論文真實性、全專案整理與刪除無用檔案）+ 最終全量驗收。
> **嚴禁**：顯示或提交 Secret、自動 force push、刪除測試、在條件未全數滿足前輸出 `GOAL_COMPLETE`。

## 0. Token 紀律
1. 不要重述規格、不要建新規格檔、直接改程式、整理目錄並跑最終驗證。
2. 不得把 `NOT RUN` 寫成 `PASS`。

## 1. P12：Web 權限、解耦與部署
- Web route 嚴禁直接送單、執行 ML training、yfinance download 或 BRAIN polling；長任務寫入 `commands` queue，由獨立 worker 執行。
- 建立認證與角色權限（`VIEWER`/`OPERATOR`/`ADMIN`）、Argon2id、secure HttpOnly cookie、SameSite、CSRF、Origin validation、rate limit、request size limit、audit log；無 Cloudflare Access 時強制 read-only。
- 移除 UI／文件中的未證實宣稱（固定 WorldQuant 分潤公式、4,800 美元預估、保證 24/7、Quick Tunnel 極度安全、永久零費率、一鍵正式提交等）。
- 補齊唯讀健康端點：`/health/live`、`/health/ready`、`/health/trading`、`/health/alpha-miner`、`/health/scheduler`。
- 修復 `Dockerfile` line continuation、`.dockerignore`、non-root user、healthcheck（不帶入 `.env`/DB/runtime files），實際執行 `docker build`。
- 新增：`tests/test_ui_safety_state.py`、`tests/test_health_endpoints.py`、`tests/test_web_authz.py`。

## 2. P13 & 全專案徹底整理
1. 建立 `pyproject.toml`（拆 optional dependencies）與去除 `sys.path.insert`，改用正式 `pip install -e .`。
2. `collect_crypto_videos.py` 與 `fetch_top_transcripts.py` 若為硬編碼佔位符，標註 `SYNTHETIC_PLACEHOLDER` 或刪除。
3. 驗證論文 reference 並標註 `PAPER_MENTIONED / PAPER_INSPIRED / PAPER_ADAPTED / PAPER_REPRODUCED / NOT_IMPLEMENTED`（未驗證 2025/2026 arXiv 標 `UNVERIFIED_REFERENCE`）。
4. 合併功能重複的模組；刪除過時腳本（如 `start_cloudflare_tunnel.bat`）、空資料夾、測試殘留、以及 `docs/` 下過時的 Batch／Round／Prompt 暫存檔；**嚴格保留**：`GEMINI.md`、`README.md`、`docs/BATCH_ACCEPTANCE_MATRIX.md`、`docs/BRAIN_AUTO_SUBMIT_ACCEPTANCE_MATRIX.md`、`docs/USER_ACTION_GUIDE.md`、`docs/handbook/`、`docs/research/`、`docs/adr/`。
5. 確認全部 `import` 路徑、`pyproject.toml`、`Dockerfile`、`render.yaml`、`kaggle_notebooks/` 與 `README.md` 目錄樹完全對齊無斷裂。

## 3. 最終全量驗收命令
實際執行並誠實回報（工具未安裝者寫 `NOT RUN`）：
```bash
python -m compileall -q .
pytest -m "not live" -q
ruff check .
ruff format --check .
gitleaks detect --source . --verbose
docker build --no-cache -t antigravity-quant:test .
python scripts/package_for_review.py
```
最後列出「已合併／已刪除清單」、15 點最終驗收狀態與部署階段。所有驗收條件全數成立才可輸出 `GOAL_COMPLETE`，否則回報 `PARTIAL` 或 `BLOCKED`。
