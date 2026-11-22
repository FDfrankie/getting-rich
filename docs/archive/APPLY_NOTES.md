# Antigravity Complete Enhancements Bundle

## Included Files (14)
1. `worldquant_brain/pre_submission_gate.py` — Hard pre-submission gate enforcing Sharpe, Fitness, Turnover, Margin, Self-Correlation (`< 0.70`), and official IS-check verification.
2. `worldquant_brain/direct_submit_service.py` — Integrates `evaluate_pre_submission_gate` before direct BRAIN alpha submission.
3. `worldquant_brain/submission_service.py` — Integrates `evaluate_pre_submission_gate` into `execute_daily_top_k`.
4. `worldquant_brain/brain_job_queue.py` — Adds SQLite transient I/O retry (`disk I/O error` / `database is locked`) so Google Drive sync locks do not crash worker threads.
5. `web_ui/server.py` — Attaches `reconciliation_service` to `active_bot`, supports multi-symbol background stepping, and resolves per-symbol instrument specs.
6. `crypto_trading/bot_runner.py` — Normalizes `amount` (`step_size`) and `current_price` (`tick_size`), and sets `exchange_connected=True` in `LOCAL_PAPER` mode.
7. `crypto_trading/risk_manager.py` — Centralized fail-closed risk gate with step/tick and drawdown protection.
8. `crypto_trading/walk_forward_guard.py` — Purged chronological walk-forward fold builder and OOS Sharpe/drawdown validator.
9. `portfolio/__init__.py` & `portfolio/risk_allocator.py` — Multi-symbol exposure and max-open-position risk allocator.
10. `notifications/__init__.py` & `notifications/alert_manager.py` — Cooldown-deduplicated alert manager.
11. `tests/test_antigravity_enhancements.py` — Offline `unittest`/`pytest` verification suite.
12. `APPLY_NOTES.md` — This integration guide.

## Verification Commands
```powershell
python -m compileall -q worldquant_brain web_ui crypto_trading portfolio notifications tests
python -m pytest tests/test_antigravity_enhancements.py -v
```
