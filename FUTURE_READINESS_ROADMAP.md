# Future Readiness Roadmap

## Immediate foundations

1. Official Taiwan universe snapshots: TWSE + TPEx mainboard, optional emerging market, daily manifests and point-in-time history.
2. Research-video catalogue: educational claims are hypotheses, never production gates.
3. Shared causal indicator library: MA/EMA, RSI, MACD, Bollinger, ATR, OBV, VWAP and Stochastic.
4. BRAIN-to-crypto bridge: only delayed aggregate research context, no direct order permission and no same-day look-ahead.
5. Cloud separation: web, persistent worker, cron, PostgreSQL; no SQLite or CSV as the only cloud state.

## Taiwan UI inspired by Yuanta's interaction principles

- Simple/pro mode switch.
- Whole-share/odd-lot context switch.
- Institutional flow, chip changes, ETF ownership and inventory PnL shown as compact visual cards.
- Search all TWSE/TPEx symbols, watchlists, sector heatmap, price/volume, foreign/investment-trust/dealer flows.
- Read-only research first. Brokerage order integration is a separate audited phase.

## Model research gates

Every video-derived hypothesis requires: frozen formula, causal features, purged walk-forward, costs, multiple markets/timeframes, ablation, false-discovery control, shadow results and a model manifest.

## Deployment rule

Render cron uses UTC, has no persistent disk and must exit. Alpha mining belongs in a background worker with PostgreSQL/queue persistence. GitHub Actions is CI or bounded scheduled orchestration, not the authoritative long-running queue.
