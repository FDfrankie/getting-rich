# Research Sources and Implementation Map

## Official Taiwan market sources

- TWSE security-code query: https://wwwc.twse.com.tw/zh/products/code/query.html
- TWSE market information services: https://www.twse.com.tw/zh/products/information/information.html
- TPEx OpenAPI Swagger: https://www.tpex.org.tw/openapi/
- TPEx company lookup: https://www.tpex.org.tw/zh-tw/mainboard/listed/company.html
- Government open data, TPEx company fundamentals: https://data.gov.tw/dataset/25036

Use official symbol and point-in-time snapshots for the production universe. `yfinance` remains a secondary price convenience, not the authoritative security master.

## UI reference

Yuanta Investor APP principles to adapt, not clone: simple/pro mode, whole-share/odd-lot switch, visual institution/chip/ETF-ownership cards, and account-context proximity. Source: https://www.yuanta.com.tw/eYuanta/securiti/Node/Index/2026101508436650?MainId=00409&C1=2026100209727353&Level=2

## Video research

The machine-readable list is `research_catalog/video_catalog.json`. Treat every claim as a hypothesis. Prioritize videos that explain indicator failure conditions, timeframe dependence, lag, costs, and overfitting. Avoid copying proprietary scripts or treating clickbait performance claims as evidence.

## Cloud architecture sources

- Render cron jobs: https://render.com/docs/cronjobs
- Render background-worker concepts: https://github.com/render-oss/skills/blob/main/skills/render-background-workers/SKILL.md

Cron is UTC, bounded, and diskless. Long-running Alpha mining must use a worker and an external durable database/queue. GitHub Actions should run CI and bounded jobs, not hold the sole Alpha queue.
