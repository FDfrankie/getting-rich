# LightGBM Strict OOS & Cost Stress Validation Report
- Gate Passed: **False** | Recommended Stage: **SHADOW**
- Global Split Test Rank IC: **-0.008288** (Val Rank IC: 0.02709)

## Cost Stress Scenarios (Global Test)
| Scenario | Avg Net Return % | Avg Daily Sharpe | Avg Max DD % | Total Trades |
| :--- | :--- | :--- | :--- | :--- |
| base | -99.3189% | -75.4012 | 99.3205% | 25164 |
| stress_1_5x | -99.9457% | -90.5705 | 99.9458% | 25164 |
| severe_2_0x_funding | -99.9957% | -98.9329 | 99.9957% | 25164 |

## Leave-One-Asset-Out (LOAO)
| Held-Out Symbol | Rank IC | Base Daily Sharpe | Base Net Return % |
| :--- | :--- | :--- | :--- |
| BTC/USDT | 0.039829 | -87.5831 | -100.0% |
| ETH/USDT | 0.050371 | -79.2756 | -100.0% |
| SOL/USDT | 0.034143 | -90.2756 | -100.0% |
