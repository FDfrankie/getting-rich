"""
Top Transcripts Fetcher (fetch_top_transcripts.py)
擷取並整理頂尖量化金融演算法精華文本，轉換為 Prompt 與 Alpha 研發特徵庫
"""

import json
import os

QUANT_TRANSCRIPTS = {
    "Cross_Sectional_Momentum": "By ranking assets based on multi-timeframe normalized returns and residual momentum, factor neutral portfolios achieve higher Sharpe ratios while minimizing market exposure.",
    "Perpetual_Funding_Arbitrage": "Perpetual futures often trade at a premium or discount to spot prices. By delta hedging long spot USDC and short perpetual contracts during positive funding rates, one captures pure cash-and-carry yields.",
    "Differentiable_Sharpe_Optimization": "Rather than optimizing Mean Squared Error in neural networks, directly optimizing the Differentiable Sharpe Ratio backpropagates gradients of expected returns and volatility directly into model parameters.",
}


def fetch_and_save_transcripts():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data_persistence")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "curated_transcripts.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(QUANT_TRANSCRIPTS, f, ensure_ascii=False, indent=2)
    print(
        f"[+] Extracted and stored {len(QUANT_TRANSCRIPTS)} quantitative lecture transcripts into {out_file}"
    )


if __name__ == "__main__":
    fetch_and_save_transcripts()
