import os
import html
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
try:
    from zoneinfo import ZoneInfo
    _TAIPEI_TZ = ZoneInfo("Asia/Taipei")
except Exception:
    _TAIPEI_TZ = timezone(timedelta(hours=8))

@dataclass(frozen=True)
class ReportSnapshot:
    status: str
    session_type: str
    generated_at: str
    last_valid_trade_date: str | None
    market_info: dict
    screener_res: dict

import pandas as pd
import yfinance as yf

from data_persistence.gdrive_sync import GDrivePersistenceSync
from taiwan_stock.config import DEFAULT_WATCHLIST, REPORTS_DIR
from taiwan_stock.market_analyzer import MarketAnalyzer
from taiwan_stock.ml_ranker import TaiwanStockMLRanker
from taiwan_stock.stock_screener import StockScreener


class ReportGenerator:
    """
    台股每日晨報自動生成器
    整合大盤多空結構、技術突破選股與 ML 多因子預測排序，輸出格式化 Markdown 報告與文字摘要
    """

    def __init__(self):
        self.market_analyzer = MarketAnalyzer()
        self.stock_screener = StockScreener()
        self.ml_ranker = TaiwanStockMLRanker(model_type="lightgbm")
        self.sync = GDrivePersistenceSync()
        self._last_snapshot: ReportSnapshot | None = None

    def collect_report_snapshot(self, session_type: str = '盤前晨報', cached_market_info: dict = None) -> ReportSnapshot:
        now_str = datetime.now(_TAIPEI_TZ).strftime('%Y-%m-%d %H:%M:%S')
        m_info = cached_market_info if cached_market_info else self.market_analyzer.analyze()
        is_valid = m_info.get('status') == '正常' and m_info.get('latest_close') not in (None, 'N/A')
        s_res = self.stock_screener.run_screener_detailed() if is_valid else {'scanned_count': 0, 'qualified_count': 0, 'picks': [], 'near_picks': [], 'defense_hedges': []}
        snap = ReportSnapshot(
            status='VALID' if is_valid else 'DATA_UNAVAILABLE',
            session_type=session_type,
            generated_at=now_str,
            last_valid_trade_date=m_info.get('date') if is_valid else None,
            market_info=m_info,
            screener_res=s_res,
        )
        self._last_snapshot = snap
        return snap

    def generate(self, session_type: str = "盤前晨報", cached_market_info: dict = None) -> str:
        today_str = datetime.now(_TAIPEI_TZ).strftime("%Y-%m-%d")
        now_time_str = datetime.now(_TAIPEI_TZ).strftime("%Y-%m-%d %H:%M:%S")

        snap = self.collect_report_snapshot(session_type=session_type, cached_market_info=cached_market_info)
        if snap.status != 'VALID':
            return f"# 📊 台股量化決策報告【{session_type}】({today_str})\n\nREPORT_STATUS: DATA_UNAVAILABLE\n\n> ⚠️ 無法取得有效大盤收盤資料，已中止計算多空分數與選股建議。"
        market_info = snap.market_info
        screener_res = snap.screener_res
        scanned_count = screener_res["scanned_count"]
        qualified_count = screener_res["qualified_count"]
        picks = screener_res["picks"]
        near_picks = screener_res["near_picks"]
        defense_hedges = screener_res["defense_hedges"]

        last_trade_date = (market_info.get("date") or "DATA_UNAVAILABLE")

        report_lines = [
            f"# 📊 台股量化決策報告【{session_type}】({today_str})",
            f"*最後交易日: {last_trade_date} | 產出時間: {now_time_str} (Asia/Taipei) | 系統: Antigravity Automated Quant Agent*",
            "",
            "---",
            "## 🧭 1. 大盤環境與多空溫度診斷",
            f"- **加權指數最新收盤**: `{market_info.get('latest_close', 'N/A')}` ({market_info.get('change_pct', 0):+0.2f}%)",
            f"- **關鍵均線狀況**: MA5=`{market_info.get('ma5')}` | MA20(月線)=`{market_info.get('ma20')}` | MA60(季線)=`{market_info.get('ma60')}`",
            f"- **均線型態判定**: `{market_info.get('ma_pattern', '多空判斷')}`",
            f"- **多空評分**: **{market_info.get('score', 50)} / 100**",
            f"- **市場狀態**: **【{market_info.get('regime', '觀望')}】**",
            f"- **20日年化波動率**: `{market_info.get('volatility_20d', 0)}%`",
            f"- **市場量能狀態**: `{market_info.get('volume_status', '正常')}`",
            f"- **操作方針**: > {market_info.get('suggestion', '無')}",
            "",
            "---",
            "## 🎯 2. 技術選股透明度與推薦標的",
            f"**選股掃描統計**：已掃描 `{scanned_count}` 檔標的，符合突破門檻 `{qualified_count}` 檔。",
            "",
        ]

        if not picks:
            report_lines.append("> ⚠️ 今日篩選池中無標的完全符合突破門檻 (分數 >= 50 分)，系統啟動防守保護機制！")
            if near_picks:
                report_lines.append("\n### 🔍 前 3 檔最接近門檻標的 (密切追蹤名單)：")
                report_lines.append("| 股票代碼 | 收盤價 | 漲跌幅 | 目前評分 | 觀察信號 | 建議防守停損 | 目標價位 |")
                report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
                for p in near_picks:
                    sigs = "<br>".join(p["signals"])
                    report_lines.append(
                        f"| **{p['symbol']}** | {p['close']} | {p['change_pct']:+0.2f}% | **{p['score']}** | {sigs} | {p['stop_loss']} | {p['target_price']} |"
                    )
        else:
            report_lines.append(
                "| 股票代碼 | 收盤價 | 漲跌幅 | 量化評分 | 觸發技術信號 | 建議防守停損 | 目標價位 | 風險報酬比 |"
            )
            report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for p in picks:
                sigs = "<br>".join(p["signals"])
                report_lines.append(
                    f"| **{p['symbol']}** | {p['close']} | {p['change_pct']:+0.2f}% | **{p['score']}** | {sigs} | {p['stop_loss']} | {p['target_price']} | {p['risk_reward_ratio']} |"
                )

        # 空頭避險閉環區塊
        if market_info.get("score", 50) <= 50 or defense_hedges:
            report_lines.extend(
                [
                    "",
                    "### 🛡️ 空頭避險與轉弱標的監控清單 (Hedging & Defense)",
                    "| 避險/弱勢標的 | 最新收盤 | 漲跌幅 | 避險原因與破線警訊 |",
                    "| :--- | :--- | :--- | :--- |",
                ]
            )
            for d in defense_hedges:
                report_lines.append(f"| **{d['symbol']}** | {d['close']} | {d['change_pct']:+0.2f}% | {d['reason']} |")

        # 區塊 3: 機器學習多因子 Alpha 排名 (讀取已訓練之 taiwan_stock_ranker.joblib，禁止 HTTP request 現場重訓)
        print("[*] 正在載入 ML 多因子預測排序模型...")
        model_saved_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data_persistence",
            "models",
            "taiwan_stock_ranker.joblib",
        )
        ml_rankings = []
        is_fallback = False
        try:
            stock_data_map = {}
            for sym in DEFAULT_WATCHLIST[:8]:
                t = yf.Ticker(sym)
                h = t.history(period="6mo")
                if len(h) >= 30:
                    stock_data_map[sym] = h

            if stock_data_map:
                if os.path.exists(model_saved_path):
                    ranker = TaiwanStockMLRanker.load_model(model_saved_path)
                    rank_df = ranker.predict_rankings(stock_data_map)
                else:
                    is_fallback = True
                    # RULE_BASED fallback
                    rows = []
                    for s, df_s in stock_data_map.items():
                        c = float(df_s["Close"].iloc[-1])
                        ma20 = float(df_s["Close"].rolling(20).mean().iloc[-1])
                        bias = (c - ma20) / ma20 if ma20 > 0 else 0.0
                        delta = df_s["Close"].diff()
                        gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
                        loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
                        rsi = 100.0 - (100.0 / (1.0 + (gain / (loss + 1e-9))))
                        fwd_ret = bias * 0.5
                        rows.append(
                            {
                                "symbol": s,
                                "current_price": c,
                                "predicted_fwd_5d_ret_pct": fwd_ret * 100.0,
                                "bias_ma20": bias * 100.0,
                                "rsi_14": rsi,
                            }
                        )
                    rank_df = (
                        pd.DataFrame(rows)
                        .sort_values("predicted_fwd_5d_ret_pct", ascending=False)
                        .reset_index(drop=True)
                    )
                    rank_df["rank"] = range(1, len(rank_df) + 1)

                badge_str = "【RULE_BASED fallback】" if is_fallback else "【LightGBM ML 預測】"
                report_lines.extend(
                    [
                        "",
                        "---",
                        f"## 🤖 3. AI / ML 多因子未來 5 日超額報酬排序榜 {badge_str}",
                        "| 排名 | 標的 | 最新收盤 | 預測 5 日報酬 (%) | 月線乖離率 (%) | RSI(14) |",
                        "| :---: | :--- | :--- | :---: | :---: | :---: |",
                    ]
                )
                for _, row in rank_df.head(5).iterrows():
                    ml_rankings.append(row.to_dict())
                    report_lines.append(
                        f"| #{int(row['rank'])} | **{row['symbol']}** | {row['current_price']} | **{row['predicted_fwd_5d_ret_pct']:+0.2f}%** | {row['bias_ma20']:+0.2f}% | {row['rsi_14']:.1f} |"
                    )
        except Exception as e:
            report_lines.append(f"\n> ℹ️ ML 多因子預測模型運行略過: {html.escape(str(e), quote=True)}")

        # 區塊 4: 每日覆盤與虧損檢討 (Post-Mortem Analysis)
        post_mortems = self.sync.get_recent_post_mortems(limit=5)
        report_lines.extend(
            [
                "",
                "---",
                "## 🔍 4. 每日自動化覆盤與虧損歸因檢討 (Daily Post-Mortem Loss Analysis)",
            ]
        )
        if not post_mortems:
            report_lines.append("> ℹ️ 尚無足夠平倉 / Post-Mortem 紀錄，無法計算執行紀律率（Insufficient data）。")
        else:
            report_lines.append("| 日期 | 標的 | 進場價 | 出場價 | 虧損金額 | 虧損幅度 | 歸因分析與優化對策 |")
            report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for pm in post_mortems:
                report_lines.append(
                    f"| {pm['date']} | **{pm['symbol']}** | {pm['entry_price']} | {pm['exit_price']} | -${abs(pm['loss_amount']):.2f} | {pm['loss_pct']:+.2f}% | {pm['cause_analysis']} |"
                )

        report_lines.extend(
            [
                "",
                "---",
                "## 🛡️ 5. 紀律與風控指引",
                "1. **大盤濾網原則**：若大盤評分低於 40 分，嚴格控制總多單部位於 30% 以下，切勿重倉追高。",
                "2. **單筆風險限制**：任何單一標的承擔之損失金額不可超過總投資部位之 1.5%~2.0%。",
                "3. **嚴格執行停損**：跌破上述「建議防守停損」無條件離場，保本永遠是獲利的第一前提。",
                "",
                "---",
                "*免責聲明：本報告由量化統計模型自動生成，所提供之數據與資訊僅供學術研究與投資策略參考，不構成任何具體投資邀約或保證獲利承諾。*",
            ]
        )

        report_content = "\n".join(report_lines)

        # 寫入歷史報告存檔
        report_filename = f"morning_report_{today_str}.md"
        report_path = os.path.join(REPORTS_DIR, report_filename)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"[✓] 晨報已成功生成並存檔至: {report_path}")
        return report_content

    def generate_html_report(self, session_type: str = "盤前晨報", cached_market_info: dict = None) -> str:
        """
        生成具備精緻視覺排版、色彩標籤、多空燈號與響應式表格的報告 HTML 視圖
        支援渲染為 HTML/PDF 格式，顯示產出時間與最後交易日
        """
        now_time_str = datetime.now(_TAIPEI_TZ).strftime("%Y-%m-%d %H:%M:%S")

        snap = self._last_snapshot if (self._last_snapshot and cached_market_info is not None) else self.collect_report_snapshot(session_type=session_type, cached_market_info=cached_market_info)
        if snap.status != 'VALID':
            return "<div class='p-4 rounded-xl bg-rose-950/40 border border-rose-500/30 text-rose-300'><strong>REPORT_STATUS: DATA_UNAVAILABLE</strong> — 無法取得有效大盤收盤資料，已中止計算多空分數與選股建議。</div>"
        market_info = snap.market_info
        screener_res = snap.screener_res
        scanned_count = screener_res["scanned_count"]
        qualified_count = screener_res["qualified_count"]
        picks = screener_res["picks"]
        near_picks = screener_res["near_picks"]
        defense_hedges = screener_res["defense_hedges"]

        score = market_info.get("score", 50)
        regime = market_info.get("regime", "震盪整理")
        last_trade_date = (market_info.get("date") or "DATA_UNAVAILABLE")
        ma_pattern = market_info.get("ma_pattern", "均線架構判斷中")

        # 燈號顏色邏輯
        if score >= 70:
            badge_class = "bg-rose-500/20 text-rose-400 border border-rose-500/30"
            light_icon = "🟢 多頭強勢格局"
        elif score <= 40:
            badge_class = "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
            light_icon = "🔴 空頭防守格局"
        else:
            badge_class = "bg-amber-500/20 text-amber-400 border border-amber-500/30"
            light_icon = "🟡 中性震盪盤整"

        html_out = f"""
        <div class="space-y-6 text-slate-100 font-sans print:text-black">
            <!-- 頂部 Header 卡片 -->
            <div class="bg-gradient-to-r from-slate-900 to-indigo-950/80 border border-indigo-500/30 rounded-2xl p-5 shadow-2xl">
                <div class="flex flex-col md:flex-row items-start md:items-center justify-between gap-3">
                    <div>
                        <div class="flex items-center space-x-2">
                            <span class="px-2.5 py-0.5 rounded-full text-xs font-bold {badge_class}">{light_icon}</span>
                            <span class="text-xs text-slate-400 font-mono">📅 最後交易日: {last_trade_date}</span>
                        </div>
                        <h2 class="text-2xl font-black mt-2 text-white tracking-wide">台股量化決策報告【{session_type}】</h2>
                        <p class="text-xs text-slate-400 mt-1">產出時間: {now_time_str} (Asia/Taipei) | Antigravity AI Quantitative Core</p>
                    </div>
                    <div class="text-right bg-slate-950/60 p-3 rounded-xl border border-slate-800">
                        <span class="text-[11px] text-slate-400">大盤多空能量指數</span>
                        <div class="text-2xl font-black text-rose-400">{score} <span class="text-xs text-slate-400">/ 100</span></div>
                    </div>
                </div>
            </div>

            <!-- 大盤關鍵均線與操作方針 -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="bg-slate-900/90 border border-slate-800 p-4 rounded-xl">
                    <span class="text-xs text-slate-400">加權指數最新點位</span>
                    <div class="text-xl font-bold text-white mt-1">{market_info.get("latest_close", "N/A")} <span class="text-xs {"text-rose-400" if market_info.get("change_pct", 0) >= 0 else "text-emerald-400"}">({market_info.get("change_pct", 0):+0.2f}%)</span></div>
                    <div class="text-[11px] text-slate-500 mt-1">MA5: {market_info.get("ma5")} | MA20: {market_info.get("ma20")}</div>
                    <div class="text-[11px] text-indigo-400 mt-0.5">{ma_pattern}</div>
                </div>
                <div class="bg-slate-900/90 border border-slate-800 p-4 rounded-xl">
                    <span class="text-xs text-slate-400">市場狀態判定</span>
                    <div class="text-xl font-bold text-amber-400 mt-1">【{regime}】</div>
                    <div class="text-[11px] text-slate-500 mt-1">20日年化波動率: {market_info.get("volatility_20d", 0)}%</div>
                    <div class="text-[11px] text-emerald-400 mt-0.5">量能: {market_info.get("volume_status", "正常")}</div>
                </div>
                <div class="bg-slate-900/90 border border-slate-800 p-4 rounded-xl">
                    <span class="text-xs text-slate-400">今日操盤方針</span>
                    <div class="text-xs text-slate-200 mt-1.5 leading-relaxed bg-slate-950 p-2 rounded border border-slate-800/80">{market_info.get("suggestion", "順勢而為，控制總部位風險。")}</div>
                </div>
            </div>

            <!-- 主動精選強勢突破標的表格 -->
            <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-4">
                <div class="flex items-center justify-between mb-3">
                    <h3 class="font-bold text-sm text-slate-200 flex items-center space-x-2">
                        <span class="w-2 h-2 rounded-full bg-rose-500 animate-ping"></span>
                        <span>🎯 今日主動量化強勢突破選股池</span>
                    </h3>
                    <span class="text-xs text-slate-400 font-mono">已掃描 {scanned_count} 檔，符合門檻 {qualified_count} 檔</span>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs">
                        <thead class="text-slate-400 border-b border-slate-800 bg-slate-950/50">
                            <tr>
                                <th class="py-2.5 px-3">股票代碼</th>
                                <th class="py-2.5 px-3">最新價</th>
                                <th class="py-2.5 px-3">漲跌幅</th>
                                <th class="py-2.5 px-3">量化評分</th>
                                <th class="py-2.5 px-3">觸發信號</th>
                                <th class="py-2.5 px-3">防守停損</th>
                                <th class="py-2.5 px-3">目標價位</th>
                                <th class="py-2.5 px-3">風報比</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-800/60 font-mono">
        """

        if not picks:
            html_out += """
                            <tr>
                                <td colspan="8" class="text-center py-4 text-amber-400 font-sans bg-amber-950/20">
                                    ⚠️ 今日篩選池中無標的完全符合突破門檻 (分數 &ge; 50 分)，系統啟動防守保護機制！以下為最接近門檻標的：
                                </td>
                            </tr>
            """
            for p in near_picks:
                change_color = "text-rose-400 font-bold" if p["change_pct"] >= 0 else "text-emerald-400 font-bold"
                sigs = " ".join(
                    [
                        f"<span class='px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 text-[10px] mr-1'>{s}</span>"
                        for s in p["signals"]
                    ]
                )
                html_out += f"""
                            <tr class="hover:bg-slate-800/40 transition">
                                <td class="py-2.5 px-3 font-bold text-white font-sans">{p["symbol"]} <span class="text-[10px] text-amber-400">(觀察中)</span></td>
                                <td class="py-2.5 px-3 text-slate-200">${p["close"]}</td>
                                <td class="py-2.5 px-3 {change_color}">{p["change_pct"]:+0.2f}%</td>
                                <td class="py-2.5 px-3"><span class="px-2 py-0.5 rounded-full bg-slate-800 text-amber-400 font-bold">{p["score"]}分</span></td>
                                <td class="py-2.5 px-3 font-sans">{sigs}</td>
                                <td class="py-2.5 px-3 text-emerald-400">${p["stop_loss"]}</td>
                                <td class="py-2.5 px-3 text-amber-300">${p["target_price"]}</td>
                                <td class="py-2.5 px-3 text-slate-400 font-bold">{p["risk_reward_ratio"]}</td>
                            </tr>
                """
        else:
            for p in picks:
                change_color = "text-rose-400 font-bold" if p["change_pct"] >= 0 else "text-emerald-400 font-bold"
                sigs = " ".join(
                    [
                        f"<span class='px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 text-[10px] mr-1'>{s}</span>"
                        for s in p["signals"]
                    ]
                )
                html_out += f"""
                            <tr class="hover:bg-slate-800/40 transition">
                                <td class="py-2.5 px-3 font-bold text-white font-sans">{p["symbol"]}</td>
                                <td class="py-2.5 px-3 text-slate-200">${p["close"]}</td>
                                <td class="py-2.5 px-3 {change_color}">{p["change_pct"]:+0.2f}%</td>
                                <td class="py-2.5 px-3"><span class="px-2 py-0.5 rounded-full bg-rose-500/20 text-rose-300 font-bold">{p["score"]}分</span></td>
                                <td class="py-2.5 px-3 font-sans">{sigs}</td>
                                <td class="py-2.5 px-3 text-emerald-400">${p["stop_loss"]}</td>
                                <td class="py-2.5 px-3 text-amber-300">${p["target_price"]}</td>
                                <td class="py-2.5 px-3 text-indigo-300 font-bold">{p["risk_reward_ratio"]}</td>
                            </tr>
                """

        # 空頭防禦避險標的表格
        if defense_hedges:
            html_out += """
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- 空頭防禦與轉弱標的表格 -->
            <div class="bg-slate-900/90 border border-rose-500/30 rounded-xl p-4">
                <div class="flex items-center justify-between mb-3">
                    <h3 class="font-bold text-sm text-rose-300 flex items-center space-x-2">
                        <i class="fa-solid fa-shield-halved text-rose-400"></i>
                        <span>🛡️ 空頭避險與破線轉弱標的監控清單 (Hedging & Defense)</span>
                    </h3>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs">
                        <thead class="text-slate-400 border-b border-slate-800 bg-slate-950/50">
                            <tr>
                                <th class="py-2.5 px-3">避險 / 弱勢標的</th>
                                <th class="py-2.5 px-3">最新價</th>
                                <th class="py-2.5 px-3">漲跌幅</th>
                                <th class="py-2.5 px-3">避險原因與破線警訊</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-800/60 font-mono">
            """
            for d in defense_hedges:
                html_out += f"""
                            <tr class="hover:bg-slate-800/40 transition">
                                <td class="py-2.5 px-3 font-bold text-amber-300 font-sans">{d["symbol"]}</td>
                                <td class="py-2.5 px-3 text-slate-200">${d["close"]}</td>
                                <td class="py-2.5 px-3 {"text-rose-400" if d["change_pct"] >= 0 else "text-emerald-400"} font-bold">{d["change_pct"]:+0.2f}%</td>
                                <td class="py-2.5 px-3 text-slate-300 font-sans">{d["reason"]}</td>
                            </tr>
                """
            html_out += """
                        </tbody>
                    </table>
                </div>
            </div>
            """

        # 區塊 3: ML 多因子排序榜 (HTML 視圖)
        model_saved_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data_persistence",
            "models",
            "taiwan_stock_ranker.joblib",
        )
        is_html_fallback = False
        try:
            stock_data_map = {}
            for sym in DEFAULT_WATCHLIST[:8]:
                t = yf.Ticker(sym)
                h = t.history(period="6mo")
                if len(h) >= 30:
                    stock_data_map[sym] = h

            if stock_data_map:
                if os.path.exists(model_saved_path):
                    ranker = TaiwanStockMLRanker.load_model(model_saved_path)
                    rank_df = ranker.predict_rankings(stock_data_map)
                else:
                    is_html_fallback = True
                    rows = []
                    for s, df_s in stock_data_map.items():
                        c = float(df_s["Close"].iloc[-1])
                        ma20 = float(df_s["Close"].rolling(20).mean().iloc[-1])
                        bias = (c - ma20) / ma20 if ma20 > 0 else 0.0
                        delta = df_s["Close"].diff()
                        gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
                        loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
                        rsi = 100.0 - (100.0 / (1.0 + (gain / (loss + 1e-9))))
                        fwd_ret = bias * 0.5
                        rows.append(
                            {
                                "symbol": s,
                                "current_price": c,
                                "predicted_fwd_5d_ret_pct": fwd_ret * 100.0,
                                "bias_ma20": bias * 100.0,
                                "rsi_14": rsi,
                            }
                        )
                    rank_df = (
                        pd.DataFrame(rows)
                        .sort_values("predicted_fwd_5d_ret_pct", ascending=False)
                        .reset_index(drop=True)
                    )
                    rank_df["rank"] = range(1, len(rank_df) + 1)

                badge_tag = (
                    '<span class="text-xs px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 font-bold">RULE_BASED fallback</span>'
                    if is_html_fallback
                    else '<span class="text-xs px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-bold">LightGBM ML Verified</span>'
                )
                html_out += f"""
                <!-- 第 3 區塊：ML 多因子排序榜 -->
                <div class="bg-slate-900/90 border border-indigo-500/30 rounded-xl p-4">
                    <div class="flex items-center justify-between mb-3">
                        <h3 class="font-bold text-sm text-indigo-300 flex items-center space-x-2">
                            <i class="fa-solid fa-robot text-indigo-400"></i>
                            <span>🤖 3. AI / ML 多因子未來 5 日超額報酬排序榜</span>
                        </h3>
                        {badge_tag}
                    </div>
                    <div class="overflow-x-auto">
                        <table class="w-full text-left text-xs">
                            <thead class="text-slate-400 border-b border-slate-800 bg-slate-950/50">
                                <tr>
                                    <th class="py-2.5 px-3">排名</th>
                                    <th class="py-2.5 px-3">標的代碼</th>
                                    <th class="py-2.5 px-3">最新股價</th>
                                    <th class="py-2.5 px-3">預測 5 日報酬</th>
                                    <th class="py-2.5 px-3">月線乖離率</th>
                                    <th class="py-2.5 px-3">RSI(14)</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-slate-800/60 font-mono">
                """
                for _, r in rank_df.head(5).iterrows():
                    ret_val = float(r["predicted_fwd_5d_ret_pct"])
                    ret_color = "text-rose-400" if ret_val >= 0 else "text-emerald-400"
                    html_out += f"""
                                <tr class="hover:bg-slate-800/40 transition">
                                    <td class="py-2.5 px-3 font-bold text-indigo-400 font-sans">#{int(r["rank"])}</td>
                                    <td class="py-2.5 px-3 font-bold text-white font-sans">{r["symbol"]}</td>
                                    <td class="py-2.5 px-3 text-slate-200">${r["current_price"]}</td>
                                    <td class="py-2.5 px-3 {ret_color} font-bold">{ret_val:+0.2f}%</td>
                                    <td class="py-2.5 px-3 text-slate-300">{float(r["bias_ma20"]):+0.2f}%</td>
                                    <td class="py-2.5 px-3 text-slate-300">{float(r["rsi_14"]):.1f}</td>
                                </tr>
                    """
                html_out += """
                            </tbody>
                        </table>
                    </div>
                </div>
                """
        except Exception as e:
            html_out += f"""
            <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-4 text-xs text-slate-500">
                ML 多因子排序榜載入略過: {html.escape(str(e), quote=True)}
            </div>
            """

        post_mortems = self.sync.get_recent_post_mortems(limit=5)
        html_out += """
            <!-- 覆盤與虧損檢討區塊 -->
            <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-4">
                <div class="flex items-center justify-between mb-3">
                    <h3 class="font-bold text-sm text-slate-200 flex items-center space-x-2">
                        <span class="w-2 h-2 rounded-full bg-amber-500"></span>
                        <span>🔍 每日自動化覆盤與虧損歸因檢討 (Post-Mortem Loss Analysis)</span>
                    </h3>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs">
                        <thead class="text-slate-400 border-b border-slate-800 bg-slate-950/50">
                            <tr>
                                <th class="py-2.5 px-3">日期</th>
                                <th class="py-2.5 px-3">標的</th>
                                <th class="py-2.5 px-3">進場價</th>
                                <th class="py-2.5 px-3">出場價</th>
                                <th class="py-2.5 px-3">虧損金額</th>
                                <th class="py-2.5 px-3">虧損幅度</th>
                                <th class="py-2.5 px-3">歸因分析與優化對策</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-800/60 font-mono">
        """
        if not post_mortems:
            html_out += """
                            <tr>
                                <td colspan="7" class="text-center py-4 text-slate-500 font-sans">近期無停損或虧損事件，系統執行紀律良好。</td>
                            </tr>
            """
        else:
            for pm in post_mortems:
                html_out += f"""
                            <tr class="hover:bg-slate-800/40 transition">
                                <td class="py-2.5 px-3 text-slate-300 font-sans">{pm["date"]}</td>
                                <td class="py-2.5 px-3 font-bold text-white font-sans">{pm["symbol"]}</td>
                                <td class="py-2.5 px-3 text-slate-200">${pm["entry_price"]}</td>
                                <td class="py-2.5 px-3 text-slate-200">${pm["exit_price"]}</td>
                                <td class="py-2.5 px-3 text-emerald-400 font-bold">-${abs(pm["loss_amount"]):.2f}</td>
                                <td class="py-2.5 px-3 text-emerald-400 font-bold">{pm["loss_pct"]:+.2f}%</td>
                                <td class="py-2.5 px-3 text-slate-300 font-sans">{pm["cause_analysis"]}</td>
                            </tr>
                """
        html_out += """
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        """
        return html_out


if __name__ == "__main__":
    generator = ReportGenerator()
    content = generator.generate()
    print("\n" + "=" * 50)
    print(content)
