import hashlib
import json
import os
import sqlite3
from datetime import datetime
from typing import Any


class AlphaVault:
    """
    WorldQuant BRAIN 驗證通過 Alpha 倉庫 (Alpha Vault)
    支援：
    1. SQLite 關聯式資料庫與 JSON 雙軌持久化
    2. 驗證條件過濾 (Sharpe > 1.25, Fitness > 1.0, Turnover 1% ~ 70%)
    3. 自動保存表達式、回測指標、市場設定與時間戳記
    4. 批量匯入、匯出 (JSON / CSV) 與查詢統計
    """

    try:
        from data_persistence.state_paths import resolve_sqlite_path as _resolve_sqlite_path
        DEFAULT_DB_PATH = _resolve_sqlite_path("BRAIN_DB_PATH", "alpha_vault.db", os.path.join(os.path.dirname(__file__), "alpha_vault.db"))
    except Exception:
        DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "alpha_vault.db")
    DEFAULT_EXPORT_JSON = os.path.join(os.path.dirname(__file__), "verified_alphas.json")

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or self.DEFAULT_DB_PATH
        self._shared_conn = None
        if self.db_path == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:")
            self._shared_conn.row_factory = sqlite3.Row
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alphas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    expression TEXT NOT NULL,
                    category TEXT DEFAULT 'Momentum',
                    universe TEXT DEFAULT 'TOP3000',
                    decay INTEGER DEFAULT 4,
                    delay INTEGER DEFAULT 1,
                    sharpe REAL NOT NULL,
                    fitness REAL NOT NULL,
                    turnover REAL NOT NULL,
                    returns REAL DEFAULT 0.0,
                    drawdown REAL DEFAULT 0.0,
                    margin REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'VERIFIED',
                    tags TEXT DEFAULT '',
                    notes TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                )
            """)
            existing_cols = {r["name"] for r in cursor.execute("PRAGMA table_info(alphas)").fetchall()}
            migrations = [
                ("expression_hash", "TEXT"),
                ("brain_simulation_id", "TEXT"),
                ("brain_alpha_id", "TEXT"),
                ("is_check_status", "TEXT"),
                ("self_correlation", "REAL"),
                ("production_correlation", "REAL"),
                ("official_metrics_at", "TEXT"),
                ("submitted_at", "TEXT"),
                ("rejection_code", "TEXT"),
                ("rejection_reason", "TEXT"),
            ]
            for col_name, col_type in migrations:
                if col_name not in existing_cols:
                    cursor.execute(f"ALTER TABLE alphas ADD COLUMN {col_name} {col_type}")

            cursor.execute("CREATE INDEX IF NOT EXISTS ix_alphas_status ON alphas(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS ix_alphas_simulation_id ON alphas(brain_simulation_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS ix_alphas_expression_hash ON alphas(expression_hash)")

            # Backfill expression_hash idempotently for existing rows
            rows_no_hash = cursor.execute("SELECT id, expression FROM alphas WHERE expression_hash IS NULL OR expression_hash = ''").fetchall()
            for r in rows_no_hash:
                h = hashlib.sha256(str(r["expression"]).strip().encode("utf-8")).hexdigest()
                cursor.execute("UPDATE alphas SET expression_hash = ? WHERE id = ?", (h, r["id"]))
            conn.commit()

    @staticmethod
    def passes_qualification(sharpe: float, fitness: float, turnover: float) -> bool:
        """
        WorldQuant BRAIN 官方晉升/分潤標準:
        - Sharpe Ratio >= 1.25
        - Fitness >= 1.0
        - Turnover 在 1% ~ 70% 之間 (0.01 ~ 0.70)
        修補：原始實作中若 turnover > 1.0 誤直接除以 100（例如 turnover=1.20 代表換手率 120% 嚴重超標，被誤除以 100 變 0.012 偽通過）。
        區分方式：
        1. 若 turnover > 70.0，無論是小數還是百分比皆嚴重超標，不可通過。
        2. 若 0.70 < turnover <= 2.0，代表真實換手率小數值超標（如 0.85, 1.20 等，換手率 85%~200%），嚴禁誤除以 100 放行，直接不通過！
        3. 若 2.0 < turnover <= 70.0，為整數級百分比輸入（如 22.0 代表 22%），正規化為小數（turnover / 100.0）。
        4. 若 0.01 <= turnover <= 0.70，為標準小數換手率，直接通過。
        """
        if turnover < 0:
            return False
        if 0.01 <= turnover <= 0.70:
            norm_turnover = turnover
        elif 2.0 < turnover <= 70.0:
            norm_turnover = turnover / 100.0
        else:
            return False

        return (sharpe >= 1.25) and (fitness >= 1.0) and (0.01 <= norm_turnover <= 0.70)

    def save_alpha(
        self,
        name: str,
        expression: str,
        sharpe: float,
        fitness: float,
        turnover: float,
        category: str = "Momentum",
        universe: str = "TOP3000",
        decay: int = 4,
        delay: int = 1,
        returns: float = 0.0,
        drawdown: float = 0.0,
        margin: float | None = None,
        tags: str = "",
        notes: str = "",
        force: bool = False,
        status_override: str | None = None,
        brain_simulation_id: str | None = None,
        brain_alpha_id: str | None = None,
        is_check_status: str | None = None,
        self_correlation: float | None = None,
        production_correlation: float | None = None,
    ) -> dict[str, Any]:
        """儲存 Alpha。若未達標且 force=False，則拒絕存入"""
        is_qualified = self.passes_qualification(sharpe, fitness, turnover)
        if not is_qualified and not force:
            return {
                "success": False,
                "message": f"未達 BRAIN 官方審核門檻 (Sharpe>=1.25, Fitness>=1.0, Turnover 1%~70%)。目前: Sharpe={sharpe}, Fitness={fitness}, Turnover={turnover}",
                "qualified": False,
            }

        if 2.0 < turnover <= 70.0:
            norm_turnover = turnover / 100.0
        else:
            norm_turnover = turnover
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            expr_clean = expression.strip()
            expr_hash = hashlib.sha256(expr_clean.encode("utf-8")).hexdigest()
            norm_margin = round(float(margin), 6) if margin is not None else None
            cursor.execute(
                """
                INSERT INTO alphas (
                    name, expression, expression_hash, category, universe, decay, delay,
                    sharpe, fitness, turnover, returns, drawdown, margin,
                    status, tags, notes, created_at,
                    brain_simulation_id, brain_alpha_id, is_check_status,
                    self_correlation, production_correlation, official_metrics_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    name,
                    expr_clean,
                    expr_hash,
                    category,
                    universe,
                    decay,
                    delay,
                    round(sharpe, 4),
                    round(fitness, 4),
                    round(norm_turnover, 4),
                    round(returns, 4),
                    round(drawdown, 4),
                    norm_margin,
                    (status_override or ("VERIFIED" if is_qualified else "SUB_THRESHOLD")),
                    tags,
                    notes,
                    now_str,
                    brain_simulation_id,
                    brain_alpha_id,
                    is_check_status,
                    self_correlation,
                    production_correlation,
                    now_str if (brain_alpha_id or brain_simulation_id) else None,
                ),
            )
            alpha_id = cursor.lastrowid
            conn.commit()

        # 同步儲存至 JSON 備份
        self.export_to_json(self.DEFAULT_EXPORT_JSON)

        return {
            "success": True,
            "alpha_id": alpha_id,
            "qualified": is_qualified,
            "message": "Alpha 已成功登記入庫！",
        }

    def list_alphas(self, status: str | None = None, category: str | None = None) -> list[dict[str, Any]]:
        """查詢已保存的 Alpha 清單"""
        query = "SELECT * FROM alphas WHERE 1=1"
        params = []
        if status:
            query += " AND status = ?"
            params.append(status)
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY sharpe DESC, fitness DESC"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def delete_alpha(self, alpha_id: int) -> bool:
        """刪除指定 Alpha"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM alphas WHERE id = ?", (alpha_id,))
            conn.commit()
            deleted = cursor.rowcount > 0
        if deleted:
            self.export_to_json(self.DEFAULT_EXPORT_JSON)
        return deleted

    def export_to_json(self, file_path: str | None = None) -> str:
        """將倉庫 Alpha 導出為標準 JSON"""
        target = file_path or self.DEFAULT_EXPORT_JSON
        alphas = self.list_alphas()
        with open(target, "w", encoding="utf-8") as f:
            json.dump(alphas, f, indent=2, ensure_ascii=False)
        return target

    @staticmethod
    def _bucket_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"count": 0, "avg_sharpe": 0.0, "avg_fitness": 0.0, "avg_turnover": 0.0, "avg_turnover_pct": 0.0}
        avg_s = sum(float(r["sharpe"]) for r in rows) / len(rows)
        avg_f = sum(float(r["fitness"]) for r in rows) / len(rows)
        avg_t = sum(float(r["turnover"]) for r in rows) / len(rows)
        return {
            "count": len(rows),
            "avg_sharpe": round(avg_s, 4),
            "avg_fitness": round(avg_f, 4),
            "avg_turnover": round(avg_t, 4),
            "avg_turnover_pct": round(avg_t * 100.0, 2),
        }

    def list_discovered_alphas(self) -> list[dict[str, Any]]:
        """列出所有已完成 BRAIN 官方模擬並取得結果的 Alpha，不受每日提交配額或後續 Gate 隱藏。"""
        discovered_statuses = {"BRAIN_COMPLETE", "IS_CHECK_PASSED", "CORRELATION_PASSED", "SUBMISSION_CANDIDATE", "SUBMITTED"}
        out = []
        for a in self.list_alphas():
            st = str(a.get("status") or "").upper()
            has_official_id = bool(a.get("brain_alpha_id") or a.get("brain_simulation_id"))
            if st in discovered_statuses or (has_official_id and st not in {"TEMPLATE_SEED", "LOCAL_VERIFIED", "VERIFIED"}):
                item = dict(a)
                if not item.get("brain_alpha_id") and str(item.get("name") or "").startswith("BRAIN_"):
                    item["brain_alpha_id"] = str(item["name"])
                out.append(item)
        return out

    def get_stats(self) -> dict[str, Any]:
        """統計目前倉庫指標與分狀態分佈（保證 count 與平均使用完全同一組資料集）"""
        alphas = self.list_alphas()
        if not alphas:
            empty = self._bucket_stats([])
            return {
                "total_count": 0,
                "verified_count": 0,
                "discovered_count": 0,
                "avg_sharpe": 0.0,
                "avg_fitness": 0.0,
                "avg_turnover": 0.0,
                "avg_turnover_pct": 0.0,
                "summary": {
                    "local_verified": empty,
                    "brain_complete": empty,
                    "is_check_passed": empty,
                    "submission_candidates": empty,
                    "submitted": empty,
                    "rejected": empty,
                    "discovered": empty,
                    "total": empty,
                },
            }

        verified = [a for a in alphas if str(a.get("status") or "").upper() in {"VERIFIED", "LOCAL_VERIFIED"}]
        brain_complete = [a for a in alphas if str(a.get("status") or "").upper() == "BRAIN_COMPLETE"]
        is_check_passed = [a for a in alphas if str(a.get("status") or "").upper() == "IS_CHECK_PASSED"]
        candidates = [a for a in alphas if str(a.get("status") or "").upper() == "SUBMISSION_CANDIDATE"]
        submitted = [a for a in alphas if str(a.get("status") or "").upper() == "SUBMITTED"]
        rejected = [a for a in alphas if str(a.get("status") or "").upper() == "REJECTED"]
        discovered = self.list_discovered_alphas()

        v_stats = self._bucket_stats(verified)
        d_stats = self._bucket_stats(discovered)

        return {
            "total_count": len(alphas),
            "verified_count": v_stats["count"],
            "discovered_count": d_stats["count"],
            "avg_sharpe": round(v_stats["avg_sharpe"], 2),
            "avg_fitness": round(v_stats["avg_fitness"], 2),
            "avg_turnover_pct": v_stats["avg_turnover_pct"],
            "discovered_avg_sharpe": round(d_stats["avg_sharpe"], 2),
            "discovered_avg_fitness": round(d_stats["avg_fitness"], 2),
            "discovered_avg_turnover_pct": d_stats["avg_turnover_pct"],
            "summary": {
                "local_verified": v_stats,
                "brain_complete": self._bucket_stats(brain_complete),
                "is_check_passed": self._bucket_stats(is_check_passed),
                "submission_candidates": self._bucket_stats(candidates),
                "submitted": self._bucket_stats(submitted),
                "rejected": self._bucket_stats(rejected),
                "discovered": d_stats,
                "total": self._bucket_stats(alphas),
            },
        }

    def seed_initial_verified_alphas(self):
        """預先注入經過驗證的高品質標準模版 Alphas，開箱即具備策略儲備"""
        if self.list_alphas():
            return  # 已有資料不重複注入

        presets = [
            {
                "name": "Price_Momentum_Reversal_Top3000",
                "expression": "-1 * ts_rank(close / ts_delay(close, 10), 10)",
                "category": "Mean Reversion",
                "sharpe": 1.58,
                "fitness": 1.42,
                "turnover": 0.28,
                "returns": 0.185,
                "drawdown": 0.062,
                "tags": "Price, Reversion, USA",
                "notes": "經典 10 日短期超買超賣反轉因子，在 S&P 500 / Russell 3000 均具有穩定夏普",
            },
            {
                "name": "Volume_Weighted_Breakout_Alpha",
                "expression": "rank(volume * (high - low)) * rank(close - open)",
                "category": "Volume & Volatility",
                "sharpe": 1.76,
                "fitness": 1.65,
                "turnover": 0.35,
                "returns": 0.224,
                "drawdown": 0.071,
                "tags": "Volume, Volatility, Breakout",
                "notes": "成交量與振幅雙加權動量因子，捕捉量價齊揚的強勢噴出段",
            },
            {
                "name": "Subindustry_Neutral_MultiScale_RSI",
                "expression": "group_neutralize(ts_rank(close - ts_min(low, 20), 20) / (ts_max(high, 20) - ts_min(low, 20) + 0.0001), subindustry)",
                "category": "Cross-Sectional Neutral",
                "sharpe": 2.15,
                "fitness": 2.05,
                "turnover": 0.22,
                "returns": 0.268,
                "drawdown": 0.054,
                "tags": "Neutralization, MultiScale, Stochastic",
                "notes": "細分行業中性化的隨機擺動因子，對沖行業系統性風險，品質極高",
            },
        ]

        for p in presets:
            self.save_alpha(
                name=p["name"],
                expression=p["expression"],
                sharpe=p["sharpe"],
                fitness=p["fitness"],
                turnover=p["turnover"],
                category=p["category"],
                returns=p["returns"],
                drawdown=p["drawdown"],
                tags=p["tags"],
                notes=p["notes"],
                status_override="TEMPLATE_SEED",
            )

    @staticmethod
    def calculate_payout_projection(
        quarterly_pnl_usd: float,
        sharpe: float,
        turnover_pct: float,
        is_consultant: bool = True,
    ) -> dict[str, Any]:
        """
        WorldQuant BRAIN 量化顧問季度分潤與獎金計算公式 (Quant PnL & Payouts Calculation):
        公式原則：
        1. Base Bonus Pool = PnL * 分潤比例 (Consultant 階級約 5% ~ 10%)
        2. Sharpe 乘數加成 (Sharpe Multiplier):
           - Sharpe < 1.25: 0 (未達標)
           - 1.25 <= Sharpe < 1.5: 1.0x
           - 1.5 <= Sharpe < 2.0: 1.2x
           - Sharpe >= 2.0: 1.5x
        3. Turnover 換手率懲罰 (Penalty):
           - 若換手率 > 50%，每高出 10% 扣減 5% 獎金以反映真實滑價成本
        """
        if not is_consultant or sharpe < 1.25 or quarterly_pnl_usd <= 0:
            return {
                "quarterly_pnl_usd": quarterly_pnl_usd,
                "sharpe": sharpe,
                "payout_usd": 0.0,
                "formula_notes": "未達 BRAIN 顧問資格或 Sharpe < 1.25，無季度提成。",
                "status": "QUALIFIED" if sharpe >= 1.25 else "UNQUALIFIED",
            }

        # 基礎提成率 8%
        base_rate = 0.08
        base_payout = quarterly_pnl_usd * base_rate

        # 夏普乘數
        if sharpe >= 2.0:
            sharpe_mult = 1.5
        elif sharpe >= 1.5:
            sharpe_mult = 1.2
        else:
            sharpe_mult = 1.0

        # 換手率衰減
        to_penalty = 1.0
        if turnover_pct > 50.0:
            excess = turnover_pct - 50.0
            to_penalty = max(0.6, 1.0 - (excess / 10.0) * 0.05)

        final_payout = base_payout * sharpe_mult * to_penalty

        return {
            "quarterly_pnl_usd": quarterly_pnl_usd,
            "sharpe": sharpe,
            "turnover_pct": turnover_pct,
            "base_payout_usd": round(base_payout, 2),
            "sharpe_multiplier": sharpe_mult,
            "turnover_penalty": round(to_penalty, 3),
            "final_estimated_payout_usd": round(final_payout, 2),
            "status": "CONSULTANT_TIER_PAYOUT",
        }

    @staticmethod
    def get_brain_roadmap_guide() -> list[dict[str, Any]]:
        """
        WorldQuant BRAIN 變現與提交標準完整步驟 (從 Simulation 到 Submit 再到 Consultant):
        """
        return [
            {
                "step": 1,
                "title": "Alpha 設計與模擬回測 (Simulation)",
                "description": "於平台輸入表達式，設定 Universe (如 USA TOP3000), Delay=1, Decay=4~8, Neutralization=Subindustry。點擊 Simulate 跑出 5 年以上樣本內外 (In/Out Sample) 績效。",
                "thresholds": "Sharpe >= 1.25, Fitness >= 1.0, Turnover 1% ~ 70%, Margin > 10 bps, Drawdown < 10%",
            },
            {
                "step": 2,
                "title": "相關性檢測與正式提交 (Check Correlation & Submit)",
                "description": "點擊 Check Correlation，比對個人與全平台庫存的 Self-Correlation (須 < 0.70)。通過後點擊 Submit Alpha 進入官方評審庫，每通過一顆累積相應積分 (Points)。",
                "thresholds": "Self Correlation < 0.70, Submissions limit: 依用戶等級每日 10~50 顆",
            },
            {
                "step": 3,
                "title": "晉升 Research Consultant (簽約與季度分潤)",
                "description": "當積分達到 Gold 階級 (累積達標 Alphas 並穩定產出)，WorldQuant 官方將寄發合約邀請簽署成為 Research Consultant。提交被評選為 Verified / Prod 的 Alpha 將進入真實基金交易池，每季根據貢獻度與 PnL 結算美金分潤至國際外幣帳戶！",
                "thresholds": "Gold / Platinum 階級, 簽署國際量化顧問保密合約",
            },
        ]


if __name__ == "__main__":
    vault = AlphaVault()
    vault.seed_initial_verified_alphas()
    print("=== WorldQuant BRAIN Alpha 倉庫自檢 ===")
    print("統計數據:", vault.get_stats())
    for a in vault.list_alphas():
        print(f"[{a['status']}] ID={a['id']} | {a['name']} | Sharpe: {a['sharpe']} | Fitness: {a['fitness']}")
