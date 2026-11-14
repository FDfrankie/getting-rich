import os
from typing import List

# 台股大盤與標的監控清單 (Yahoo Finance 代碼格式)
MARKET_INDEX_SYMBOL = "^TWII"  # 加權指數

# 預設自選監控池 (涵蓋半導體、AI伺服器、蘋概股、電子權值與人氣熱門股)
DEFAULT_WATCHLIST: List[str] = [
    "2330.TW",  # 台積電
    "2454.TW",  # 聯發科
    "2317.TW",  # 鴻海
    "2382.TW",  # 廣達
    "3231.TW",  # 緯創
    "2308.TW",  # 台達電
    "2603.TW",  # 長榮
    "2609.TW",  # 陽明
    "3008.TW",  # 大立光
    "3443.TW",  # 創意
    "3661.TW",  # 世芯-KY
    "2356.TW",  # 英業達
    "6669.TW",  # 緯穎
    "2376.TW",  # 技嘉
    "2002.TW",  # 中鋼
]

# 選股技術參數
PARAMS = {
    "short_ma": 5,
    "mid_ma": 20,
    "long_ma": 60,
    "volume_surge_multiplier": 1.5,  # 成交量放大倍數 (相比5日均量)
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "kd_period": 9,
    "bbands_period": 20,
    "bbands_std": 2.0,
    "atr_period": 14,
    "atr_stop_loss_multiplier": 2.0  # ATR 停損距離乘數
}

# 每日排程時間 (24小時制)
REPORT_SCHEDULE_TIME = "08:00"

# 報告儲存目錄
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# 外部推播設定 (可填入 LINE Notify Token 或 Discord Webhook URL)
WEBHOOK_SETTINGS = {
    "enable_push": False,
    "line_notify_token": os.getenv("LINE_NOTIFY_TOKEN", ""),
    "discord_webhook_url": os.getenv("DISCORD_WEBHOOK_URL", "")
}
