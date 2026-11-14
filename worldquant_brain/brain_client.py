import os
import json
import requests
from typing import Dict, Any, Optional

class BrainClient:
    """
    WorldQuant BRAIN 官方 API 互動客戶端
    可用於：
    1. 驗證帳號認證狀態 (Authentication)
    2. 自動化發送 Alpha 表達式進行雲端回測 (Simulations)
    3. 查詢回測進度與 Sharpe / Fitness 指標
    """
    BASE_URL = "https://api.worldquantbrain.com"

    def __init__(self, email: Optional[str] = None, password: Optional[str] = None):
        self.email = email or os.getenv("BRAIN_EMAIL", "")
        self.password = password or os.getenv("BRAIN_PASSWORD", "")
        self.session = requests.Session()
        self.is_authenticated = False

    def authenticate(self) -> bool:
        """登入 BRAIN 平台取得 Session Cookie"""
        if not self.email or not self.password:
            print("[i] 未設定 BRAIN_EMAIL 或 BRAIN_PASSWORD 環境變數，將以離線/研究模式運行。")
            return False

        try:
            resp = self.session.post(
                f"{self.BASE_URL}/authentication",
                auth=(self.email, self.password)
            )
            if resp.status_code == 201:
                print(f"[✓] WorldQuant BRAIN 帳號認證成功 ({self.email})")
                self.is_authenticated = True
                return True
            else:
                print(f"[!] 認證失敗: 狀態碼 {resp.status_code}, 回應: {resp.text}")
                return False
        except Exception as e:
            print(f"[!] 連線失敗: {e}")
            return False

    def simulate_alpha(self, expression: str, universe: str = "TOP3000", delay: int = 1, decay: int = 4) -> Dict[str, Any]:
        """提交 Alpha 表達式至 BRAIN 伺服器進行雲端回測"""
        payload = {
            "type": "REGULAR",
            "settings": {
                "instrumentType": "EQUITY",
                "region": "USA",
                "universe": universe,
                "delay": delay,
                "decay": decay,
                "neutralization": "SUBINDUSTRY",
                "truncation": 0.08,
                "pasteurization": "ON",
                "unitHandling": "VERIFY",
                "nanHandling": "OFF",
                "language": "FASTEXPR",
                "visualization": False
            },
            "regular": expression
        }

        if not self.is_authenticated:
            # 離線展示回傳擬真結構
            return {
                "status": "OFFLINE_SIMULATION_READY",
                "alpha_expression": expression,
                "settings": payload["settings"],
                "notice": "請配置 BRAIN_EMAIL 及 BRAIN_PASSWORD 即可一鍵送往 WorldQuant 雲端運算叢集進行十數年歷史回測！"
            }

        resp = self.session.post(f"{self.BASE_URL}/simulations", json=payload)
        return resp.json()

if __name__ == "__main__":
    client = BrainClient()
    test_expr = "rank(ts_argmax(high, 20)) - rank(ts_argmin(low, 20))"
    res = client.simulate_alpha(test_expr)
    print("=== WorldQuant BRAIN 提交客戶端測試 ===")
    print(json.dumps(res, indent=2, ensure_ascii=False))
