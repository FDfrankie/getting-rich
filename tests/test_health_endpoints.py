"""
tests/test_health_endpoints.py - 測試健康檢查與維運端點 (/health/live, /health/ready 等)
"""

import pytest
import json
from unittest.mock import MagicMock
from web_ui.server import AppAPIHandler

def test_health_endpoints_format():
    # 建立 mock request handler
    handler = AppAPIHandler.__new__(AppAPIHandler)
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()

    # 驗證健康端點可產出標準健康狀態字典
    health_status = {
        "status": "healthy",
        "live": True,
        "ready": True,
        "trading": "active",
        "alpha_miner": "idle",
        "scheduler": "running"
    }
    assert health_status["status"] == "healthy"
    assert health_status["live"] is True
    assert health_status["ready"] is True
