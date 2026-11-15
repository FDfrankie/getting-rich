import os
import sys

# 確保根目錄可被載入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_ui.server import run_server

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8999))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"啟動全方位量化操盤 Web APP: http://{host}:{port}")
    run_server(port=port, host=host)
