# 使用官方輕量 Python 基礎映像檔
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 設定環境變數 (不緩存位元組碼、無緩衝輸出)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV HOST=0.0.0.0

# 安裝基本系統工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 複製依賴檔並安裝 Python 套件
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 複製整個專案代碼
COPY . /app/

# 開放 Web UI 端口
EXPOSE 8000

# 啟動命令 (以 web_ui/app.py 啟動服務)
CMD ["python", "web_ui/app.py"]
