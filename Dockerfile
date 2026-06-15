FROM python:3.10-slim

WORKDIR /app

# 安裝套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製所有程式碼
COPY . .

# 📢 開放 Docker 的 8000 連線埠
EXPOSE 8000

# 📢 啟動命令：改用 chainlit 啟動網頁，並允許外部連線
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000", "--headless"]