"""
YouTube OAuth2 初始化腳本
========================
執行一次，將 token 存到 /app/secrets/yt_token.json。
之後 upload_to_youtube 工具會自動使用並刷新 token。

使用方式（在本機，不是容器內）：
  python scripts/auth_youtube.py

前置條件：
  1. 在 Google Cloud Console 建立專案並啟用 YouTube Data API v3
  2. 建立 OAuth 2.0 用戶端憑證（類型選「桌面應用程式」）
  3. 下載 client_secrets.json 放到 secrets/client_secrets.json
"""

import json
import os
import sys

SECRETS_DIR    = os.path.join(os.path.dirname(__file__), "..", "secrets")
CLIENT_SECRETS = os.path.join(SECRETS_DIR, "client_secrets.json")
TOKEN_PATH     = os.path.join(SECRETS_DIR, "yt_token.json")

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def main():
    if not os.path.exists(CLIENT_SECRETS):
        print(f"[錯誤] 找不到 {CLIENT_SECRETS}")
        print()
        print("步驟：")
        print("  1. 前往 https://console.cloud.google.com/")
        print("  2. 建立（或選擇）專案 → API 與服務 → 啟用 YouTube Data API v3")
        print("  3. 憑證 → 建立憑證 → OAuth 用戶端 ID → 類型：桌面應用程式")
        print("  4. 下載 JSON → 存為 secrets/client_secrets.json")
        sys.exit(1)

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
    except ImportError:
        print("[錯誤] 請先安裝：pip install google-auth-oauthlib google-auth-httplib2")
        sys.exit(1)

    creds = None

    # Reuse existing token if still valid
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH) as f:
            creds = Credentials.from_authorized_user_info(json.load(f), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            print("Token 已自動刷新。")
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS, SCOPES)
            # run_local_server opens a browser for OAuth consent
            creds = flow.run_local_server(port=8080, open_browser=True)
            print("授權完成！")

        os.makedirs(SECRETS_DIR, exist_ok=True)
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())
        print(f"Token 已儲存至 {TOKEN_PATH}")

    print()
    print("設定完成。現在可以啟動容器並使用 upload_to_youtube 工具。")
    print(f"容器掛載指令：-v $(pwd)/secrets:/app/secrets")


if __name__ == "__main__":
    main()
