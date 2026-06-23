# custom_tools.py
# 這裡一開始是空的，專門留給 Daedalus 自主寫入新工具。


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """
    模擬發送電子郵件的工具。
    
    Args:
        to (str): 收件人 Email 地址。
        subject (str): 郵件主旨。
        body (str): 郵件內容。
        
    Returns:
        str: 發送結果訊息。
    """
    print(f"--- [Email Outgoing] ---")
    print(f"To: {to}")
    print(f"Subject: {subject}")
    print(f"Body: {body}")
    print(f"------------------------")
    return f"Email sent successfully to {to}"

