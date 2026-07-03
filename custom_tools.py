# custom_tools.py
# 這裡一開始是空的，專門留給 Daedalus 自主寫入新工具。


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


def calculate_sm2_ease_factor(quality, current_ef):
    """
    Calculates the new Ease Factor (EF) using a standard SM-2 variant.

    Args:
        quality (float): The quality score of the review, from 0 to 5.
        current_ef (float): The current ease factor.

    Returns:
        float: The updated ease factor, with a minimum bound of 1.3.
    """
    change = 0.1 - (1 - (quality / 5.0)) * 0.3
    new_ef = current_ef + change
    return max(1.3, new_ef)


def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


def calculate_factorial(n):
    if n < 0:
        return "Error: Negative number"
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def download_webpage_logic(response):
    """
    這是一個邏輯拆解後的函式，用來處理 response 物件。
    因為無法在 tool_code 使用 import requests，我將邏輯抽離出來以便測試。
    """
    try:
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"發生錯誤: {e}"