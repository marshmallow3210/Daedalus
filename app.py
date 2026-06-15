import os
import chainlit as cl
# 📢 注意：請根據妳先前在 agent.py 底部的變數名稱（app 或 agent），維持正確的匯入與別名
from agent import app as graph 

@cl.on_chat_start
async def start():
    """網頁打開時觸發"""
    await cl.Message(
        content="🕶️ **Daedalus 系統已上線**。特務 marshmallow3210，請下達指令。（⚡ 即時噴字串流模式已全面啟動）"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """當使用者在網頁輸入指令時觸發"""
    
    # 1. 先建立一個空的訊息框，準備用來實時塞入文字 Token
    msg = cl.Message(content="")
    
    inputs = {"messages": [("user", message.content)]}
    config = {"configurable": {"thread_id": "daedalus_chat"}}
    
    try:
        # 2. 監聽 LangGraph 的內部事件串流
        async for event in graph.astream_events(inputs, config=config, version="v2"):
            kind = event.get("event")
            
            # 3. 當偵測到 LLM 大腦（Chat Model）正在吐出即時 Token 時
            if kind == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token:
                    # 4. 把拿到的小 Token 即時噴射到網頁上！
                    await msg.stream_token(token)
                    
        # 5. 串流結束後，正式將整段訊息存檔
        await msg.update()
        
    except Exception as e:
        # 錯誤處理機制
        msg.content = f"❌ 報告特務，串流運作時發生錯誤：\n```text\n{str(e)}\n```"
        await msg.update()