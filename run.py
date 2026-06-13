from agent import app
from langchain_core.messages import HumanMessage

if __name__ == "__main__":
    print("🚀 啟動全能助理 Daedalus (雙工具配置)...")
    
    # 給它一個需要「計算」加「搜尋」的綜合複雜問題
    prompt = "請幫我計算 2 的 24 次方是多少？並順便上網查 2026 年最新款 M5 晶片的傳聞。"
    inputs = {"messages": [HumanMessage(content=prompt)]}
    
    for output in app.stream(inputs, stream_mode="values"):
        messages = output.get("messages", [])
        if messages:
            last_msg = messages[-1]
            role = "🤖 Daedalus" if last_msg.type == "ai" else "👤 User" if last_msg.type == "human" else "🛠️ Tool"
            print(f"\n🔹 [{role}]:")
            print(last_msg.content if last_msg.content else f"(發動工具呼叫: {last_msg.tool_calls})")