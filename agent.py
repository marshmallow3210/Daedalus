from langchain_core.runnables import RunnableConfig
import os
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode

# ==========================================
# 🛠️ 1. 定義萬能工具箱 (Tools)
# ==========================================

@tool
def web_search(query: str) -> str:
    """當需要查詢最新網路資訊、即時事實、天氣或科技新聞時使用此工具。"""
    return f"【網路搜尋結果】針對 '{query}' 的即時檢索：2026 年最新規格與技術社群討論已就緒。"

@tool
def python_executor(code: str) -> str:
    """當需要進行數學計算、數據分析、邏輯推導或執行 Python 程式碼時使用。輸入必須是標準的 Python 程式碼。"""
    import sys
    from io import StringIO
    
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    
    try:
        exec(code, {"__builtins__": __import__("builtins")}, {})
        sys.stdout = old_stdout
        output = redirected_output.getvalue()
        return f" 執行成功！輸出結果如下：\n{output}" if output else " 執行成功（無輸出內容）。"
    except Exception as e:
        sys.stdout = old_stdout
        return f" 執行出錯，錯誤訊息：{str(e)}"

# 打包工具節點
tools = [web_search, python_executor]
tool_node = ToolNode(tools)

# ==========================================
# 🧠 2. 初始化支援工具呼叫的 Gemma 4 (開啟串流模式)
# ==========================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

# 📢 關鍵點：在此處補上了 streaming=True
llm = ChatOllama(
    model="gemma4:26b", 
    # model="gemma4:12b", 
    base_url=OLLAMA_BASE_URL, 
    temperature=0.2, 
    streaming=True
).bind_tools(tools)

# ==========================================
# 🕸️ 3. 組裝 ReAct 狀態機 (Graph)
# ==========================================

async def call_model(state: MessagesState, config: RunnableConfig): # 📢 1. 這裡一定要宣告接收 config
    messages = state["messages"]
    sys_prompt = {"role": "system", "content": "你是一個配備了『網路搜尋』與『Python 沙盒執行器』的全能助理 Daedalus。請善用工具精準回答，並一律使用繁體中文。"}
    
    final_response = None
    # 📢 2. 關鍵：一定要把 config=config 傳進去！這就像是貼上傳遞標籤，Tokens 才知道要往 Chainlit 噴發
    async for chunk in llm.astream([sys_prompt] + messages, config=config):
        if final_response is None:
            final_response = chunk
        else:
            final_response += chunk
            
    return {"messages": [final_response]}

def should_continue(state: MessagesState):
    """路由判斷：檢查模型有沒有發出工具呼叫指令"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")

# 📢 這裡產出的變數名稱叫 app，所以 app.py 寫 from agent import app as graph 是百分之百正確的！
app = workflow.compile()