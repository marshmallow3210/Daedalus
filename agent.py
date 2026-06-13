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
    # 實務上可串接 Tavily API，這裡先做標準的結構化回傳供大模型讀取
    return f"【網路搜尋結果】針對 '{query}' 的即時檢索：2026 年最新規格與技術社群討論已就緒。"

@tool
def python_executor(code: str) -> str:
    """當需要進行數學計算、數據分析、邏輯推導或執行 Python 程式碼時使用。輸入必須是標準的 Python 程式碼。"""
    import sys
    from io import StringIO
    
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    
    try:
        # 在 Docker 沙盒內相對安全地執行大模型生成的代碼
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
# 🧠 2. 初始化支援工具呼叫的 Gemma 4
# ==========================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
llm = ChatOllama(model="gemma4:26b", base_url=OLLAMA_BASE_URL, temperature=0.2).bind_tools(tools)

# ==========================================
# 🕸️ 3. 組裝 ReAct 狀態機 (Graph)
# ==========================================

def call_model(state: MessagesState):
    """思考節點：讓模型決定下一步要回答還是用工具"""
    messages = state["messages"]
    sys_prompt = {"role": "system", "content": "你是一個配備了『網路搜尋』與『Python 沙盒執行器』的全能助理 Daedalus。請善用工具精準回答，並一律使用繁體中文。"}
    response = llm.invoke([sys_prompt] + messages)
    return {"messages": [response]}

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

app = workflow.compile()