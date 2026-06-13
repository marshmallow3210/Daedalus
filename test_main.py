from unittest.mock import patch
from langchain_core.messages import AIMessage
from agent import app

@patch('agent.llm')  # 💡 關鍵修正：這裡改成 Mock 整個 llm 物件，而不是 llm.invoke
def test_agent_tool_routing(mock_llm):
    """測試當模型發出工具呼叫指令時，Daedalus 的狀態機路由是否能正確運作"""
    
    # 模擬第一步：模型決定調用 python_executor
    first_ai_response = AIMessage(
        content="",
        tool_calls=[{"name": "python_executor", "args": {"code": "print(2**10)"}, "id": "call_123"}]
    )
    # 模擬第二步：拿到工具結果後，模型給出最終回答
    second_ai_response = AIMessage(content="計算結果是 1024。")
    
    # 💡 關鍵修正：把模擬的連續回傳值，綁定到 mock_llm 的 invoke 方法上
    mock_llm.invoke.side_effect = [first_ai_response, second_ai_response]
    
    init_state = {"messages": [{"role": "user", "content": "幫我算 2 的 10 次方"}]}
    result = app.invoke(init_state)
    
    # 驗證訊息鏈長度與最終結果，確保流程順利走完
    assert len(result["messages"]) == 4
    assert result["messages"][-1].content == "計算結果是 1024。"