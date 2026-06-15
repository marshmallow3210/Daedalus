import pytest
from unittest.mock import patch
from langchain_core.messages import AIMessage
from agent import app

# 這兩個非同步生成器保持不變
async def mock_stream_1(*args, **kwargs):
    yield AIMessage(
        content="",
        tool_calls=[{"name": "python_executor", "args": {"code": "print(2**10)"}, "id": "call_123"}]
    )

async def mock_stream_2(*args, **kwargs):
    yield AIMessage(content="計算結果是 1024。")


@pytest.mark.asyncio
@patch('agent.llm')
async def test_agent_tool_routing(mock_llm):
    """測試當模型發出工具呼叫指令時，Daedalus 的狀態機路由是否能正確運作"""
    
    # 💡 嚴謹修正：加上括號 ()，依序回傳實例化後的非同步生成器物件
    mock_llm.astream.side_effect = [mock_stream_1(), mock_stream_2()]
    
    init_state = {"messages": [{"role": "user", "content": "幫我算 2 的 10 次方"}]}
    
    # 執行非同步狀態機
    result = await app.ainvoke(init_state)
    
    # 驗證
    assert len(result["messages"]) == 4
    assert result["messages"][-1].content == "計算結果是 1024。"