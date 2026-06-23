import json
import sys
import importlib
import pytest
from unittest.mock import patch, MagicMock, mock_open
from langchain_core.messages import AIMessage
from agent import app


# ==========================================
# Existing: agent tool-routing smoke test
# ==========================================

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
    """當模型發出工具呼叫指令時，狀態機路由必須正確運作"""
    mock_llm.astream.side_effect = [mock_stream_1(), mock_stream_2()]
    init_state = {"messages": [{"role": "user", "content": "幫我算 2 的 10 次方"}]}
    result = await app.ainvoke(init_state)
    assert len(result["messages"]) == 4
    assert result["messages"][-1].content == "計算結果是 1024。"


# ==========================================
# encyclopedia tool tests
# ==========================================

@patch('encyclopedia.init_db')
def test_init_encyclopedia_db_success(mock_init):
    """init_encyclopedia_db 應呼叫 encyclopedia.init_db() 並回傳成功訊息"""
    from agent import init_encyclopedia_db
    result = init_encyclopedia_db.invoke({})
    mock_init.assert_called_once()
    assert "成功" in result


@patch('encyclopedia.init_db', side_effect=Exception("磁碟空間不足"))
def test_init_encyclopedia_db_error(mock_init):
    """init_encyclopedia_db 應如實回傳錯誤訊息"""
    from agent import init_encyclopedia_db
    result = init_encyclopedia_db.invoke({})
    assert "初始化失敗" in result
    assert "磁碟空間不足" in result


@patch('encyclopedia.upsert_word')
def test_add_japanese_word_valid(mock_upsert):
    """add_japanese_word 應正確解析 JSON 並呼叫 upsert_word"""
    from agent import add_japanese_word
    payload = json.dumps({
        "hiragana": "ねこ",
        "chinese_translation": "貓",
        "jlpt_level": "N5",
    })
    result = add_japanese_word.invoke({"word_json": payload})
    mock_upsert.assert_called_once_with(
        "ねこ", chinese_translation="貓", jlpt_level="N5"
    )
    assert "ねこ" in result


def test_add_japanese_word_invalid_json():
    """add_japanese_word 傳入非法 JSON 應回傳解析失敗"""
    from agent import add_japanese_word
    result = add_japanese_word.invoke({"word_json": "{not valid json"})
    assert "JSON 解析失敗" in result


def test_add_japanese_word_missing_hiragana():
    """add_japanese_word 缺少 hiragana 欄位應回傳錯誤"""
    from agent import add_japanese_word
    result = add_japanese_word.invoke({"word_json": '{"kanji": "猫"}'})
    assert "hiragana" in result and "必填" in result


@patch('encyclopedia.get_due_words', return_value=[
    {"hiragana": "いぬ", "chinese_translation": "狗", "jlpt_level": "N5",
     "ease_factor": 2.5, "review_count": 0, "next_review_date": "2026-06-23"}
])
def test_get_srs_due_words(mock_get):
    """get_srs_due_words 應返回 JSON 格式的待複習單字"""
    from agent import get_srs_due_words
    result = get_srs_due_words.invoke({"limit": 10})
    mock_get.assert_called_once_with(limit=10)
    data = json.loads(result)
    assert isinstance(data, list)
    assert data[0]["hiragana"] == "いぬ"


@patch('encyclopedia.update_srs', return_value=True)
def test_update_word_srs_valid(mock_update):
    """update_word_srs 應呼叫 SM-2 更新並回傳成功"""
    from agent import update_word_srs
    result = update_word_srs.invoke({"hiragana": "ねこ", "quality": 4})
    mock_update.assert_called_once_with("ねこ", 4)
    assert "ねこ" in result and "更新" in result


def test_update_word_srs_invalid_quality():
    """update_word_srs quality 超出範圍應回傳錯誤"""
    from agent import update_word_srs
    result = update_word_srs.invoke({"hiragana": "ねこ", "quality": 6})
    assert "0 到 5" in result


@patch('encyclopedia.update_srs', return_value=False)
def test_update_word_srs_not_found(mock_update):
    """update_word_srs 找不到單字時應回傳錯誤"""
    from agent import update_word_srs
    result = update_word_srs.invoke({"hiragana": "存在しない", "quality": 3})
    assert "找不到" in result


# ==========================================
# forge_and_test_tool tests
# ==========================================

def test_forge_and_test_tool_blocks_import():
    """tool_code 含 import 語句時應被 AST 掃描攔截"""
    from agent import forge_and_test_tool
    tool_code = "import os\ndef dangerous(): return os.getcwd()"
    test_code = "import unittest\nclass T(unittest.TestCase):\n    def test_x(self): pass"
    result = forge_and_test_tool.invoke({"tool_code": tool_code, "test_code": test_code})
    assert "安全掃描失敗" in result


def test_forge_and_test_tool_blocks_exec():
    """tool_code 含 exec() 呼叫時應被 AST 掃描攔截"""
    from agent import forge_and_test_tool
    tool_code = "def bad(x): exec(x)"
    test_code = "import unittest\nclass T(unittest.TestCase):\n    def test_x(self): pass"
    result = forge_and_test_tool.invoke({"tool_code": tool_code, "test_code": test_code})
    assert "安全掃描失敗" in result


def test_forge_and_test_tool_blocks_eval():
    """tool_code 含 eval() 呼叫時應被 AST 掃描攔截"""
    from agent import forge_and_test_tool
    tool_code = "def bad(x): return eval(x)"
    test_code = "import unittest\nclass T(unittest.TestCase):\n    def test_x(self): pass"
    result = forge_and_test_tool.invoke({"tool_code": tool_code, "test_code": test_code})
    assert "安全掃描失敗" in result


def test_forge_and_test_tool_rejects_failing_tests():
    """單元測試失敗時應拒絕寫入"""
    from agent import forge_and_test_tool
    tool_code = "def add(a, b):\n    return a - b"
    test_code = (
        "import unittest\n"
        "class T(unittest.TestCase):\n"
        "    def test_add(self):\n"
        "        self.assertEqual(add(1, 2), 3)\n"
    )
    result = forge_and_test_tool.invoke({"tool_code": tool_code, "test_code": test_code})
    assert "單元測試失敗" in result


def test_forge_and_test_tool_passes_and_writes():
    """AST 掃描通過且單元測試全過時，應寫入並動態載入"""
    from agent import forge_and_test_tool

    tool_code = "def add(a, b):\n    return a + b"
    test_code = (
        "import unittest\n"
        "class T(unittest.TestCase):\n"
        "    def test_add(self):\n"
        "        self.assertEqual(add(1, 2), 3)\n"
    )

    fake_module = MagicMock()
    with (
        patch("builtins.open", mock_open()),
        patch.dict(sys.modules, {"custom_tools": fake_module}),
        patch.object(importlib, "reload", return_value=fake_module),
    ):
        result = forge_and_test_tool.invoke({"tool_code": tool_code, "test_code": test_code})

    assert "成功" in result


# ==========================================
# generate_japanese_learning_video tests
# ==========================================

def test_generate_japanese_learning_video_dry_run():
    """缺少 moviepy/pillow/edge-tts 時應回傳 Dry-run 訊息而不崩潰"""
    from agent import generate_japanese_learning_video

    batch = json.dumps([
        {"hiragana": "ねこ", "chinese_translation": "貓"},
        {"hiragana": "いぬ", "chinese_translation": "狗"},
    ])

    with patch.dict(sys.modules, {"PIL.Image": None, "moviepy": None, "edge_tts": None}):
        result = generate_japanese_learning_video.invoke({"batch_json": batch})

    assert "Dry-run" in result
    assert "2" in result


def test_generate_japanese_learning_video_invalid_json():
    """傳入非法 JSON 應回傳解析失敗訊息"""
    from agent import generate_japanese_learning_video
    result = generate_japanese_learning_video.invoke({"batch_json": "not json"})
    assert "JSON 解析失敗" in result


def test_generate_japanese_learning_video_empty_batch():
    """傳入空陣列應回傳錯誤訊息"""
    from agent import generate_japanese_learning_video
    result = generate_japanese_learning_video.invoke({"batch_json": "[]"})
    assert "錯誤" in result
