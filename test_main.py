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
# web_search
# ==========================================

def test_web_search_ddg_success():
    """web_search 使用 DuckDuckGo 回傳真實結果"""
    from agent import web_search

    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.__enter__ = lambda s: s
    mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
    mock_ddgs_instance.text.return_value = [
        {"title": "JLPT N5 單字", "body": "ねこ 貓", "href": "https://example.com"}
    ]
    mock_ddgs_cls = MagicMock(return_value=mock_ddgs_instance)
    mock_module = MagicMock()
    mock_module.DDGS = mock_ddgs_cls

    with patch.dict(sys.modules, {"duckduckgo_search": mock_module}):
        result = web_search.invoke({"query": "JLPT N5 單字"})
    assert "JLPT N5 單字" in result
    assert "example.com" in result


def test_web_search_import_error_fallback():
    """duckduckgo_search 未安裝時回傳友善提示而不崩潰"""
    from agent import web_search
    with patch.dict(sys.modules, {"duckduckgo_search": None}):
        result = web_search.invoke({"query": "test"})
    assert "duckduckgo-search" in result or "未安裝" in result


# ==========================================
# fetch_web_page
# ==========================================

def test_fetch_web_page_success():
    """fetch_web_page 應回傳去除 HTML 標籤的純文字"""
    from agent import fetch_web_page
    html = b"<html><body><p>Hello</p><script>evil()</script><p>World</p></body></html>"

    class _FakeResp:
        def read(self): return html
        def __enter__(self): return self
        def __exit__(self, *a): pass

    with patch("urllib.request.urlopen", return_value=_FakeResp()):
        result = fetch_web_page.invoke({"url": "https://example.com"})
    assert "Hello" in result
    assert "World" in result
    assert "evil" not in result


def test_fetch_web_page_error():
    """fetch_web_page 網路錯誤時回傳友善訊息"""
    from agent import fetch_web_page
    with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
        result = fetch_web_page.invoke({"url": "https://unreachable.invalid"})
    assert "無法取得網頁" in result


# ==========================================
# encyclopedia tools
# ==========================================

@patch('encyclopedia.init_db')
def test_init_encyclopedia_db_success(mock_init):
    from agent import init_encyclopedia_db
    result = init_encyclopedia_db.invoke({})
    mock_init.assert_called_once()
    assert "成功" in result


@patch('encyclopedia.init_db', side_effect=Exception("磁碟空間不足"))
def test_init_encyclopedia_db_error(mock_init):
    from agent import init_encyclopedia_db
    result = init_encyclopedia_db.invoke({})
    assert "初始化失敗" in result and "磁碟空間不足" in result


@patch('encyclopedia.upsert_word')
def test_add_japanese_word_valid(mock_upsert):
    """add_japanese_word 應正確解析 JSON（含新欄位）並呼叫 upsert_word"""
    from agent import add_japanese_word
    payload = json.dumps({
        "hiragana": "ねこ",
        "chinese_translation": "貓",
        "jlpt_level": "N5",
        "emoji": "🐱",
        "tags": "動物,寵物",
    })
    result = add_japanese_word.invoke({"word_json": payload})
    mock_upsert.assert_called_once_with(
        "ねこ",
        chinese_translation="貓",
        jlpt_level="N5",
        emoji="🐱",
        tags="動物,寵物",
    )
    assert "ねこ" in result


def test_add_japanese_word_invalid_json():
    from agent import add_japanese_word
    result = add_japanese_word.invoke({"word_json": "{not valid"})
    assert "JSON 解析失敗" in result


def test_add_japanese_word_missing_hiragana():
    from agent import add_japanese_word
    result = add_japanese_word.invoke({"word_json": '{"kanji": "猫"}'})
    assert "hiragana" in result and "必填" in result


@patch('encyclopedia.get_due_words', return_value=[
    {"hiragana": "いぬ", "chinese_translation": "狗", "jlpt_level": "N5",
     "ease_factor": 2.5, "review_count": 0, "next_review_date": "2026-06-23",
     "emoji": "🐶"}
])
def test_get_srs_due_words(mock_get):
    from agent import get_srs_due_words
    result = get_srs_due_words.invoke({"limit": 10})
    mock_get.assert_called_once_with(limit=10)
    data = json.loads(result)
    assert data[0]["hiragana"] == "いぬ"


@patch('encyclopedia.update_srs', return_value=True)
def test_update_word_srs_valid(mock_update):
    from agent import update_word_srs
    result = update_word_srs.invoke({"hiragana": "ねこ", "quality": 4})
    mock_update.assert_called_once_with("ねこ", 4)
    assert "更新" in result


def test_update_word_srs_invalid_quality():
    from agent import update_word_srs
    result = update_word_srs.invoke({"hiragana": "ねこ", "quality": 6})
    assert "0 到 5" in result


@patch('encyclopedia.update_srs', return_value=False)
def test_update_word_srs_not_found(mock_update):
    from agent import update_word_srs
    result = update_word_srs.invoke({"hiragana": "存在しない", "quality": 3})
    assert "找不到" in result


# ==========================================
# get_video_candidate_words
# ==========================================

@patch('encyclopedia.get_words_for_video', return_value=[
    {"hiragana": "さくら", "kanji": "桜", "chinese_translation": "櫻花",
     "jlpt_level": "N4", "emoji": "🌸", "video_count": 0}
])
def test_get_video_candidate_words(mock_get):
    from agent import get_video_candidate_words
    result = get_video_candidate_words.invoke({"limit": 10, "jlpt_level": "N4"})
    mock_get.assert_called_once_with(limit=10, jlpt_level="N4")
    data = json.loads(result)
    assert data[0]["hiragana"] == "さくら"


@patch('encyclopedia.get_words_for_video', return_value=[])
def test_get_video_candidate_words_empty(mock_get):
    from agent import get_video_candidate_words
    result = get_video_candidate_words.invoke({"limit": 50, "jlpt_level": ""})
    assert "add_japanese_word" in result or "新增" in result


# ==========================================
# forge_and_test_tool
# ==========================================

def test_forge_and_test_tool_blocks_import():
    from agent import forge_and_test_tool
    result = forge_and_test_tool.invoke({
        "tool_code": "import os\ndef bad(): return os.getcwd()",
        "test_code": "import unittest\nclass T(unittest.TestCase):\n    def test_x(self): pass",
    })
    assert "安全掃描失敗" in result


def test_forge_and_test_tool_blocks_exec():
    from agent import forge_and_test_tool
    result = forge_and_test_tool.invoke({
        "tool_code": "def bad(x): exec(x)",
        "test_code": "import unittest\nclass T(unittest.TestCase):\n    def test_x(self): pass",
    })
    assert "安全掃描失敗" in result


def test_forge_and_test_tool_blocks_eval():
    from agent import forge_and_test_tool
    result = forge_and_test_tool.invoke({
        "tool_code": "def bad(x): return eval(x)",
        "test_code": "import unittest\nclass T(unittest.TestCase):\n    def test_x(self): pass",
    })
    assert "安全掃描失敗" in result


def test_forge_and_test_tool_rejects_failing_tests():
    from agent import forge_and_test_tool
    result = forge_and_test_tool.invoke({
        "tool_code": "def add(a, b):\n    return a - b",
        "test_code": (
            "import unittest\n"
            "class T(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(add(1, 2), 3)\n"
        ),
    })
    assert "單元測試失敗" in result


def test_forge_and_test_tool_passes_and_writes():
    from agent import forge_and_test_tool
    fake_module = MagicMock()
    with (
        patch("builtins.open", mock_open()),
        patch.dict(sys.modules, {"custom_tools": fake_module}),
        patch.object(importlib, "reload", return_value=fake_module),
    ):
        result = forge_and_test_tool.invoke({
            "tool_code": "def add(a, b):\n    return a + b",
            "test_code": (
                "import unittest\n"
                "class T(unittest.TestCase):\n"
                "    def test_add(self):\n"
                "        self.assertEqual(add(1, 2), 3)\n"
            ),
        })
    assert "成功" in result


# ==========================================
# generate_japanese_learning_video
# ==========================================

def test_generate_video_dry_run_missing_packages():
    """缺少 pillow/moviepy/edge-tts 時回傳 Dry-run 不崩潰"""
    from agent import generate_japanese_learning_video
    batch = json.dumps([
        {"hiragana": "ねこ", "chinese_translation": "貓", "emoji": "🐱"},
        {"hiragana": "いぬ", "chinese_translation": "狗", "emoji": "🐶"},
    ])
    with patch.dict(sys.modules, {"PIL.Image": None, "moviepy": None, "edge_tts": None}):
        result = generate_japanese_learning_video.invoke({"batch_json": batch})
    assert "Dry-run" in result


@patch('encyclopedia.get_words_for_video', return_value=[])
def test_generate_video_empty_encyclopedia(mock_get):
    """百科全書為空時回傳友善提示"""
    from agent import generate_japanese_learning_video
    result = generate_japanese_learning_video.invoke({"batch_json": "", "jlpt_level": "N5"})
    assert "新增" in result or "沒有可用" in result


def test_generate_video_invalid_json():
    from agent import generate_japanese_learning_video
    result = generate_japanese_learning_video.invoke({"batch_json": "{bad json"})
    assert "JSON 解析失敗" in result
