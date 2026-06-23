import json
import sys
import importlib
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock, mock_open
from langchain_core.messages import AIMessage, ToolMessage
from agent import app, route_after_agent, DANGEROUS_TOOL_NAMES


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Provide an isolated SQLite DB for encyclopedia unit tests."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("ENCYCLOPEDIA_DB_PATH", db_path)
    import encyclopedia as enc
    importlib.reload(enc)
    enc.init_db()
    return enc


# ──────────────────────────────────────────────────────────────
# Graph routing
# ──────────────────────────────────────────────────────────────

async def mock_stream_1(*args, **kwargs):
    yield AIMessage(
        content="",
        tool_calls=[{"name": "python_executor", "args": {"code": "print(2**10)"}, "id": "c1"}],
    )

async def mock_stream_2(*args, **kwargs):
    yield AIMessage(content="計算結果是 1024。")


@pytest.mark.asyncio
@patch("agent.llm")
async def test_agent_tool_routing_safe(mock_llm):
    """Safe tool call should flow through tools node without interrupt."""
    mock_llm.astream.side_effect = [mock_stream_1(), mock_stream_2()]
    config = {"configurable": {"thread_id": "test_safe_routing"}}
    result = await app.ainvoke({"messages": [{"role": "user", "content": "算 2^10"}]}, config=config)
    assert result["messages"][-1].content == "計算結果是 1024。"


def test_route_after_agent_safe_tool():
    """route_after_agent returns 'tools' for safe tool calls."""
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "web_search", "args": {"query": "test"}, "id": "c1"}],
    )
    assert route_after_agent({"messages": [msg]}) == "tools"


def test_route_after_agent_dangerous_tool():
    """route_after_agent returns 'pre_tool_check' for dangerous tools."""
    for tool_name in DANGEROUS_TOOL_NAMES:
        msg = AIMessage(
            content="",
            tool_calls=[{"name": tool_name, "args": {}, "id": "c1"}],
        )
        assert route_after_agent({"messages": [msg]}) == "pre_tool_check"


def test_route_after_agent_no_tool_calls():
    """route_after_agent returns 'end' when there are no tool calls."""
    msg = AIMessage(content="普通回覆，沒有工具呼叫。")
    assert route_after_agent({"messages": [msg]}) == "end"


# ──────────────────────────────────────────────────────────────
# web_search
# ──────────────────────────────────────────────────────────────

def test_web_search_ddg_success():
    from agent import web_search
    mock_inst = MagicMock()
    mock_inst.__enter__ = lambda s: s
    mock_inst.__exit__ = MagicMock(return_value=False)
    mock_inst.text.return_value = [
        {"title": "JLPT N5 詞彙", "body": "ねこ 貓", "href": "https://example.com"}
    ]
    mock_mod = MagicMock(); mock_mod.DDGS = MagicMock(return_value=mock_inst)
    with patch.dict(sys.modules, {"duckduckgo_search": mock_mod}):
        result = web_search.invoke({"query": "JLPT N5 詞彙"})
    assert "JLPT N5 詞彙" in result and "example.com" in result


def test_web_search_import_error_fallback():
    from agent import web_search
    with patch.dict(sys.modules, {"duckduckgo_search": None}):
        result = web_search.invoke({"query": "test"})
    assert "未安裝" in result or "duckduckgo" in result


# ──────────────────────────────────────────────────────────────
# fetch_web_page
# ──────────────────────────────────────────────────────────────

def test_fetch_web_page_strips_html():
    from agent import fetch_web_page
    html = b"<html><body><p>Hello</p><script>evil()</script><p>World</p></body></html>"

    class _Resp:
        def read(self): return html
        def __enter__(self): return self
        def __exit__(self, *a): pass

    with patch("urllib.request.urlopen", return_value=_Resp()):
        r = fetch_web_page.invoke({"url": "https://example.com"})
    assert "Hello" in r and "World" in r and "evil" not in r


def test_fetch_web_page_error():
    from agent import fetch_web_page
    with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
        r = fetch_web_page.invoke({"url": "https://unreachable.invalid"})
    assert "無法取得網頁" in r


# ──────────────────────────────────────────────────────────────
# encyclopedia tools (via mocked module)
# ──────────────────────────────────────────────────────────────

@patch("encyclopedia.init_db")
def test_init_encyclopedia_db_success(mock_init):
    from agent import init_encyclopedia_db
    r = init_encyclopedia_db.invoke({})
    mock_init.assert_called_once()
    assert "成功" in r


@patch("encyclopedia.init_db", side_effect=Exception("磁碟錯誤"))
def test_init_encyclopedia_db_error(mock_init):
    from agent import init_encyclopedia_db
    r = init_encyclopedia_db.invoke({})
    assert "磁碟錯誤" in r


@patch("encyclopedia.upsert_word")
def test_add_japanese_word_valid(mock_upsert):
    from agent import add_japanese_word
    payload = json.dumps({
        "hiragana": "ねこ", "chinese_translation": "貓",
        "jlpt_level": "N5", "emoji": "🐱", "etymology": "",
    })
    r = add_japanese_word.invoke({"word_json": payload})
    mock_upsert.assert_called_once_with(
        "ねこ", chinese_translation="貓", jlpt_level="N5", emoji="🐱", etymology="",
    )
    assert "ねこ" in r


def test_add_japanese_word_invalid_json():
    from agent import add_japanese_word
    r = add_japanese_word.invoke({"word_json": "{ bad"})
    assert "JSON 解析失敗" in r


def test_add_japanese_word_missing_hiragana():
    from agent import add_japanese_word
    r = add_japanese_word.invoke({"word_json": '{"kanji":"猫"}'})
    assert "hiragana" in r and "必填" in r


@patch("encyclopedia.get_words_for_video", return_value=[
    {"hiragana": "さくら", "kanji": "桜", "chinese_translation": "櫻花",
     "jlpt_level": "N4", "emoji": "🌸", "video_count": 0}
])
def test_get_video_candidate_words(mock_get):
    from agent import get_video_candidate_words
    r = get_video_candidate_words.invoke({"limit": 10, "jlpt_level": "N4"})
    mock_get.assert_called_once_with(limit=10, jlpt_level="N4")
    data = json.loads(r)
    assert data[0]["hiragana"] == "さくら"


@patch("encyclopedia.get_words_for_video", return_value=[])
def test_get_video_candidate_words_empty(mock_get):
    from agent import get_video_candidate_words
    r = get_video_candidate_words.invoke({"limit": 50, "jlpt_level": ""})
    assert "新增" in r or "沒有" in r


# ──────────────────────────────────────────────────────────────
# forge_and_test_tool
# ──────────────────────────────────────────────────────────────

def test_forge_blocks_import():
    from agent import forge_and_test_tool
    r = forge_and_test_tool.invoke({
        "tool_code": "import os\ndef f(): return os.getcwd()",
        "test_code": "import unittest\nclass T(unittest.TestCase):\n    def test_x(self): pass",
    })
    assert "安全掃描失敗" in r


def test_forge_blocks_exec():
    from agent import forge_and_test_tool
    r = forge_and_test_tool.invoke({
        "tool_code": "def f(x): exec(x)",
        "test_code": "import unittest\nclass T(unittest.TestCase):\n    def test_x(self): pass",
    })
    assert "安全掃描失敗" in r


def test_forge_blocks_eval():
    from agent import forge_and_test_tool
    r = forge_and_test_tool.invoke({
        "tool_code": "def f(x): return eval(x)",
        "test_code": "import unittest\nclass T(unittest.TestCase):\n    def test_x(self): pass",
    })
    assert "安全掃描失敗" in r


def test_forge_rejects_failing_tests():
    from agent import forge_and_test_tool
    r = forge_and_test_tool.invoke({
        "tool_code": "def add(a,b): return a - b",
        "test_code": (
            "import unittest\nclass T(unittest.TestCase):\n"
            "    def test_add(self): self.assertEqual(add(1,2),3)\n"
        ),
    })
    assert "單元測試失敗" in r


def test_forge_passes_and_writes():
    from agent import forge_and_test_tool
    fake = MagicMock()
    with (
        patch("builtins.open", mock_open()),
        patch.dict(sys.modules, {"custom_tools": fake}),
        patch.object(importlib, "reload", return_value=fake),
    ):
        r = forge_and_test_tool.invoke({
            "tool_code": "def add(a,b): return a+b",
            "test_code": (
                "import unittest\nclass T(unittest.TestCase):\n"
                "    def test_add(self): self.assertEqual(add(1,2),3)\n"
            ),
        })
    assert "成功" in r


# ──────────────────────────────────────────────────────────────
# generate_japanese_learning_video
# ──────────────────────────────────────────────────────────────

def test_generate_video_dry_run():
    from agent import generate_japanese_learning_video
    batch = json.dumps([{"hiragana": "ねこ", "chinese_translation": "貓", "emoji": "🐱"}])
    with patch.dict(sys.modules, {"PIL.Image": None, "moviepy": None, "edge_tts": None}):
        r = generate_japanese_learning_video.invoke({"batch_json": batch})
    assert "Dry-run" in r


@patch("encyclopedia.get_words_for_video", return_value=[])
def test_generate_video_empty_source(mock_get):
    from agent import generate_japanese_learning_video
    r = generate_japanese_learning_video.invoke({"batch_json": "", "jlpt_level": "N5"})
    assert "沒有可用" in r or "新增" in r


def test_generate_video_invalid_json():
    from agent import generate_japanese_learning_video
    r = generate_japanese_learning_video.invoke({"batch_json": "{bad"})
    assert "JSON 解析失敗" in r


# ──────────────────────────────────────────────────────────────
# upload_to_youtube
# ──────────────────────────────────────────────────────────────

def test_upload_to_youtube_missing_file():
    from agent import upload_to_youtube
    r = upload_to_youtube.invoke({
        "local_path": "/nonexistent/video.mp4",
        "batch_id": "test001",
    })
    assert "找不到影片" in r


def test_upload_to_youtube_missing_token(tmp_path):
    from agent import upload_to_youtube
    # File exists but no token
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    with patch("os.path.exists", side_effect=lambda p: p == str(video)):
        r = upload_to_youtube.invoke({
            "local_path": str(video),
            "batch_id": "test001",
        })
    assert "yt_token.json" in r or "認證失敗" in r


@patch("encyclopedia.get_today_upload_count", return_value=5)
def test_upload_to_youtube_quota_exceeded(mock_count, tmp_path):
    from agent import upload_to_youtube
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    token = tmp_path / "yt_token.json"
    token.write_text("{}")
    with patch("os.path.exists", return_value=True):
        r = upload_to_youtube.invoke({
            "local_path": str(video),
            "batch_id": "test001",
        })
    assert "配額" in r


# ──────────────────────────────────────────────────────────────
# delete_local_video
# ──────────────────────────────────────────────────────────────

@patch("encyclopedia.get_batch", return_value=None)
def test_delete_local_video_no_db_record(mock_get):
    from agent import delete_local_video
    r = delete_local_video.invoke({"local_path": "/tmp/x.mp4", "batch_id": "b1"})
    assert "拒" in r or "找不到" in r


@patch("encyclopedia.get_batch", return_value={"batch_id": "b1", "youtube_video_id": None})
def test_delete_local_video_no_youtube_id(mock_get):
    from agent import delete_local_video
    r = delete_local_video.invoke({"local_path": "/tmp/x.mp4", "batch_id": "b1"})
    assert "video_id" in r or "拒" in r


@patch("encyclopedia.get_batch", return_value={"batch_id": "b1", "youtube_video_id": "abc123"})
@patch("encyclopedia.clear_batch_local_path")
def test_delete_local_video_success(mock_clear, mock_get, tmp_path):
    from agent import delete_local_video
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    r = delete_local_video.invoke({"local_path": str(video), "batch_id": "b1"})
    assert "已刪除" in r
    assert not video.exists()
    mock_clear.assert_called_once_with("b1")


@patch("encyclopedia.get_batch", return_value={"batch_id": "b1", "youtube_video_id": "abc123"})
def test_delete_local_video_already_gone(mock_get):
    from agent import delete_local_video
    r = delete_local_video.invoke({"local_path": "/tmp/not_here.mp4", "batch_id": "b1"})
    assert "已不存在" in r or "不存在" in r


# ──────────────────────────────────────────────────────────────
# encyclopedia — video_batches unit tests (isolated DB)
# ──────────────────────────────────────────────────────────────

def test_encyclopedia_batch_lifecycle(temp_db):
    enc = temp_db
    enc.create_batch("b001", 50, "/app/videos/test.mp4")

    batch = enc.get_batch("b001")
    assert batch["status"] == "generating"
    assert batch["word_count"] == 50
    assert batch["local_path"] == "/app/videos/test.mp4"
    assert batch["youtube_video_id"] is None

    enc.update_batch_status("b001", "uploading")
    assert enc.get_batch("b001")["status"] == "uploading"

    enc.update_batch_youtube_id("b001", "yt_XYZ")
    b = enc.get_batch("b001")
    assert b["youtube_video_id"] == "yt_XYZ"
    assert b["status"] == "completed"
    assert b["completed_at"] is not None

    enc.clear_batch_local_path("b001")
    assert enc.get_batch("b001")["local_path"] is None


def test_encyclopedia_get_today_upload_count(temp_db):
    enc = temp_db
    assert enc.get_today_upload_count() == 0
    enc.create_batch("c1", 10, "/tmp/a.mp4")
    enc.update_batch_youtube_id("c1", "vid1")
    assert enc.get_today_upload_count() == 1


def test_encyclopedia_upsert_and_query(temp_db):
    enc = temp_db
    enc.upsert_word("ねこ", kanji="猫", chinese_translation="貓",
                    jlpt_level="N5", emoji="🐱")
    words = enc.get_words_for_video(limit=10)
    assert any(w["hiragana"] == "ねこ" for w in words)


def test_encyclopedia_mark_words_in_video(temp_db):
    enc = temp_db
    enc.upsert_word("いぬ", chinese_translation="狗")
    enc.upsert_word("ねこ", chinese_translation="貓")
    updated = enc.mark_words_in_video(["いぬ", "ねこ"], "batch01", "yt_abc")
    assert updated == 2
    remaining = enc.get_words_for_video(limit=10)
    assert all(w["hiragana"] not in ("いぬ", "ねこ") for w in remaining)


def test_encyclopedia_get_words_for_video_jlpt_filter(temp_db):
    enc = temp_db
    enc.upsert_word("やま", chinese_translation="山", jlpt_level="N5")
    enc.upsert_word("かわ", chinese_translation="河", jlpt_level="N4")
    n5_words = enc.get_words_for_video(jlpt_level="N5")
    assert all(w["jlpt_level"] == "N5" for w in n5_words)
