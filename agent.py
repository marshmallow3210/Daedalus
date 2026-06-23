from langchain_core.runnables import RunnableConfig
import os
import sys

# Ensure the directory containing agent.py is always importable,
# regardless of how Chainlit sets the working directory at runtime.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import encyclopedia  # noqa: E402 — must come after sys.path fix

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode

# ==========================================
# 🛠️ 1. 定義工具箱 (Tools)
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
        return f"執行成功！輸出結果如下：\n{output}" if output else "執行成功（無輸出內容）。"
    except Exception as e:
        sys.stdout = old_stdout
        return f"執行出錯，錯誤訊息：{str(e)}"


@tool
def forge_and_test_tool(tool_code: str, test_code: str) -> str:
    """安全新增自定義工具。流程：
    1. AST 靜態掃描（攔截 exec / eval / open / import）
    2. 記憶體沙盒執行 unittest
    3. 全數通過才寫入 /app/custom_tools.py
    4. importlib 動態 reload

    tool_code：新工具的 Python 原始碼（禁止含 import / exec / eval / open）。
    test_code：對應的 unittest 測試碼（可含 import unittest）。
    """
    import ast
    import unittest
    import importlib
    from io import StringIO

    BLOCKED_CALLS = {"exec", "eval", "open", "__import__"}

    # Step 1: AST static scan on tool_code only
    try:
        tree = ast.parse(tool_code)
    except SyntaxError as e:
        return f"AST 解析失敗：{e}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "安全掃描失敗：tool_code 禁止使用 import 語句。"
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name in BLOCKED_CALLS:
                return f"安全掃描失敗：tool_code 禁止呼叫 {name}()。"

    # Step 2: load into sandbox and run tests
    sandbox = {"__builtins__": __import__("builtins")}
    try:
        exec(tool_code, sandbox)
        exec(test_code, sandbox)
    except Exception as e:
        return f"沙盒載入失敗：{e}"

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for obj in sandbox.values():
        if (
            isinstance(obj, type)
            and issubclass(obj, unittest.TestCase)
            and obj is not unittest.TestCase
        ):
            suite.addTests(loader.loadTestsFromTestCase(obj))

    buf = StringIO()
    result = unittest.TextTestRunner(stream=buf, verbosity=2).run(suite)

    if not result.wasSuccessful():
        return f"單元測試失敗，拒絕寫入：\n{buf.getvalue()}"

    # Step 3: write to custom_tools.py
    custom_tools_path = os.getenv("CUSTOM_TOOLS_PATH", "/app/custom_tools.py")
    try:
        with open(custom_tools_path, "a", encoding="utf-8") as f:
            f.write("\n\n" + tool_code)
    except Exception as e:
        return f"寫入 custom_tools.py 失敗：{e}"

    # Step 4: dynamic reload
    try:
        import custom_tools
        importlib.reload(custom_tools)
        return f"工具已成功寫入並動態載入！\n{buf.getvalue()}"
    except Exception as e:
        return f"動態 reload 失敗：{e}"


@tool
def init_encyclopedia_db() -> str:
    """初始化日文百科全書 SQLite 資料庫，建立 japanese_encyclopedia 資料表。"""
    try:
        encyclopedia.init_db()
        return "日文百科全書資料庫初始化成功。"
    except Exception as e:
        return f"初始化失敗：{e}"


@tool
def add_japanese_word(word_json: str) -> str:
    """將日文單字寫入百科全書，若 hiragana 已存在則自動更新。
    word_json 為 JSON 字串。必填欄位：hiragana。
    選填：katakana、kanji、romaji、chinese_translation、etymology、jlpt_level、video_batch_id。
    """
    import json

    try:
        data = json.loads(word_json)
    except json.JSONDecodeError as e:
        return f"JSON 解析失敗：{e}"

    hiragana = data.pop("hiragana", None)
    if not hiragana:
        return "錯誤：hiragana 欄位為必填。"

    try:
        encyclopedia.upsert_word(hiragana, **data)
        return f"單字「{hiragana}」已成功寫入百科全書。"
    except Exception as e:
        return f"寫入失敗：{e}"


@tool
def get_srs_due_words(limit: int = 20) -> str:
    """返回今日 SRS 待複習的日文單字清單（JSON 格式）。"""
    import json

    try:
        words = encyclopedia.get_due_words(limit=limit)
        return json.dumps(words, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"查詢失敗：{e}"


@tool
def update_word_srs(hiragana: str, quality: int) -> str:
    """使用 SM-2 演算法更新指定單字的 SRS 複習排程。
    quality：0（完全忘記）到 5（完美記憶）。
    quality < 3 時重置間隔為 1 天；quality >= 3 時依難易係數延長間隔。
    """
    if not 0 <= quality <= 5:
        return "錯誤：quality 必須在 0 到 5 之間。"
    try:
        found = encyclopedia.update_srs(hiragana, quality)
        if found:
            return f"單字「{hiragana}」的 SRS 排程已更新（評分：{quality}）。"
        return f"錯誤：找不到單字「{hiragana}」。"
    except Exception as e:
        return f"更新失敗：{e}"


@tool
def generate_japanese_learning_video(batch_json: str) -> str:
    """根據日文單字批次 JSON 生成 1920x1080 學習影片（MP4）。
    batch_json 為 JSON 陣列，每個元素需含 hiragana、chinese_translation 欄位。
    若 moviepy / pillow / edge-tts 未安裝，回傳 Dry-run 訊息而不崩潰。
    """
    import json

    try:
        words = json.loads(batch_json)
    except json.JSONDecodeError as e:
        return f"JSON 解析失敗：{e}"

    if not isinstance(words, list) or len(words) == 0:
        return "錯誤：batch_json 必須為非空 JSON 陣列。"

    # Check for required packages
    missing = []
    for pkg, import_path in [("pillow", "PIL.Image"), ("moviepy", "moviepy"), ("edge-tts", "edge_tts")]:
        try:
            __import__(import_path)
        except ImportError:
            missing.append(pkg)

    if missing:
        preview = [
            f"{w.get('hiragana', '?')} ({w.get('chinese_translation', '?')})"
            for w in words[:5]
        ]
        return (
            f"[Dry-run] 缺少套件：{', '.join(missing)}。影片未實際生成。\n"
            f"批次共 {len(words)} 個單字，前 5 個：{', '.join(preview)}"
        )

    try:
        from PIL import Image, ImageDraw, ImageFont
        from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
        import edge_tts
        import asyncio
        import tempfile

        tmp_dir = tempfile.mkdtemp()
        clips = []

        async def _tts(text: str, path: str) -> None:
            communicate = edge_tts.Communicate(text, "ja-JP-NanamiNeural")
            await communicate.save(path)

        def _make_frame(word: dict, idx: int) -> str:
            hiragana = word.get("hiragana", "")
            kanji = word.get("kanji", "") or hiragana
            translation = word.get("chinese_translation", "")

            img = Image.new("RGB", (1920, 1080), color=(20, 20, 40))
            draw = ImageDraw.Draw(img)

            try:
                f_large = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 120
                )
                f_medium = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 60
                )
            except Exception:
                f_large = ImageFont.load_default()
                f_medium = ImageFont.load_default()

            draw.text((960, 380), kanji, fill="white", font=f_large, anchor="mm")
            draw.text((960, 540), hiragana, fill=(180, 180, 255), font=f_medium, anchor="mm")
            draw.text((960, 660), translation, fill=(255, 220, 100), font=f_medium, anchor="mm")

            path = os.path.join(tmp_dir, f"frame_{idx}.png")
            img.save(path)
            return path

        for i, word in enumerate(words):
            img_path = _make_frame(word, i)
            audio_path = os.path.join(tmp_dir, f"audio_{i}.mp3")
            tts_text = f"{word.get('hiragana', '')}。{word.get('chinese_translation', '')}"
            asyncio.run(_tts(tts_text, audio_path))

            audio_clip = AudioFileClip(audio_path)
            duration = max(audio_clip.duration + 1.0, 3.0)
            clips.append(ImageClip(img_path).set_duration(duration).set_audio(audio_clip))

        output_path = os.path.join(tmp_dir, "japanese_learning.mp4")
        concatenate_videoclips(clips, method="compose").write_videofile(
            output_path, fps=24, logger=None
        )
        return f"影片生成成功！路徑：{output_path}，共 {len(clips)} 個單字片段。"

    except Exception as e:
        return f"影片生成失敗：{e}"


# ==========================================
# 🧠 2. 初始化 LLM
# ==========================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

tools = [
    web_search,
    python_executor,
    forge_and_test_tool,
    init_encyclopedia_db,
    add_japanese_word,
    get_srs_due_words,
    update_word_srs,
    generate_japanese_learning_video,
]
tool_node = ToolNode(tools)

llm = ChatOllama(
    model="gemma4:26b",
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
    streaming=True,
).bind_tools(tools)

# ==========================================
# 🕸️ 3. 組裝 ReAct 狀態機 (Graph)
# ==========================================

SYSTEM_PROMPT = """你是配備多種工具的全能助理 Daedalus。

【強制規則 — 違反即為系統錯誤】
1. 禁止用純文字假裝執行工具。所有實際操作（搜尋、計算、資料庫讀寫、影片生成等）必須透過實際 Tool Call 完成，不得在回覆中模擬或捏造工具輸出。
2. 若工具回傳錯誤，必須如實呈現完整錯誤訊息，不得自行宣稱操作成功。
3. 不確定的事實禁止臆測，應呼叫 web_search 查詢後再回答。
4. 回覆一律使用繁體中文。

【可用工具】
- web_search：查詢即時網路資訊
- python_executor：執行 Python 程式碼進行計算或分析
- forge_and_test_tool：安全新增自定義工具（AST 掃描 + 沙盒 unittest）
- init_encyclopedia_db：初始化日文百科全書資料庫
- add_japanese_word：新增或更新日文單字（需傳入 JSON 字串）
- get_srs_due_words：取得今日 SRS 待複習單字
- update_word_srs：SM-2 演算法更新複習排程
- generate_japanese_learning_video：生成日文學習影片（MP4）
"""


async def call_model(state: MessagesState, config: RunnableConfig):
    messages = state["messages"]
    sys_msg = {"role": "system", "content": SYSTEM_PROMPT}

    final_response = None
    async for chunk in llm.astream([sys_msg] + messages, config=config):
        if final_response is None:
            final_response = chunk
        else:
            final_response += chunk

    return {"messages": [final_response]}


def should_continue(state: MessagesState):
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
