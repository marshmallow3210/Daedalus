from langchain_core.runnables import RunnableConfig
import os
import sys

# Ensure the project directory is always in sys.path regardless of how
# Chainlit sets the working directory at runtime.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import encyclopedia  # noqa: E402

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode

# ==========================================
# 🛠️ Tools
# ==========================================

@tool
def web_search(query: str) -> str:
    """搜尋最新網路資訊、日文單字列表、JLPT 教材、即時新聞等。
    優先使用 DuckDuckGo 真實搜尋；套件缺失時回傳提示。
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=8))
        if not results:
            return f"沒有找到「{query}」的搜尋結果。"
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] {r.get('title', '')}\n"
                f"{r.get('body', '')}\n"
                f"來源：{r.get('href', '')}"
            )
        return "\n\n".join(lines)
    except ImportError:
        return (
            f"【提示】duckduckgo-search 未安裝，無法進行真實搜尋。"
            f"請在 requirements.txt 加入 duckduckgo-search 並重建映像。"
        )
    except Exception as e:
        return f"搜尋失敗：{e}"


@tool
def fetch_web_page(url: str) -> str:
    """取得指定網頁的純文字內容（HTML 標籤已移除），最多回傳 4000 字。
    適合用於擷取日文單字教材網頁的詞彙資料。
    """
    try:
        import urllib.request
        from html.parser import HTMLParser

        class _Extractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.chunks = []
                self._skip = False

            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style", "nav", "footer", "head"):
                    self._skip = True

            def handle_endtag(self, tag):
                if tag in ("script", "style", "nav", "footer", "head"):
                    self._skip = False

            def handle_data(self, data):
                if not self._skip and data.strip():
                    self.chunks.append(data.strip())

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        parser = _Extractor()
        parser.feed(html)
        text = "\n".join(parser.chunks)
        if len(text) > 4000:
            text = text[:4000] + "\n…（內容已截斷）"
        return text
    except Exception as e:
        return f"無法取得網頁：{e}"


@tool
def python_executor(code: str) -> str:
    """當需要進行數學計算、數據分析、邏輯推導或執行 Python 程式碼時使用。"""
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = buf = StringIO()
    try:
        exec(code, {"__builtins__": __import__("builtins")}, {})
        sys.stdout = old_stdout
        output = buf.getvalue()
        return f"執行成功！\n{output}" if output else "執行成功（無輸出）。"
    except Exception as e:
        sys.stdout = old_stdout
        return f"執行出錯：{e}"


@tool
def forge_and_test_tool(tool_code: str, test_code: str) -> str:
    """安全新增自定義工具。
    流程：AST 靜態掃描（攔截 exec/eval/open/import）→ 記憶體沙盒 unittest →
    全數通過才寫入 /app/custom_tools.py → importlib 動態 reload。
    """
    import ast
    import unittest
    import importlib
    from io import StringIO

    BLOCKED_CALLS = {"exec", "eval", "open", "__import__"}

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

    custom_tools_path = os.getenv("CUSTOM_TOOLS_PATH", "/app/custom_tools.py")
    try:
        with open(custom_tools_path, "a", encoding="utf-8") as f:
            f.write("\n\n" + tool_code)
    except Exception as e:
        return f"寫入 custom_tools.py 失敗：{e}"

    try:
        import custom_tools
        importlib.reload(custom_tools)
        return f"工具已成功寫入並動態載入！\n{buf.getvalue()}"
    except Exception as e:
        return f"動態 reload 失敗：{e}"


@tool
def init_encyclopedia_db() -> str:
    """初始化（或升級）日文百科全書 SQLite 資料庫。"""
    try:
        encyclopedia.init_db()
        return "日文百科全書資料庫初始化成功。"
    except Exception as e:
        return f"初始化失敗：{e}"


@tool
def add_japanese_word(word_json: str) -> str:
    """將日文單字寫入百科全書，若 hiragana 已存在則自動更新。
    word_json 為 JSON 字串。必填：hiragana。
    選填：katakana、kanji、romaji、chinese_translation、etymology、
          jlpt_level、part_of_speech、tags、example_sentence、
          example_sentence_reading、emoji。
    emoji 範例：「🐱」用於貓（ねこ）。
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
    """SM-2 演算法更新複習排程。quality：0（完全忘記）到 5（完美記憶）。"""
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
def get_video_candidate_words(limit: int = 50, jlpt_level: str = "") -> str:
    """從百科全書取得尚未出現在任何影片中的單字清單（JSON 格式）。
    用於在生成影片前確認有哪些新單字可用。
    limit: 最多返回幾個（預設 50）
    jlpt_level: 篩選等級，如 N5 / N4，留空則不篩選
    """
    import json
    try:
        words = encyclopedia.get_words_for_video(
            limit=limit,
            jlpt_level=jlpt_level or None,
        )
        if not words:
            return (
                "百科全書中目前沒有待拍攝的單字。"
                "請先使用 web_search + add_japanese_word 新增單字。"
            )
        return json.dumps(words, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"查詢失敗：{e}"


@tool
def generate_japanese_learning_video(
    batch_json: str = "",
    batch_id: str = "",
    jlpt_level: str = "",
) -> str:
    """生成 1920x1080 日文單字教學影片（MP4）。

    影片格式（每個單字）：
      1. 顯示單字卡（emoji + 日文 + 中文翻譯 + 詞源）
      2. 音訊：日文 TTS → 中文 TTS → 日文 TTS
      3. 若有詞源（etymology），播放詞源說明音訊

    參數：
      batch_json  — JSON 陣列（每項含 hiragana 等欄位）；留空則自動從百科全書
                    取出 50 個尚未拍攝的單字。
      batch_id    — 影片批次 ID（留空自動生成）；完成後會寫回百科全書。
      jlpt_level  — 留空=不篩選；N5/N4 等值在自動取詞時生效。

    若 moviepy / pillow / edge-tts 未安裝，回傳 Dry-run 訊息而不崩潰。
    """
    import json
    import threading
    import asyncio
    import tempfile
    import uuid

    # ── 1. 決定單字來源 ───────────────────────────────────────────
    if batch_json.strip():
        try:
            words = json.loads(batch_json)
        except json.JSONDecodeError as e:
            return f"JSON 解析失敗：{e}"
    else:
        words = encyclopedia.get_words_for_video(
            limit=50,
            jlpt_level=jlpt_level or None,
        )

    if not isinstance(words, list) or len(words) == 0:
        return "錯誤：沒有可用的單字。請先使用 add_japanese_word 或 web_search 新增單字。"

    # ── 2. 套件檢查 ───────────────────────────────────────────────
    missing = []
    for pkg, imp in [("pillow", "PIL.Image"), ("moviepy", "moviepy"), ("edge-tts", "edge_tts")]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)

    if missing:
        preview = [
            f"{w.get('hiragana','?')} {w.get('emoji','')} ({w.get('chinese_translation','?')})"
            for w in words[:5]
        ]
        return (
            f"[Dry-run] 缺少套件：{', '.join(missing)}。影片未實際生成。\n"
            f"批次共 {len(words)} 個單字，前 5 個：{', '.join(preview)}"
        )

    # ── 3. 匯入實際套件 ───────────────────────────────────────────
    try:
        from PIL import Image, ImageDraw, ImageFont
        from moviepy import (
            ImageClip, AudioFileClip,
            concatenate_videoclips, concatenate_audioclips,
        )
        import edge_tts
    except ImportError as e:
        return f"套件匯入失敗：{e}"

    # ── 4. 輔助工具 ───────────────────────────────────────────────
    def _run_coro(coro):
        """Run an asyncio coroutine in a dedicated thread+loop (safe from LangGraph's loop)."""
        result = [None]
        exc = [None]

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result[0] = loop.run_until_complete(coro)
            except Exception as e:
                exc[0] = e
            finally:
                loop.close()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=120)
        if exc[0]:
            raise exc[0]
        return result[0]

    async def _tts(text: str, voice: str, path: str):
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(path)

    def _find_font(size: int) -> ImageFont.FreeTypeFont:
        candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJKjp-Regular.otf",
            "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",
            "/usr/share/fonts/truetype/fonts-ipafont-gothic/ipagp.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansCJKjp-Regular.otf",
        ]
        for p in candidates:
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, size)
                except Exception:
                    continue
        return ImageFont.load_default()

    def _make_frame(word: dict, tmp_dir: str, idx: int) -> str:
        W, H = 1920, 1080
        BG      = (12, 12, 35)
        GOLD    = (255, 210, 80)
        WHITE   = (240, 240, 240)
        LBLUE   = (160, 190, 255)
        GRAY    = (160, 160, 160)
        ACCENT  = (80, 130, 255)

        img  = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(img)

        # accent bar top
        draw.rectangle([(0, 0), (W, 6)], fill=ACCENT)

        # JLPT badge
        jlpt = word.get("jlpt_level", "")
        if jlpt:
            f_badge = _find_font(28)
            draw.rounded_rectangle([(W - 130, 20), (W - 20, 60)], radius=8, fill=ACCENT)
            draw.text((W - 75, 40), jlpt, fill=WHITE, font=f_badge, anchor="mm")

        # emoji (top-left area)
        emoji_str = word.get("emoji", "")
        if emoji_str:
            f_emoji = _find_font(110)
            draw.text((100, 120), emoji_str, fill=WHITE, font=f_emoji, anchor="lm")

        # main word (kanji preferred, else katakana, else hiragana)
        main = word.get("kanji") or word.get("katakana") or word.get("hiragana", "")
        f_main = _find_font(130)
        draw.text((W // 2, 300), main, fill=WHITE, font=f_main, anchor="mm")

        # reading (hiragana) — only if main is not already hiragana
        reading = word.get("hiragana", "")
        if reading and reading != main:
            f_read = _find_font(60)
            draw.text((W // 2, 410), f"（{reading}）", fill=LBLUE, font=f_read, anchor="mm")

        # divider
        draw.rectangle([(W // 2 - 200, 450), (W // 2 + 200, 453)], fill=ACCENT)

        # Chinese translation
        cn = word.get("chinese_translation", "")
        f_cn = _find_font(90)
        draw.text((W // 2, 560), cn, fill=GOLD, font=f_cn, anchor="mm")

        # etymology
        etym = word.get("etymology", "")
        if etym:
            f_etym = _find_font(36)
            # wrap long etymology
            max_chars = 40
            lines = [etym[i:i+max_chars] for i in range(0, min(len(etym), max_chars*3), max_chars)]
            for li, line in enumerate(lines[:3]):
                draw.text((W // 2, 680 + li * 50), line, fill=GRAY, font=f_etym, anchor="mm")

        # accent bar bottom
        draw.rectangle([(0, H - 6), (W, H)], fill=ACCENT)

        path = os.path.join(tmp_dir, f"frame_{idx:03d}.png")
        img.save(path, "PNG")
        return path

    # ── 5. 生成影片 ───────────────────────────────────────────────
    try:
        tmp_dir = tempfile.mkdtemp()
        vid_id = batch_id or uuid.uuid4().hex[:8]
        clips = []

        JP_VOICE = "ja-JP-NanamiNeural"
        CN_VOICE = "zh-TW-HsiaoYuNeural"

        for idx, word in enumerate(words):
            hiragana   = word.get("hiragana", "")
            cn_text    = word.get("chinese_translation", "")
            etym_text  = word.get("etymology", "")
            jp_say     = word.get("katakana") or word.get("kanji") or hiragana

            # TTS paths
            jp1_path  = os.path.join(tmp_dir, f"{idx:03d}_jp1.mp3")
            cn_path   = os.path.join(tmp_dir, f"{idx:03d}_cn.mp3")
            jp2_path  = os.path.join(tmp_dir, f"{idx:03d}_jp2.mp3")
            etym_path = os.path.join(tmp_dir, f"{idx:03d}_etym.mp3") if etym_text else None

            _run_coro(_tts(jp_say, JP_VOICE, jp1_path))
            if cn_text:
                _run_coro(_tts(cn_text, CN_VOICE, cn_path))
            _run_coro(_tts(jp_say, JP_VOICE, jp2_path))
            if etym_text:
                _run_coro(_tts(etym_text, JP_VOICE, etym_path))

            # Combine audio: JP1 → CN → JP2 → etymology
            audio_parts = [AudioFileClip(jp1_path)]
            if cn_text and os.path.exists(cn_path):
                audio_parts.append(AudioFileClip(cn_path))
            audio_parts.append(AudioFileClip(jp2_path))
            if etym_path and os.path.exists(etym_path):
                audio_parts.append(AudioFileClip(etym_path))

            full_audio = concatenate_audioclips(audio_parts)
            duration   = full_audio.duration + 0.5  # brief pause after each card

            # Image frame
            frame_path = _make_frame(word, tmp_dir, idx)
            clip = ImageClip(frame_path).with_duration(duration).with_audio(full_audio)
            clips.append(clip)

        if not clips:
            return "批次為空，未生成影片。"

        # Output path — use /app/videos/ so it can be volume-mounted
        out_dir = "/app/videos"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"jp_lesson_{vid_id}.mp4")

        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(out_path, fps=24, logger=None, audio_codec="aac")

        # ── 6. 寫回百科全書 ────────────────────────────────────────
        filmed = [w["hiragana"] for w in words if w.get("hiragana")]
        updated = encyclopedia.mark_words_in_video(filmed, vid_id)

        return (
            f"影片生成成功！\n"
            f"路徑：{out_path}\n"
            f"批次 ID：{vid_id}\n"
            f"共 {len(clips)} 個單字片段，{updated} 筆已標記為已拍攝。"
        )

    except Exception as e:
        return f"影片生成失敗：{e}"


# ==========================================
# 🧠 LLM + Graph
# ==========================================

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

tools = [
    web_search,
    fetch_web_page,
    python_executor,
    forge_and_test_tool,
    init_encyclopedia_db,
    add_japanese_word,
    get_srs_due_words,
    update_word_srs,
    get_video_candidate_words,
    generate_japanese_learning_video,
]
tool_node = ToolNode(tools)

llm = ChatOllama(
    model="gemma4:26b",
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
    streaming=True,
).bind_tools(tools)

SYSTEM_PROMPT = """你是配備多種工具的全能助理 Daedalus，專精日文學習影片製作。

【強制規則 — 違反即為系統錯誤】
1. 禁止用純文字假裝執行工具。所有操作必須透過實際 Tool Call 完成，不得模擬或捏造輸出。
2. 若工具回傳錯誤，必須如實呈現完整錯誤訊息，不得宣稱操作成功。
3. 不確定的事實禁止臆測，應呼叫 web_search 查詢後再回答。
4. 回覆一律使用繁體中文。

【日文影片製作標準流程】
1. 呼叫 get_video_candidate_words 確認百科全書中有哪些單字尚未拍攝
2. 若單字不足，呼叫 web_search 搜尋（如「JLPT N5 常用日文單字 料理 英日中對照」）
3. 解析搜尋結果，為每個單字呼叫 add_japanese_word（含 emoji、etymology、jlpt_level）
4. 呼叫 generate_japanese_learning_video 生成影片（留空 batch_json 自動取詞）
5. 影片完成後，系統自動將這批單字標記為已拍攝，下次不會重複

【可用工具】
- web_search：真實 DuckDuckGo 網路搜尋
- fetch_web_page：取得特定網頁純文字（適合擷取詞彙列表）
- python_executor：執行 Python 計算/分析
- forge_and_test_tool：安全新增自定義工具
- init_encyclopedia_db：初始化資料庫
- add_japanese_word：新增/更新單字（支援 emoji、etymology、tags、example_sentence 等）
- get_srs_due_words：今日 SRS 待複習
- update_word_srs：SM-2 更新複習排程
- get_video_candidate_words：查詢尚未拍攝的單字
- generate_japanese_learning_video：生成教學影片（日→中→日 TTS + 詞源字幕）
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
    last = state["messages"][-1]
    return "tools" if last.tool_calls else "end"


workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")

app = workflow.compile()
