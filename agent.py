from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import json
import os
import re
import sys
import uuid
from typing import Literal, Optional
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import encyclopedia  # noqa: E402

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# ──────────────────────────────────────────────────────────────
# Pydantic models for structured reporting
# ──────────────────────────────────────────────────────────────

class TaskCompletionReport(BaseModel):
    task_id: str = ""
    task_description: str
    modified_files: list[str]
    change_summary: str
    test_executed: bool
    test_passed: Optional[bool] = None
    risk_level: Literal["low", "medium", "high"]
    risk_reason: str
    tool_calls_made: list[str]


class IssueItem(BaseModel):
    severity: Literal["blocker", "major", "minor"]
    category: Literal["correctness", "security", "edge_case", "test", "scope"]
    description: str
    location: Optional[str] = None
    suggestion: Optional[str] = None


class ReviewResult(BaseModel):
    verdict: Literal["approve", "needs_changes"]
    reviewer_risk: Literal["low", "medium", "high"]
    issues: list[IssueItem]
    approved_aspects: list[str]
    one_line_summary: str


# ──────────────────────────────────────────────────────────────
# Extended graph state
# ──────────────────────────────────────────────────────────────

class DaedalusState(MessagesState):
    task_report: Optional[dict]       # TaskCompletionReport.model_dump() or None
    review_result: Optional[dict]     # ReviewResult.model_dump() or None
    retry_count: int                  # Reviewer-driven retry counter
    schema_error_count: int           # submit_task_completion parse failures
    original_task: str                # snapshot of task_description at submit time
    task_reporter_status: str         # "ok" | "retry" | "escalate" | ""
    retry_messages: Optional[list]    # selective-amnesia context from context_surgeon; cleared by call_model
    total_iterations: int             # Coder call_model runs since last successful submit; hard-capped at 6
    last_task_iterations: int         # total_iterations snapshot at submit time (display only; survives the reset)


SUBMIT_TOOL_NAME = "submit_task_completion"


# ──────────────────────────────────────────────────────────────
# Tools — safe (no human review needed)
# ──────────────────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """搜尋最新網路資訊、日文詞頻研究、JLPT 教材等。使用真實 DuckDuckGo 搜尋。"""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=8))
        if not results:
            return f"沒有找到「{query}」的搜尋結果。"
        return "\n\n".join(
            f"[{i}] {r.get('title','')}\n{r.get('body','')}\n來源：{r.get('href','')}"
            for i, r in enumerate(results, 1)
        )
    except ImportError:
        return "duckduckgo-search 未安裝，請重建 Docker 映像。"
    except Exception as e:
        return f"搜尋失敗：{e}"


@tool
def fetch_web_page(url: str) -> str:
    """取得指定網頁的純文字內容（最多 4000 字），用於擷取日文詞彙資料。"""
    try:
        import urllib.request
        from html.parser import HTMLParser

        class _X(HTMLParser):
            def __init__(self):
                super().__init__()
                self.chunks, self._skip = [], False
            def handle_starttag(self, tag, attrs):
                self._skip = tag in ("script", "style", "nav", "footer", "head")
            def handle_endtag(self, tag):
                if tag in ("script", "style", "nav", "footer", "head"):
                    self._skip = False
            def handle_data(self, data):
                if not self._skip and data.strip():
                    self.chunks.append(data.strip())

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            html = r.read().decode("utf-8", errors="replace")
        p = _X(); p.feed(html)
        text = "\n".join(p.chunks)
        return text[:4000] + "\n…（已截斷）" if len(text) > 4000 else text
    except Exception as e:
        return f"無法取得網頁：{e}"


@tool
def python_executor(code: str) -> str:
    """執行 Python 程式碼進行計算或分析。"""
    import sys
    from io import StringIO
    old = sys.stdout; sys.stdout = buf = StringIO()
    try:
        exec(code, {"__builtins__": __import__("builtins")}, {})
        sys.stdout = old
        out = buf.getvalue()
        return f"執行成功！\n{out}" if out else "執行成功（無輸出）。"
    except Exception as e:
        sys.stdout = old
        return f"執行出錯：{e}"


@tool
def forge_and_test_tool(tool_code: str, test_code: str) -> str:
    """安全新增自定義工具：AST 掃描 → 沙盒 unittest → 寫入 custom_tools.py → reload。
    tool_code 規則：
      - 禁止含 import / exec / eval / open / __builtins__ / getattr 等危險操作
      - 禁止使用 @tool decorator（沙盒不提供 LangChain 環境，只寫純 def 函式）
    回傳中含三個機器可讀標籤：
      [AST_RESULT]PASS|FAIL[/AST_RESULT]
      [TEST_RESULT]PASS|FAIL|SKIP[/TEST_RESULT]
      [DISK_WRITE_RESULT]SUCCESS|FAIL|SKIP[/DISK_WRITE_RESULT]
    modified_files 欄位只能在 DISK_WRITE_RESULT=SUCCESS 時填入對應檔案路徑。
    """
    import ast, unittest, importlib
    from io import StringIO

    # ── Stage 1: AST Security Scan ──────────────────────────────────────────
    # BLOCKED_CALLS: direct calls (func.id) or method calls (func.attr) both checked
    BLOCKED_CALLS = {"exec", "eval", "open", "__import__", "getattr", "setattr",
                     "import_module", "compile", "breakpoint"}
    # BLOCKED_NAMES: bare Name nodes that give access to Python internals
    BLOCKED_NAMES = {"__builtins__", "__globals__", "__locals__",
                     "__import__", "__loader__", "__spec__"}

    def _ast_fail(reason: str) -> str:
        return (
            f"[AST_RESULT]FAIL: {reason}[/AST_RESULT]\n"
            f"[TEST_RESULT]SKIP[/TEST_RESULT]\n"
            f"[DISK_WRITE_RESULT]SKIP[/DISK_WRITE_RESULT]\n"
            f"安全掃描失敗：{reason}"
        )

    try:
        tree = ast.parse(tool_code)
    except SyntaxError as e:
        return _ast_fail(f"AST 解析失敗：{e}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return _ast_fail("tool_code 禁止使用 import 陳述式")
        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            return _ast_fail(f"禁止存取受限識別字 {node.id!r}（間接 import 途徑）")
        if isinstance(node, ast.Call):
            call_name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if call_name in BLOCKED_CALLS:
                return _ast_fail(f"禁止呼叫 {call_name}()")

    _ast_tag = "[AST_RESULT]PASS: 安全掃描通過[/AST_RESULT]"

    # ── Stage 2: Sandbox Unit Test ──────────────────────────────────────────
    # No-op `tool` handles both @tool (bare) and @tool("desc")/(@tool(return_direct=True) forms.
    # lambda f: f only handles bare @tool; callable-check version handles both.
    def _noop_tool(f=None, **_kw):
        if callable(f):
            return f        # @tool  — decorator applied directly to function
        return lambda g: g  # @tool(...) — called with args, returns decorator

    sb = {"__builtins__": __import__("builtins"), "tool": _noop_tool}
    try:
        exec(tool_code, sb)
        exec(test_code, sb)
    except Exception as e:
        return (
            f"{_ast_tag}\n"
            f"[TEST_RESULT]FAIL: 沙盒載入失敗：{e}[/TEST_RESULT]\n"
            f"[DISK_WRITE_RESULT]SKIP[/DISK_WRITE_RESULT]\n"
            f"沙盒載入失敗：{e}"
        )

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for obj in sb.values():
        if isinstance(obj, type) and issubclass(obj, unittest.TestCase) and obj is not unittest.TestCase:
            suite.addTests(loader.loadTestsFromTestCase(obj))

    buf    = StringIO()
    result = unittest.TextTestRunner(stream=buf, verbosity=2).run(suite)
    if not result.wasSuccessful():
        return (
            f"{_ast_tag}\n"
            f"[TEST_RESULT]FAIL: 單元測試失敗（{result.testsRun} tests, "
            f"{len(result.failures)} failures, {len(result.errors)} errors）[/TEST_RESULT]\n"
            f"[DISK_WRITE_RESULT]SKIP[/DISK_WRITE_RESULT]\n"
            f"單元測試失敗，拒絕寫入：\n{buf.getvalue()}"
        )

    _test_tag = f"[TEST_RESULT]PASS: {result.testsRun} 個測試通過[/TEST_RESULT]"

    # ── Stage 3: Disk Write ─────────────────────────────────────────────────
    path = os.getenv("CUSTOM_TOOLS_PATH", "/app/custom_tools.py")

    # Strip @tool decorators before writing: custom_tools.py is plain Python,
    # no LangChain runtime, so @tool would crash importlib.reload().
    clean_code = re.sub(r'^[ \t]*@tool\b[^\n]*\n?', '', tool_code, flags=re.MULTILINE).strip()

    def _upsert(file_path: str, new_code: str) -> None:
        """Replace existing top-level functions with same name; append if new."""
        try:
            new_tree  = ast.parse(new_code)
            new_names = {n.name for n in new_tree.body if isinstance(n, ast.FunctionDef)}
        except Exception:
            new_names = set()

        if not new_names:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write("\n\n" + new_code)
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                existing = f.read()
            existing_tree = ast.parse(existing)
        except Exception:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write("\n\n" + new_code)
            return

        # Collect line ranges of existing functions to replace (0-indexed).
        remove_lines: set = set()
        for node in existing_tree.body:
            if not (isinstance(node, ast.FunctionDef) and node.name in new_names):
                continue
            first = (min(d.lineno for d in node.decorator_list)
                     if node.decorator_list else node.lineno) - 1
            remove_lines.update(range(first, node.end_lineno))

        if not remove_lines:
            # No existing definition — just append.
            with open(file_path, "a", encoding="utf-8") as f:
                f.write("\n\n" + new_code)
            return

        lines       = existing.splitlines(keepends=True)
        kept        = [l for i, l in enumerate(lines) if i not in remove_lines]
        new_content = "".join(kept).rstrip() + "\n\n\n" + new_code.strip() + "\n"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

    try:
        _upsert(path, clean_code)
    except Exception as e:
        return (
            f"{_ast_tag}\n"
            f"{_test_tag}\n"
            f"[DISK_WRITE_RESULT]FAIL: 寫入磁碟失敗：{e}[/DISK_WRITE_RESULT]\n"
            f"⚠️ AST + 測試均通過，但磁碟寫入失敗（{e}）。\n"
            f"modified_files 必須傳 []，不得填入 {path}。"
        )

    try:
        import custom_tools
        importlib.reload(custom_tools)
        return (
            f"{_ast_tag}\n"
            f"{_test_tag}\n"
            f"[DISK_WRITE_RESULT]SUCCESS: 工具已寫入 {path} 並動態載入[/DISK_WRITE_RESULT]\n"
            f"工具已成功寫入並動態載入！\n{buf.getvalue()}"
        )
    except Exception as e:
        # File written but reload failed — disk state is still SUCCESS
        return (
            f"{_ast_tag}\n"
            f"{_test_tag}\n"
            f"[DISK_WRITE_RESULT]SUCCESS: 工具已寫入 {path}（reload 失敗：{e}）[/DISK_WRITE_RESULT]\n"
            f"工具已寫入磁碟，但動態 reload 失敗（{e}）。需要手動重啟載入。"
        )


@tool
def init_encyclopedia_db() -> str:
    """初始化（或升級）日文百科全書資料庫，建立 japanese_encyclopedia 與 video_batches 表。"""
    try:
        encyclopedia.init_db()
        return "資料庫初始化成功（含 video_batches 表）。"
    except Exception as e:
        return f"初始化失敗：{e}"


@tool
def add_japanese_word(word_json: str) -> str:
    """將日文單字寫入百科全書，若 hiragana 已存在則自動更新。
    word_json 為 JSON 字串。必填：hiragana。
    選填：katakana、kanji、romaji、chinese_translation、etymology（字源說明）、
          jlpt_level（N5/N4/N3/N2/N1）、part_of_speech、tags、
          example_sentence、example_sentence_reading、emoji。
    判斷是否需要字源解釋的標準：
      - 複合詞（可拆解為有意義的詞素）→ 需要
      - 含外來語音譯成分 → 需要
      - 慣用語或固定搭配 → 需要
      - 基礎詞彙（ありがとう 等）→ 不需要，etymology 留空

    etymology 格式規範（每段以「，」分隔）：
      每段必須遵守：「平假名或片假名（漢字）是中文意思」
      例：す（酢）是醋，し（飯）是飯，醋飯之意
      例：電（でん）是電力，車（しゃ）是車，電力驅動的車
      括弧內放漢字形，「是」後接繁體中文意思。
      最後可加一段總結（無需「是」結構），例：「醋飯之意」。
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
def get_video_candidate_words(limit: int = 50, jlpt_level: str = "") -> str:
    """從百科全書取得尚未出現在任何影片中的單字（JSON 格式）。
    生成影片前必須先呼叫此工具確認有足夠單字。
    """
    import json
    try:
        words = encyclopedia.get_words_for_video(limit=limit, jlpt_level=jlpt_level or None)
        if not words:
            return "百科全書中目前沒有待拍攝的單字。請先用 web_search + add_japanese_word 新增單字。"
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

    每個單字的影片格式：
      1. 單字卡（emoji + 日文 + 中文 + 字源說明）
      2. 音訊：日文 TTS → 中文 TTS → 日文 TTS（重用同一音檔）→ 字源說明 TTS（若有）
      3. 逐字生成片段並立即落地，避免記憶體 OOM

    流程：建立 video_batches 紀錄 → 逐字合成 → 串接完整影片 →
    更新狀態為 uploading → 回傳路徑（下一步由 upload_to_youtube 上傳）

    batch_json  — JSON 陣列；留空則從百科全書自動取 50 個未拍單字
    batch_id    — 批次 ID；留空自動產生
    jlpt_level  — N5/N4/N3 等，自動取詞時篩選用
    """
    import json, threading, asyncio, tempfile, shutil, uuid, time
    from io import StringIO

    # ── 1. 決定單字來源 ────────────────────────────────────────
    if batch_json.strip():
        try:
            words = json.loads(batch_json)
        except json.JSONDecodeError as e:
            return f"JSON 解析失敗：{e}"
    else:
        words = encyclopedia.get_words_for_video(limit=50, jlpt_level=jlpt_level or None)

    if not isinstance(words, list) or not words:
        return "沒有可用的單字。請先用 web_search + add_japanese_word 新增單字。"

    # ── 2. 套件檢查 ────────────────────────────────────────────
    missing = []
    for pkg, imp in [("pillow", "PIL.Image"), ("moviepy", "moviepy"), ("edge-tts", "edge_tts")]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        preview = [f"{w.get('hiragana','?')}（{w.get('chinese_translation','?')}）" for w in words[:5]]
        return (
            f"[Dry-run] 缺少套件：{', '.join(missing)}。\n"
            f"批次共 {len(words)} 個單字，前 5：{', '.join(preview)}"
        )

    # ── 3. 初始化 ──────────────────────────────────────────────
    from PIL import Image, ImageDraw, ImageFont
    try:
        from moviepy import (
            ImageClip, AudioFileClip, VideoFileClip,
            concatenate_videoclips, concatenate_audioclips,
        )
    except ImportError:
        from moviepy.editor import (
            ImageClip, AudioFileClip, VideoFileClip,
            concatenate_videoclips, concatenate_audioclips,
        )
    import edge_tts

    vid_id   = batch_id or uuid.uuid4().hex[:8]
    out_dir  = "/app/videos"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"jp_lesson_{vid_id}.mp4")
    tmp_dir  = tempfile.mkdtemp(prefix=f"daedalus_{vid_id}_")

    for sub in ("frames", "audio", "segments"):
        os.makedirs(os.path.join(tmp_dir, sub), exist_ok=True)

    # Record batch as generating
    try:
        encyclopedia.create_batch(vid_id, len(words), out_path)
    except Exception:
        pass  # DB may not be initialised yet; don't abort video generation

    JP_VOICE = "ja-JP-NanamiNeural"
    CN_VOICE = "zh-TW-HsiaoYuNeural"
    DAILY_QUOTA_LIMIT = 5

    # ── 4. Helpers ─────────────────────────────────────────────
    def _run_coro(coro):
        exc = [None]
        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(coro)
            except Exception as e:
                exc[0] = e
            finally:
                loop.close()
        t = threading.Thread(target=_run, daemon=True)
        t.start(); t.join(timeout=120)
        if exc[0]:
            raise exc[0]

    def _tts_with_retry(text: str, voice: str, path: str, retries: int = 3) -> bool:
        async def _gen():
            await edge_tts.Communicate(text, voice).save(path)
        for attempt in range(retries):
            try:
                _run_coro(_gen())
                return True
            except Exception:
                if attempt < retries - 1:
                    time.sleep(2)
        # Fallback: write silent mp3 (minimal valid mp3 header)
        try:
            import struct
            silent = b'\xff\xfb\x90\x00' + b'\x00' * 413  # ~26ms silent frame
            with open(path, "wb") as f:
                f.write(silent * 10)
        except Exception:
            pass
        return False

    def _find_font(size: int) -> ImageFont.FreeTypeFont:
        for p in [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJKjp-Regular.otf",
            "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",
        ]:
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, size)
                except Exception:
                    pass
        return ImageFont.load_default()

    _EMOJI_FONT_PATHS = [
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        "/usr/share/fonts/noto/NotoColorEmoji.ttf",
    ]
    # Japanese traditional color palette for category badges
    _BADGE_COLORS = {
        "動物": (183,  65,  45),   # 朱色 vermillion
        "食物": (155, 120,  45),   # 金色 gold
        "飲料": ( 49,  79, 113),   # 藍色 indigo
        "自然": ( 80, 115,  70),   # 萌葱色 green
        "交通": ( 90,  75, 110),   # 江戶紫 purple
        "物品": (120,  90,  55),   # 茶色 brown
        "身體": (160,  85,  90),   # 紅梅色 pink
    }

    def _draw_emoji_on(base_img, emoji_str, x, y, size=130, tag=""):
        """Render emoji with NotoColorEmoji (embedded_color=True), fall back to a colored badge."""
        for ep in _EMOJI_FONT_PATHS:
            if not os.path.exists(ep):
                continue
            try:
                efont  = ImageFont.truetype(ep, 109)   # NotoColorEmoji native bitmap size
                canvas = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
                cdraw  = ImageDraw.Draw(canvas)
                cdraw.text((20, 20), emoji_str, font=efont, embedded_color=True)
                bbox = canvas.getbbox()
                if bbox and (bbox[2] - bbox[0]) > 10:
                    cropped = canvas.crop(bbox)
                    ow, oh  = cropped.size
                    nw      = max(1, int(ow * size / oh))
                    canvas  = cropped.resize((nw, size), Image.LANCZOS)
                    base_img.paste(canvas, (x + (size - nw) // 2, y), canvas)
                    return
            except Exception:
                pass
        # Fallback: colored circle with category label
        color = _BADGE_COLORS.get(tag, (80, 130, 255))
        draw  = ImageDraw.Draw(base_img)
        r     = size // 2
        cx_c, cy_c = x + r, y + r
        draw.ellipse([(cx_c - r, cy_c - r), (cx_c + r, cy_c + r)], fill=color)
        label = tag[:2] if tag else (emoji_str[:1] if emoji_str else "?")
        draw.text((cx_c, cy_c), label, fill=(255, 255, 255),
                  font=_find_font(r), anchor="mm")

    def _make_frame(word: dict, idx: int) -> str:
        import re as _re
        W, H = 1920, 1080

        BG        = (252, 246, 235)
        ORANGE    = (218, 108,  48)   # 橙色 warm Japanese orange
        CHARCOAL  = ( 32,  26,  18)
        SLATE     = ( 95, 105, 118)
        VERMIL    = (183,  65,  45)   # 中文翻譯色
        WARM_GRAY = (135, 123, 108)

        img  = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(img)

        # 最外圍四邊橘色邊框
        BORDER = 10
        draw.rectangle([(0, 0),          (W, BORDER)],      fill=ORANGE)   # 上
        draw.rectangle([(0, H - BORDER), (W, H)],           fill=ORANGE)   # 下
        draw.rectangle([(0, 0),          (BORDER, H)],      fill=ORANGE)   # 左
        draw.rectangle([(W - BORDER, 0), (W, H)],           fill=ORANGE)   # 右

        # JLPT 徽章（右上角）
        jlpt = word.get("jlpt_level", "")
        if jlpt:
            draw.rounded_rectangle([(W - 152, 22), (W - 22, 64)], radius=6, fill=ORANGE)
            draw.text((W - 87, 43), jlpt, fill=(252, 246, 235), font=_find_font(28), anchor="mm")

        # ── Emoji：上下左右置中 ──────────────────────────────────
        EMOJI_SIZE = 260
        ex = W // 2 - EMOJI_SIZE // 2   # 水平置中
        ey = H // 2 - EMOJI_SIZE // 2   # 垂直置中（ey=410, 底=670）
        _draw_emoji_on(img, word.get("emoji", ""), ex, ey,
                       size=EMOJI_SIZE, tag=word.get("tags", ""))

        # ── 日文（漢字＋平假名ルビ逐字對齊，或純假名，Emoji 上方）──
        kanji_str = word.get("kanji", "")
        hira_str  = _re.sub(r"（[^）]*）", "", word.get("hiragana", "")).strip()
        kata_str  = word.get("katakana", "")

        def _is_kanji(c): return "一" <= c <= "鿿" or "㐀" <= c <= "䶿"
        def _is_kana(c):  return "ぁ" <= c <= "ヿ"

        # 比例分配演算法對特定詞產生錯誤對齊時的覆寫表
        _RUBY_OVERRIDE = {
            "でんわ":   [("電", "でん"), ("話", "わ")],
            "ひこうき": [("飛", "ひ"),   ("行", "こう"), ("機", "き")],
            "ぼうし":   [("帽", "ぼう"), ("子", "し")],
        }

        if kanji_str:
            kf = _find_font(148)
            rf = _find_font(58)
            if hira_str in _RUBY_OVERRIDE:
                assignments = _RUBY_OVERRIDE[hira_str]
            else:
                hlist = list(hira_str)
                klist = list(kanji_str)
                hi = 0
                assignments = []          # (char, ruby_hiragana)
                for i, ch in enumerate(klist):
                    if _is_kana(ch):      # 漢字串中已是假名，直接對齊跳過
                        eq = chr(ord(ch) - 0x60) if "ァ" <= ch <= "ヶ" else ch
                        while hi < len(hlist) and hlist[hi] != eq:
                            hi += 1
                        assignments.append((ch, ""))
                        if hi < len(hlist): hi += 1
                    else:                 # 漢字：掃到下一個假名確定讀音邊界
                        nxt = next((klist[j] for j in range(i+1, len(klist))
                                    if _is_kana(klist[j])), None)
                        st = hi
                        if nxt:
                            eq = chr(ord(nxt) - 0x60) if "ァ" <= nxt <= "ヶ" else nxt
                            while hi < len(hlist) and hlist[hi] != eq:
                                hi += 1
                        else:
                            rem_k = sum(1 for c in klist[i+1:] if _is_kanji(c))
                            hi = len(hlist) if rem_k == 0 else \
                                 hi + max(1, (len(hlist) - hi) // (rem_k + 1))
                        assignments.append((ch, "".join(hlist[st:hi])))
            # 計算每字寬度並置中
            try:
                ws = [max(10, int(kf.getlength(ch))) for ch, _ in assignments]
            except AttributeError:
                ws = [148] * len(assignments)
            total_w = sum(ws)
            x = W // 2 - total_w // 2
            KANJI_Y = 232
            RUBY_DY = 148 // 2 + 8 + 58 // 2   # 74 + 8 + 29 = 111
            for (ch, ruby), w in zip(assignments, ws):
                ccx = x + w // 2
                draw.text((ccx, KANJI_Y), ch, fill=CHARCOAL, font=kf, anchor="mm")
                if ruby:
                    draw.text((ccx, KANJI_Y - RUBY_DY), ruby, fill=SLATE, font=rf, anchor="mm")
                x += w
        else:
            draw.text((W // 2, 240), kata_str or hira_str,
                      fill=CHARCOAL, font=_find_font(148), anchor="mm")

        # ── 中文 + 字源（Emoji 下方，留寬鬆間距）──────────────
        cn = word.get("chinese_translation", "")
        draw.text((W // 2, 765), cn, fill=VERMIL, font=_find_font(82), anchor="mm")

        etym = word.get("etymology", "")
        if etym:
            etym_clean = _re.sub(r'\([A-Za-z ]+\)', '', etym).strip()
            # 移除全形括弧內的漢字（如 す（酢）→ す），並在「是」前加空白
            etym_clean = _re.sub(r'（[^）]*）', '', etym_clean).strip()
            etym_clean = _re.sub(r'([^\s])是', r'\1 是', etym_clean)
            parts = [p.strip() for p in _re.split(r'[，、]', etym_clean) if p.strip()]
            lines = ["；".join(parts)] if len(parts) <= 2 else parts[:3]
            f_e   = _find_font(30)
            for li, line in enumerate(lines):
                if len(line) > 48:
                    line = line[:47] + "…"
                draw.text((W // 2, 858 + li * 54), line, fill=WARM_GRAY, font=f_e, anchor="mm")

        path = os.path.join(tmp_dir, "frames", f"{idx:03d}.png")
        img.save(path, "PNG")
        return path

    # ── 5. Per-word segment generation ─────────────────────────
    segment_paths = []
    tts_failures  = []
    consecutive_fails = 0

    for idx, word in enumerate(words):
        import re as _re2b
        hiragana = _re2b.sub(r"（[^）]*）", "", word.get("hiragana", "")).strip()
        jp_say   = word.get("katakana") or hiragana or word.get("kanji") or ""
        cn_text  = word.get("chinese_translation", "")
        etym     = word.get("etymology", "")

        jp_path = os.path.join(tmp_dir, "audio", f"{idx:03d}_jp.mp3")
        cn_path = os.path.join(tmp_dir, "audio", f"{idx:03d}_cn.mp3")

        ok_jp = _tts_with_retry(jp_say, JP_VOICE, jp_path)
        ok_cn = cn_text and _tts_with_retry(cn_text, CN_VOICE, cn_path)

        # 字源 TTS：每段「かな（漢字）是中文」拆成 JP念かな + CN念「是中文」
        # 例：す（酢）是醋 → JP:「す」+ CN:「是醋」
        import re as _re2
        _seg_pat = _re2.compile(r'^(.+?)(?:（[^）]*）)?是(.+)$')
        etym_audio_paths = []
        if etym:
            for si, seg in enumerate(s.strip() for s in etym.split('，') if s.strip()):
                m = _seg_pat.match(seg)
                if m:
                    jp_e = os.path.join(tmp_dir, "audio", f"{idx:03d}_e{si}j.mp3")
                    cn_e = os.path.join(tmp_dir, "audio", f"{idx:03d}_e{si}c.mp3")
                    if _tts_with_retry(m.group(1).strip(), JP_VOICE, jp_e):
                        etym_audio_paths.append(jp_e)
                    if _tts_with_retry('是' + m.group(2).strip(), CN_VOICE, cn_e):
                        etym_audio_paths.append(cn_e)
                else:
                    seg_tts = _re2.sub(r'（[^）]*）', '', seg).strip()
                    all_e = os.path.join(tmp_dir, "audio", f"{idx:03d}_e{si}a.mp3")
                    if _tts_with_retry(seg_tts, CN_VOICE, all_e):
                        etym_audio_paths.append(all_e)

        if not ok_jp:
            consecutive_fails += 1
            tts_failures.append(hiragana)
        else:
            consecutive_fails = 0

        if consecutive_fails >= 5:
            try:
                encyclopedia.update_batch_status(vid_id, "failed", "連續 5 個 TTS 失敗，中止生成")
            except Exception:
                pass
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return f"影片生成中止：連續 5 個單字 TTS 全部失敗，疑似網路異常。已處理 {idx} 個單字。"

        # Build audio: JP → CN → JP（重用）→ 字源各段
        audio_parts = [AudioFileClip(jp_path)]
        if ok_cn:
            audio_parts.append(AudioFileClip(cn_path))
        audio_parts.append(AudioFileClip(jp_path))
        for ep in etym_audio_paths:
            audio_parts.append(AudioFileClip(ep))

        full_audio = concatenate_audioclips(audio_parts)
        duration   = full_audio.duration + 0.5

        frame_path   = _make_frame(word, idx)
        segment_path = os.path.join(tmp_dir, "segments", f"{idx:03d}.mp4")

        clip = ImageClip(frame_path).with_duration(duration).with_audio(full_audio)
        clip.write_videofile(segment_path, fps=24, logger=None, audio_codec="aac")

        # Release memory immediately
        clip.close()
        full_audio.close()
        for ac in audio_parts:
            ac.close()

        # Clean up per-word intermediates
        os.remove(frame_path)
        for p in [jp_path, cn_path] + etym_audio_paths:
            if os.path.exists(p):
                os.remove(p)

        segment_paths.append(segment_path)

    # ── 6. Final concatenation ─────────────────────────────────
    try:
        seg_clips = [VideoFileClip(p) for p in segment_paths]
        final     = concatenate_videoclips(seg_clips, method="compose")
        final.write_videofile(out_path, fps=24, logger=None, audio_codec="aac")
        for c in seg_clips:
            c.close()
        final.close()
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            encyclopedia.update_batch_status(vid_id, "failed", str(e))
        except Exception:
            pass
        return f"影片合成失敗：{e}"

    # Clean up segment files and tmp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Verify output
    try:
        vc = VideoFileClip(out_path)
        assert vc.duration > 0
        vc.close()
    except Exception as e:
        return f"影片驗證失敗（檔案可能損壞）：{e}"

    # Update batch status to uploading
    try:
        encyclopedia.update_batch_status(vid_id, "uploading")
    except Exception:
        pass

    filmed = [w["hiragana"] for w in words if w.get("hiragana")]
    try:
        encyclopedia.mark_words_in_video(filmed, vid_id)
    except Exception:
        pass

    warn = f"\n（注意：{len(tts_failures)} 個單字 TTS 失敗，已靜音：{tts_failures[:5]}）" if tts_failures else ""
    return (
        f"影片生成完成！{warn}\n"
        f"路徑：{out_path}\n"
        f"批次 ID：{vid_id}\n"
        f"共 {len(segment_paths)} 個單字片段。\n\n"
        f"下一步：呼叫 upload_to_youtube 工具上傳至 YouTube（需人工核准）。"
    )


# ──────────────────────────────────────────────────────────────
# Tools — dangerous (require interrupt_before human review)
# ──────────────────────────────────────────────────────────────

@tool
def upload_to_youtube(local_path: str, batch_id: str,
                      title: str = "", description: str = "") -> str:
    """上傳本機影片到 YouTube（unlisted）並將 video_id 寫回資料庫。
    ⚠️ 此操作需要人工核准後才會執行。
    需要 /app/secrets/yt_token.json（OAuth2 token）。
    """
    if not os.path.exists(local_path):
        return f"錯誤：找不到影片檔案 {local_path}。"

    token_path = "/app/secrets/yt_token.json"
    if not os.path.exists(token_path):
        return (
            "YouTube 認證失敗：找不到 /app/secrets/yt_token.json。\n"
            "請先在本機執行 OAuth 授權流程並將 token 檔案掛載至容器。\n"
            "參考指令：docker run -v $(pwd)/secrets:/app/secrets daedalus python /app/scripts/auth_youtube.py"
        )

    # Check daily quota
    try:
        count = encyclopedia.get_today_upload_count()
        if count >= 5:
            try:
                encyclopedia.update_batch_status(batch_id, "quota_hold")
            except Exception:
                pass
            return f"YouTube 每日上傳配額已達上限（{count}/5）。本機影片保留，請明日再試。"
    except Exception:
        pass

    try:
        import json
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request

        with open(token_path) as f:
            token_data = json.load(f)
        creds = Credentials.from_authorized_user_info(token_data)

        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(token_path, "w") as f:
                f.write(creds.to_json())

        youtube = build("youtube", "v3", credentials=creds)

        body = {
            "snippet": {
                "title": title or f"日文單字學習影片 — 批次 {batch_id}",
                "description": description or "由 Daedalus 自動生成的日文單字教學影片",
                "tags": ["日文學習", "Japanese", "JLPT", "單字"],
                "categoryId": "27",
            },
            "status": {"privacyStatus": "unlisted"},
        }

        media   = MediaFileUpload(local_path, mimetype="video/mp4", resumable=True, chunksize=10 * 1024 * 1024)
        request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

        response = None
        while response is None:
            _, response = request.next_chunk()

        video_id = response["id"]

        try:
            encyclopedia.update_batch_youtube_id(batch_id, video_id)
            encyclopedia.mark_words_in_video(
                [w["hiragana"] for w in encyclopedia.get_words_for_video(limit=0)
                 if w.get("video_batch_id") == batch_id],
                batch_id, video_id
            )
        except Exception:
            pass

        return (
            f"YouTube 上傳成功！\n"
            f"Video ID：{video_id}\n"
            f"連結：https://youtu.be/{video_id}\n\n"
            f"下一步：呼叫 delete_local_video 刪除本機影片（需人工核准）。"
        )

    except Exception as e:
        try:
            encyclopedia.update_batch_status(batch_id, "failed", str(e))
        except Exception:
            pass
        return f"YouTube 上傳失敗：{e}"


@tool
def delete_local_video(local_path: str, batch_id: str) -> str:
    """刪除本機影片檔案（不可逆）。
    ⚠️ 此操作需要人工核准後才會執行。
    執行前系統會確認 YouTube video_id 已寫入資料庫。
    """
    try:
        batch = encyclopedia.get_batch(batch_id)
        if not batch or not batch.get("youtube_video_id"):
            return (
                "刪除被拒：資料庫中找不到此批次的 YouTube video_id。\n"
                "請先確認 upload_to_youtube 已成功執行並寫入資料庫。"
            )
    except Exception as e:
        return f"資料庫驗證失敗：{e}"

    if not os.path.exists(local_path):
        return f"檔案已不存在：{local_path}（可能已被刪除）。"

    try:
        os.remove(local_path)
        encyclopedia.clear_batch_local_path(batch_id)
        return (
            f"本機影片已刪除：{local_path}\n"
            f"YouTube 連結（永久保存）：https://youtu.be/{batch['youtube_video_id']}"
        )
    except Exception as e:
        return f"刪除失敗：{e}"


# ──────────────────────────────────────────────────────────────
# Tools — submit (bind_tools only; intercepted by task_reporter, never run by ToolNode)
# ──────────────────────────────────────────────────────────────

@tool
def submit_task_completion(
    task_description: str,
    modified_files: list,
    change_summary: str,
    test_executed: bool,
    risk_level: str,
    risk_reason: str,
    tool_calls_made: list,
    test_passed: Optional[bool] = None,
) -> str:
    """[任務完成強制回報] 每次完成實作任務後必須呼叫，不得用純文字代替。

    task_description : 複述使用者當前任務需求（用執行當下最新版本，非最早那則）
    modified_files   : 確實被改動的檔案路徑清單（無改動傳 []）
    change_summary   : diff 精簡描述，100 字以內，不貼全文
    test_executed    : 是否呼叫過 forge_and_test_tool 或其他測試工具
    test_passed      : 測試結果（test_executed=false 時傳 null）
    risk_level       : low / medium / high（僅反映程式碼改動本身的品質風險）
                       low    = 只讀操作，或呼叫危險工具但無程式碼寫入
                       medium = 有寫入/DB 操作，且測試通過
                       high   = 測試失敗、修改現有工具定義、涉安全邏輯
                       ⚠️ 「呼叫危險工具」不納入此欄位——危險工具的執行風險已由 interrupt_before 獨立管控
    risk_reason      : 一句話說明風險等級理由
    tool_calls_made  : 本次任務實際呼叫的所有工具名稱
    """
    # 此函式由 task_reporter 節點攔截，不會實際執行
    return "[此工具由 task_reporter 節點攔截處理]"


# ──────────────────────────────────────────────────────────────
# Tool registry & routing
# ──────────────────────────────────────────────────────────────

DANGEROUS_TOOL_NAMES = frozenset({"upload_to_youtube", "delete_local_video"})

_safe_tools = [
    web_search, fetch_web_page, python_executor, forge_and_test_tool,
    init_encyclopedia_db, add_japanese_word,
    get_video_candidate_words, generate_japanese_learning_video,
]
_dangerous_tools = [upload_to_youtube, delete_local_video]
_submit_tools    = [submit_task_completion]  # bind_tools only; NOT in ToolNode

all_tools  = _safe_tools + _dangerous_tools          # tools ToolNode handles
_llm_tools = all_tools + _submit_tools               # full schema set for LLM
tool_node  = ToolNode(all_tools)

# ──────────────────────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

llm = ChatOllama(
    model="gemma4:26b",
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
    streaming=True,
).bind_tools(_llm_tools)

# Reviewer: qwen2.5-coder:7b — Alibaba Cloud training lineage, independent from Google gemma4.
# Must be instruction-tuned: the Reviewer must follow the system prompt and output
# strictly-formatted JSON. Selection history:
#   starcoder2:7b       → no instruction tuning, JSON validity 0/3, 100% fallback
#   codellama:7b-instruct → valid JSON 3/3, but review quality 0/3 (approved all errors)
#   qwen2.5-coder:7b    → instruction-tuned + strong code reasoning = final choice
# Security: local Ollama inference, data never leaves host, localhost-bound only.
reviewer_llm = ChatOllama(
    model="qwen2.5-coder:7b",
    base_url=OLLAMA_BASE_URL,
    temperature=0.1,
    streaming=False,
    # Reviewer input now embeds the machine-extracted tool_code + test_code;
    # Ollama's default context is too small for system prompt + code blocks.
    num_ctx=8192,
)

# ──────────────────────────────────────────────────────────────
# Graph nodes
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是配備多種工具的全能助理 Daedalus，專精日文學習影片製作。

【強制規則 — 違反即為系統錯誤】
1. 禁止用純文字假裝執行工具。所有操作必須透過實際 Tool Call 完成。
2. 若工具回傳錯誤，必須如實呈現完整錯誤訊息，不得宣稱操作成功。
3. 不確定的事實禁止臆測，應呼叫 web_search 查詢後再回答。
4. 回覆一律使用繁體中文。
5. 每次完成實作任務後，必須呼叫 submit_task_completion 提交結構化回報，不得用純文字代替。
6. 每次收到新任務或 Reviewer 修正要求時，第一步必須先輸出 2-3 句說明文字：
   「我要做什麼」與「打算怎麼做」（包含預計呼叫的工具與驗證方式），說完立即接著呼叫工具，
   不得只呼叫工具而完全沒有說明，也不得說完就停下來等使用者確認。

【建立／修改工具的強制流程 — 最常見的違規情境】
當任務涉及「建立、撰寫、修改任何 Python 函式或工具」，唯一合法的完成方式是：
  Step 0. 先用 1-3 句話說明「我要做什麼」與「打算怎麼做」（例：「我要建立計算階乘的函式，
           包含正整數、零的計算，以及負數拋出 ValueError 的防禦性處理。」），然後直接執行，不需等使用者確認。
  Step A. 呼叫 forge_and_test_tool(tool_code=<完整函式>, test_code=<單元測試>)
  Step B. 讀取回傳的 [AST_RESULT] / [TEST_RESULT] / [DISK_WRITE_RESULT] 三段標籤
  Step C. 測試失敗則修正後重試，不得宣稱成功
  Step D. 呼叫 submit_task_completion 提交結構化回報

【forge_and_test_tool 的 tool_code 寫法】
  ✅ 只寫純 Python def 函式，無任何 decorator
  ✅ 不使用 import（沙盒環境不支援，import 陳述式會直接失敗）
  ✗ 禁止加 @tool decorator（沙盒不認識，會報 name 'tool' is not defined）
  ✗ 禁止 @staticmethod / @classmethod / @property 等 decorator（除非是內部 class）

以下行為在任何情況下均屬違規（系統會偵測並強制重做）：
  ✗ 在訊息裡用 ```python 程式碼區塊展示程式碼，未呼叫任何工具
  ✗ 說「以下是程式碼，請自行貼入」或「已為你撰寫好如下」
  ✗ 以「環境限制」「目錄唯讀」等理由改用純文字展示（應試後如實回報錯誤）
  ✗ 沒有呼叫 forge_and_test_tool 就直接呼叫 submit_task_completion

【submit_task_completion 填寫規範】
- task_description : 複述當前執行的任務需求（用最新版本，非對話最早那則）
- modified_files   : 確實已寫入磁碟的檔案路徑清單（無改動傳 []）
    ⚠️ 判斷標準：forge_and_test_tool 回傳 [DISK_WRITE_RESULT]SUCCESS 才可填入對應路徑
       若回傳 DISK_WRITE_RESULT=FAIL 或 SKIP，必須傳 []，不得因沙盒測試通過就填入
- change_summary   : diff 精簡描述，100 字以內
- test_executed    : forge_and_test_tool 或其他測試工具是否執行且有明確 [TEST_RESULT]
- test_passed      : [TEST_RESULT]PASS 則 true，FAIL 則 false（test_executed=false 時傳 null）
- risk_level       : low / medium / high（僅反映程式碼改動本身的品質風險）
    low    = 只讀操作，或呼叫危險工具但無程式碼寫入（純動作任務）
    medium = 有寫入/DB 操作，且測試通過
    high   = 測試失敗、修改現有工具定義、涉及安全邏輯
    ⚠️ 「呼叫危險工具」不列入此欄位——危險工具的執行已由 interrupt_before 獨立管控，不透過 risk_level 觸發 human_escalation
- risk_reason      : 一句話說明為何是這個等級
- tool_calls_made  : 本次任務「所有呼叫過的工具名稱清單」，包含失敗／放棄的嘗試
    ⚠️ 不得只列最終生效的工具；forge_and_test_tool 嘗試多次也必須列出

【日文影片製作標準流程】
Step 1. get_video_candidate_words — 確認有哪些單字尚未拍攝
Step 2. web_search — 搜尋日文詞頻排名資料（只取排名數據，不複製單字清單）
Step 3. add_japanese_word（重複呼叫）— 逐一寫入單字（含 emoji、etymology、jlpt_level）
         字源判斷標準：複合詞/外來語/慣用語 → 需要；基礎詞彙 → 留空
Step 4. generate_japanese_learning_video — 自動取詞並生成影片（回傳本機路徑與批次 ID）
Step 5. upload_to_youtube — 上傳至 YouTube（⚠️ 需人工核准）
Step 6. delete_local_video — 刪除本機影片（⚠️ 需人工核准，確認 YouTube 已可播放後執行）

【關鍵限制】
- upload_to_youtube 與 delete_local_video 為敏感操作，系統會自動觸發人工審查，
  Agent 無需也不應自行跳過此機制。
- 每日 YouTube 上傳上限為 5 部影片（API 配額限制）。
"""


REVIEWER_SYSTEM_PROMPT = """你是一位嚴格的程式碼審查員（Code Reviewer），職責是找出問題，不是讚美。
你的唯一目標是保護系統不被引入缺陷、安全漏洞或未完成的實作。

═══════════════════════════════════════════════════════
【CRITICAL RULE】category: "scope" 的合法使用限制
═══════════════════════════════════════════════════════
以下程式碼【絕對禁止】被標記為 category: "scope"：
  ✅ isinstance / type() 型別檢查
  ✅ raise ValueError / raise TypeError
  ✅ if n < 0、if x is None、if not lst 等邊界防禦
  ✅ try/except、finally、raise、guard clause
  ✅ 任何防禦性輸入驗證或錯誤處理

【唯一合法的 scope creep】：改動新增了與原始任務完全無關的全新功能。
  ✅ 正確範例：任務是「寫階乘函式」，diff 還額外加了一個 fetch_url 函式 → scope
  ❌ 錯誤範例：任務是「寫階乘函式」，diff 包含 `if n < 0: raise ValueError` → 不是 scope，應是 edge_case

在標記 category: "scope" 之前，必須確認：
  (1) 這段程式碼不屬於輸入驗證、邊界條件、錯誤處理的任何一種
  (2) 這是一個與原始任務毫無關聯的全新功能
  兩個條件都不成立 → 禁止使用 category: "scope"
═══════════════════════════════════════════════════════

═══════════════════════════════════════════════════════
【測試有效性檢查 — 必查維度】
═══════════════════════════════════════════════════════
輸入若包含【實際提交的 test_code】，必須判斷「測試是否真的驗證了核心邏輯」，
不是形式上通過就好。以下任一情況成立，即為「測試無效或被弱化」，
必須列為 severity=blocker、category=test，且 verdict 必須是 needs_changes：
  1. 測試斷言被簡化或刪除（例如只剩 assertTrue(True)、空的測試方法）
  2. 測試只驗證無關緊要的邊角（例如只測函式 callable、只測回傳型別，
     完全沒有驗證計算結果或核心行為）
  3. 測試條件過於寬鬆（例如用範圍斷言取代精確值、只斷言不拋例外）
  4. 測試根本沒有呼叫被測函式
  5. 原始任務明確要求的行為（如邊界條件、例外處理）沒有任何斷言覆蓋
描述請寫明「測試無效或被弱化」並指出是哪一種手法。
若 Coder 宣稱測試通過但輸入中沒有 test_code 可查驗，在 issues 標記
「資訊不足，需要人工確認測試有效性」。

行為規範：
1. 凡有疑慮，必須列為 issue，不得因「看起來應該 OK」而放行。
2. 你看不到實作者的推理過程——這是刻意的設計，你的判斷必須完全基於改動的結果。
   改動摘要是 Coder 自報的，可能與實際程式碼不符；以機器擷取的 tool_code / test_code 為準。
3. 若任何欄位資訊不足以判斷，在 issues 中標記「資訊不足，需要人工確認」。

verdict 判定規則（必須遵守）：
- 有任何 severity=blocker 的 issue → verdict 必須是 needs_changes
- major issue 數量 >= 2 → verdict 必須是 needs_changes
- 只有 minor issue → 可以 approve（仍需列出 issues）

你的回覆必須只包含以下格式的 JSON，不得附加任何前言、解釋或後記：

{
  "verdict": "approve",
  "reviewer_risk": "low",
  "issues": [],
  "approved_aspects": ["邏輯正確：<引用改動摘要中的具體行為>", "測試真實性：<引用 AST/TEST/DISK 機器標籤的實際值>"],
  "one_line_summary": "一句話結論"
}

【approved_aspects 填寫規則——必須遵守】
- 不得為空陣列 []
- 每條必須引用「本次任務的具體事實」：改動摘要的內容、機器標籤（AST_RESULT / TEST_RESULT /
  DISK_WRITE_RESULT）的實際值、modified_files 的實際路徑等
- 禁止輸出籠統套話（如「邏輯正確」「邊界處理完整」「符合原始任務需求」這類沒有依據的空泛字句），
  也禁止照抄上方範例中的 <> 佔位文字——必須換成本次審查的實際內容
- 若 verdict = "approve"，approved_aspects 至少要有 2 條
- 若 verdict = "needs_changes"，至少要有 1 條（說明哪些面向是 OK 的）

issue 物件格式：
{
  "severity": "blocker",
  "category": "correctness",
  "description": "問題描述",
  "location": null,
  "suggestion": null
}

severity 只能是：blocker / major / minor
category 只能是：correctness / security / edge_case / test / scope
verdict 只能是：approve / needs_changes
reviewer_risk 只能是：low / medium / high
"""


def _extract_tagged_result(messages: list, tag: str) -> Optional[str]:
    """Return the most recent [TAG]...[/TAG] content from messages."""
    open_tag  = f"[{tag}]"
    close_tag = f"[/{tag}]"
    pattern   = re.compile(re.escape(open_tag) + r"(.*?)" + re.escape(close_tag), re.DOTALL)
    for msg in reversed(messages):
        content = getattr(msg, "content", "") or ""
        if open_tag in content:
            m = pattern.search(content)
            if m:
                return m.group(1).strip()
    return None


def _extract_ast_result(messages: list) -> Optional[str]:
    """Return the most recent [AST_RESULT]...[/AST_RESULT] content."""
    return _extract_tagged_result(messages, "AST_RESULT")


def _extract_disk_write_result(messages: list) -> Optional[str]:
    """Return the most recent [DISK_WRITE_RESULT]...[/DISK_WRITE_RESULT] content."""
    return _extract_tagged_result(messages, "DISK_WRITE_RESULT")


def _extract_last_forge_code(messages: list) -> tuple[Optional[str], Optional[str]]:
    """Return (tool_code, test_code) from the most recent forge_and_test_tool call.

    Machine-extracted from the actual tool_calls history — NOT Coder-reported —
    so the Reviewer can audit the real code and the real tests instead of
    trusting change_summary.
    """
    for msg in reversed(messages):
        for tc in reversed(getattr(msg, "tool_calls", None) or []):
            if isinstance(tc, dict) and tc.get("name") == "forge_and_test_tool":
                args = tc.get("args") or {}
                return args.get("tool_code"), args.get("test_code")
    return None, None


def _clip_code(text: Optional[str], limit: int) -> str:
    text = (text or "").strip()
    if not text:
        return "（未提供）"
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…（超過 {limit} 字元已截斷）"


def _build_reviewer_input(
    task_report: dict,
    original_task: str,
    ast_scan_result: Optional[str],
    disk_write_result: Optional[str] = None,
    tool_code: Optional[str] = None,
    test_code: Optional[str] = None,
) -> str:
    modified_files  = task_report.get("modified_files") or []
    change_summary  = task_report.get("change_summary", "（無摘要）")
    test_executed   = task_report.get("test_executed", False)
    test_passed     = task_report.get("test_passed")
    risk_level      = task_report.get("risk_level", "unknown")
    risk_reason     = task_report.get("risk_reason", "（未說明）")
    tool_calls_made = task_report.get("tool_calls_made") or []

    if not test_executed:
        test_status = "否（未執行測試）"
    elif test_passed is True:
        test_status = "是，且通過"
    elif test_passed is False:
        test_status = "是，但失敗"
    else:
        test_status = "是（結果未知）"

    ast_txt   = ast_scan_result   or "未取得（non-forge 任務或標籤缺失）"
    disk_txt  = disk_write_result or "未取得（non-forge 任務或標籤缺失）"
    files_txt = "\n".join(f"- {f}" for f in modified_files) or "- （無改動）"
    tools_txt = ", ".join(tool_calls_made) or "（無）"

    # test_code is the cheat-detection anchor — always attach it in full
    # (clip only at an absurd size); tool_code may be clipped harder.
    code_section = ""
    if tool_code or test_code:
        code_section = (
            f"【實際提交的 tool_code（機器擷取，非 Coder 自報）】\n"
            f"```python\n{_clip_code(tool_code, 1500)}\n```\n\n"
            f"【實際提交的 test_code（機器擷取——必須審查測試有效性）】\n"
            f"```python\n{_clip_code(test_code, 4000)}\n```\n\n"
        )

    return (
        f"請審查以下任務的完成情況。\n\n"
        f"【原始任務需求】\n{original_task or '（未記錄）'}\n\n"
        f"【改動摘要（Coder 自報，可能不誠實）】\n{change_summary}\n\n"
        f"{code_section}"
        f"【宣告改動的檔案（modified_files）】\n{files_txt}\n\n"
        f"【測試狀態（Coder 自報）】\n{test_status}\n\n"
        f"【實作者自評風險等級】\n{risk_level} — {risk_reason}\n\n"
        f"【AST 靜態分析結果（機器回傳）】\n{ast_txt}\n\n"
        f"【磁碟寫入結果（機器回傳）】\n{disk_txt}\n"
        f"  ⚠️ modified_files 只在 DISK_WRITE_RESULT=SUCCESS 時才合法；"
        f"若 Coder 填了 modified_files 但 DISK_WRITE_RESULT 非 SUCCESS，請標記為 blocker。\n\n"
        f"【實際使用的工具清單（Coder 自報）】\n{tools_txt}\n\n"
        f"請根據以上資訊進行審查，輸出 JSON 格式的審查結果。"
    )


def _parse_review_result(text: str) -> ReviewResult:
    """Extract JSON from LLM response text and parse into ReviewResult."""
    # Try ```json ... ``` block first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return ReviewResult(**json.loads(m.group(1)))
    # Fallback: first {...} in the text
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return ReviewResult(**json.loads(m.group(0)))
    raise ValueError(f"回應中找不到有效 JSON：{text[:300]!r}")


# Keywords that identify defensive-programming issues wrongly labelled as "scope"
_DEFENSIVE_SCOPE_KW = [
    "負數", "零", "none", "空字串", "空列表", "空 list", "空陣列",
    "isinstance", "valueerror", "typeerror",
    "raise", "try/except", "try-except", "guard",
    "錯誤處理", "邊界", "輸入驗證", "防禦性",
    "input validation", "error handling", "edge case",
    "if n <", "if n<=", "if x is none", "< 0",
]


def _strip_false_scope_issues(result: ReviewResult) -> ReviewResult:
    """
    Post-filter: drop scope-category issues that describe defensive programming.
    Recalculate verdict based on remaining issues so a false-scope blocker
    doesn't block approval.
    """
    kept, dropped = [], []
    for issue in result.issues:
        if issue.category == "scope":
            desc_lower = issue.description.lower()
            if any(kw in desc_lower for kw in _DEFENSIVE_SCOPE_KW):
                dropped.append(issue)
                continue
        kept.append(issue)

    if not dropped:
        return result

    has_blocker  = any(i.severity == "blocker" for i in kept)
    major_count  = sum(1 for i in kept if i.severity == "major")
    new_verdict  = "needs_changes" if (has_blocker or major_count >= 2) else "approve"
    suffix       = "（scope 誤判已自動過濾）" if new_verdict == "approve" else ""
    return ReviewResult(
        verdict          = new_verdict,
        reviewer_risk    = result.reviewer_risk,
        issues           = kept,
        approved_aspects = result.approved_aspects,
        one_line_summary = result.one_line_summary + suffix,
    )


# Aspects that are prompt-example parroting or contentless boilerplate — treated as absent.
_GENERIC_ASPECTS = {
    "邏輯正確", "邊界處理完整", "符合原始任務需求", "程式碼審查通過",
    "無具體說明", "無法評估", "一句話結論",
}


def _ensure_approved_aspects(
    result: ReviewResult,
    task_report: dict,
    ast_scan_result: Optional[str],
    disk_write_result: Optional[str],
) -> ReviewResult:
    """
    Guarantee approved_aspects carries concrete content: approve → >= 2 entries,
    needs_changes → >= 1. Drops prompt-example parroting / placeholder text and
    tops up from machine-verifiable facts (AST / TEST / DISK tags, report fields).
    """
    kept = []
    for a in result.approved_aspects:
        a = (a or "").strip()
        if not a or a in _GENERIC_ASPECTS or a.startswith("<") or "佔位" in a or "具體面向" in a:
            continue
        kept.append(a)

    required = 2 if result.verdict == "approve" else 1
    if len(kept) >= required:
        if kept == result.approved_aspects:
            return result
        return result.model_copy(update={"approved_aspects": kept})

    facts = []
    if ast_scan_result and ast_scan_result.startswith("PASS"):
        facts.append("安全性：AST 靜態掃描通過，無 import / exec / eval 等危險呼叫")
    if task_report.get("test_executed") and task_report.get("test_passed"):
        facts.append("測試真實性：forge_and_test_tool 回傳 [TEST_RESULT]PASS，單元測試確實執行且通過")
    if disk_write_result and disk_write_result.startswith("SUCCESS"):
        files = "、".join(task_report.get("modified_files") or []) or "（見回報）"
        facts.append(f"磁碟一致性：DISK_WRITE_RESULT=SUCCESS，與宣告的 modified_files 相符（{files}）")
    summary = (task_report.get("change_summary") or "").strip()
    if summary:
        facts.append(f"任務相符性：改動摘要「{summary[:60]}」與原始任務需求一致")

    for f in facts:
        if len(kept) >= required:
            break
        if f not in kept:
            kept.append(f)
    if not kept:
        kept.append("（Reviewer 未提供具體面向，且無機器標籤可歸納——請以問題清單為準）")

    return result.model_copy(update={"approved_aspects": kept})


async def call_model(state: DaedalusState, config: RunnableConfig):
    # Test hook: force escalation without spending an LLM call.
    if _force_escalation_requested(state):
        return {
            "messages": [AIMessage(content=(
                f"{_FORCE_ESCALATION_ACK} 偵測到測試開關 {FORCE_ESCALATION_MARKER}，"
                "直接進入人工介入畫面（未呼叫模型）。"
            ))],
            "retry_messages":   None,
            "total_iterations": (state.get("total_iterations") or 0) + 1,
        }

    sys_msg = {"role": "system", "content": SYSTEM_PROMPT}
    # After context_surgeon sets retry_messages, use that clean slate;
    # otherwise use the full accumulated history.
    context = state.get("retry_messages") or state["messages"]
    final = None
    async for chunk in llm.astream([sys_msg] + context, config=config):
        final = chunk if final is None else final + chunk
    return {
        "messages":         [final],
        "retry_messages":   None,
        "total_iterations": (state.get("total_iterations") or 0) + 1,
    }


def pre_tool_check(state: DaedalusState) -> dict:
    """No-op passthrough — interrupt_before fires here for dangerous tools."""
    return {}


def task_reporter(state: DaedalusState) -> dict:
    """
    Intercepts submit_task_completion, validates TaskCompletionReport schema,
    and implements 3-tier degradation:
      parse OK          → status="ok"   → reviewer_agent (Block 2 wires this)
      first parse fail  → status="retry" → back to agent with error
      second parse fail → status="escalate" → human_escalation
    """
    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", []) or []
    schema_error_count = state.get("schema_error_count") or 0

    submit_call = None
    other_calls = []
    for tc in tool_calls:
        if tc["name"] == SUBMIT_TOOL_NAME:
            submit_call = tc
        else:
            other_calls.append(tc)

    # Stub ToolMessages for any non-submit calls (they were never executed)
    extra_msgs = [
        ToolMessage(
            content=(
                f"[跳過] {tc['name']} 無法在 submit_task_completion 同一輪中執行。"
                "請在下一輪單獨呼叫。"
            ),
            tool_call_id=tc["id"],
        )
        for tc in other_calls
    ]

    if not submit_call:
        return {"messages": extra_msgs}

    try:
        args = dict(submit_call["args"])
        # Coerce stringified lists (some models serialize list as JSON string)
        for field in ("modified_files", "tool_calls_made"):
            val = args.get(field)
            if isinstance(val, str):
                try:
                    args[field] = json.loads(val)
                except Exception:
                    args[field] = [val]
        args["task_id"] = uuid.uuid4().hex[:8]
        report = TaskCompletionReport(**args)

        ok_msg = ToolMessage(
            content=f"[SUBMIT_OK] 任務回報驗證成功，task_id={report.task_id}",
            tool_call_id=submit_call["id"],
        )
        # Anchor original_task to the real user HumanMessage (set only once).
        # Using Coder's task_description risks vague/evolving rephrasing that
        # misleads the Reviewer into flagging the core task as scope creep.
        existing_original = state.get("original_task") or ""
        if not existing_original:
            user_msg = next(
                (m for m in state["messages"]
                 if isinstance(m, HumanMessage)
                 and not (m.content or "").startswith(_ENFORCER_MARKER)),
                None,
            )
            original_task_anchor = (user_msg.content or "").strip() if user_msg else report.task_description
        else:
            original_task_anchor = existing_original
        return {
            "messages":             extra_msgs + [ok_msg],
            "task_report":          report.model_dump(),
            "original_task":        original_task_anchor,
            "schema_error_count":   0,
            "task_reporter_status": "ok",
            "last_task_iterations": state.get("total_iterations") or 0,
            "total_iterations":     0,   # Reset counter on each successful submission
        }

    except Exception as e:
        schema_error_count += 1
        if schema_error_count >= 2:
            err_msg = ToolMessage(
                content=(
                    f"[SUBMIT_ESCALATE] submit_task_completion 連續失敗 {schema_error_count} 次，"
                    f"需要人工介入。錯誤：{e}"
                ),
                tool_call_id=submit_call["id"],
            )
            return {
                "messages":             extra_msgs + [err_msg],
                "schema_error_count":   schema_error_count,
                "task_reporter_status": "escalate",
            }
        else:
            err_msg = ToolMessage(
                content=(
                    f"[SUBMIT_ERROR] submit_task_completion 格式錯誤（第 {schema_error_count} 次）。\n"
                    f"錯誤：{e}\n"
                    f"必填欄位：task_description, modified_files, change_summary, "
                    f"test_executed, risk_level（low/medium/high）, risk_reason, tool_calls_made。"
                ),
                tool_call_id=submit_call["id"],
            )
            return {
                "messages":             extra_msgs + [err_msg],
                "schema_error_count":   schema_error_count,
                "task_reporter_status": "retry",
            }


def human_escalation(state: DaedalusState) -> dict:
    """Escalation node — interrupt_before fires here; Chainlit UI handled in _handle_interrupt."""
    return {}


def context_surgeon(state: DaedalusState) -> dict:
    """
    Selective amnesia on retry: strips all Coder AIMessages, keeps objective
    ToolMessages + the original user HumanMessage, then appends a synthesised
    Reviewer-feedback HumanMessage that includes the already-completed-tools
    summary (non-idempotent tools are flagged so the Coder avoids re-running
    them).  The result is stored in retry_messages; call_model reads and
    clears it so the full messages history is still preserved for audit.
    """
    messages      = state["messages"]
    review_result = state.get("review_result") or {}
    retry_count   = state.get("retry_count") or 0

    NON_IDEMPOTENT    = {"forge_and_test_tool", "generate_japanese_learning_video"}
    FAILURE_KEYWORDS  = ("失敗", "錯誤", "Error", "error", "Exception", "無法")
    INTERNAL_PREFIXES = ("[SUBMIT_OK]", "[SUBMIT_ERROR]", "[SUBMIT_ESCALATE]", "[跳過]")

    # ── Pass 1: tool_call_id → tool_name + forge function names ──────────────────
    # (must be done BEFORE stripping AIMessages)
    id_to_name:       dict[str, str] = {}
    id_to_forge_func: dict[str, str] = {}  # tool_call_id → def name inside tool_code
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            if not isinstance(tc, dict):
                continue
            id_to_name[tc["id"]] = tc["name"]
            if tc["name"] == "forge_and_test_tool":
                tool_code = (tc.get("args") or {}).get("tool_code", "")
                m = re.search(r"^def\s+(\w+)", tool_code, re.MULTILINE)
                if m:
                    id_to_forge_func[tc["id"]] = m.group(1)

    # ── Pass 2: classify ToolMessages ──────────────────────────────────────────
    completed: dict[str, dict] = {}
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        content = msg.content or ""
        if any(content.startswith(p) for p in INTERNAL_PREFIXES):
            continue
        name = id_to_name.get(msg.tool_call_id, "unknown")
        if name in (SUBMIT_TOOL_NAME, "unknown"):
            continue
        success = not any(kw in content for kw in FAILURE_KEYWORDS)
        if name not in completed:
            completed[name] = {
                "count": 0,
                "success": True,
                "non_idempotent": name in NON_IDEMPOTENT,
                "last_forge_func": None,
            }
        completed[name]["count"] += 1
        if not success:
            completed[name]["success"] = False
        func = id_to_forge_func.get(msg.tool_call_id)
        if func:
            completed[name]["last_forge_func"] = func

    # ── Pass 3: strip AIMessages, skip internal ToolMessages ───────────────────
    cleaned = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            continue
        if isinstance(msg, ToolMessage):
            if any((msg.content or "").startswith(p) for p in INTERNAL_PREFIXES):
                continue
        cleaned.append(msg)

    # ── Build "already completed tools" summary ─────────────────────────────────
    ni_ok_lines, ni_fail_lines, safe_lines = [], [], []
    for name, info in completed.items():
        status    = "✅" if info["success"] else "❌"
        func_hint = f"（函式名：{info['last_forge_func']}）" if info.get("last_forge_func") else ""
        line      = f"  - {name} × {info['count']}  {status}{func_hint}"
        if info["non_idempotent"]:
            (ni_ok_lines if info["success"] else ni_fail_lines).append(line)
        else:
            safe_lines.append(line)

    summary_sections = []
    if ni_ok_lines:
        summary_sections.append(
            "★ 有持久副作用（已成功寫入 / 生成，請勿重複呼叫相同版本）：\n"
            + "\n".join(ni_ok_lines)
            + "\n    ⚠️ 若需覆蓋，請以相同函式名稱修正後重新呼叫，不得另建新函式"
        )
    if ni_fail_lines:
        summary_sections.append(
            "⚠️ 寫入 / 生成失敗（磁碟未改動，可安全重新呼叫）：\n"
            + "\n".join(ni_fail_lines)
            + "\n    → 請修正程式碼後，以相同函式名稱重新呼叫 forge_and_test_tool，不得換名另建"
        )
    if safe_lines:
        summary_sections.append(
            "◎ 安全重複（冪等或純讀取）：\n" + "\n".join(safe_lines)
        )

    completed_summary = (
        "[已完成操作摘要 — 重試請確認後再執行]\n\n"
        + "\n\n".join(summary_sections)
        + "\n\n---\n\n"
    ) if summary_sections else ""

    # ── Build Reviewer feedback message ────────────────────────────────────────
    one_line   = review_result.get("one_line_summary", "")
    issues     = review_result.get("issues", [])
    _sev       = {"blocker": "[blocker]", "major": "[major]", "minor": "[minor]"}
    issues_txt = "\n".join(
        f"- {_sev.get(i['severity'], '[?]')} {i['description']}"
        + (f"（{i['location']}）" if i.get("location") else "")
        + (f" → {i['suggestion']}"  if i.get("suggestion")  else "")
        for i in issues
    ) or "（無具體問題清單）"

    feedback = HumanMessage(content=(
        f"{completed_summary}"
        f"[Reviewer 審查結果 — 第 {retry_count} 次 / 共 3 次]\n\n"
        f"結論：{one_line}\n\n"
        f"問題清單：\n{issues_txt}\n\n"
        f"以上工具執行記錄（ToolMessages）已保留供參考。\n"
        f"請重新審視原始任務，修正後再次呼叫 submit_task_completion 提交。"
    ))
    cleaned.append(feedback)

    return {"retry_messages": cleaned}


async def reviewer_agent(state: DaedalusState) -> dict:
    """
    Independent review using qwen2.5-coder:7b (Alibaba Cloud lineage, instruction-tuned).
    Context isolation: only TaskCompletionReport data + original_task are passed.
    No Coder conversation history, no Coder reasoning.
    """
    task_report   = state.get("task_report") or {}
    original_task = state.get("original_task") or ""
    retry_count   = state.get("retry_count") or 0

    ast_scan_result      = _extract_ast_result(state["messages"])
    disk_write_result    = _extract_disk_write_result(state["messages"])
    tool_code, test_code = _extract_last_forge_code(state["messages"])
    reviewer_input       = _build_reviewer_input(
        task_report, original_task, ast_scan_result, disk_write_result,
        tool_code, test_code,
    )

    try:
        response = await reviewer_llm.ainvoke([
            {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
            {"role": "user",   "content": reviewer_input},
        ])
        result = _parse_review_result(response.content)
        result = _strip_false_scope_issues(result)
        result = _ensure_approved_aspects(result, task_report, ast_scan_result, disk_write_result)
    except Exception as e:
        result = ReviewResult(
            verdict="needs_changes",
            reviewer_risk="high",
            issues=[IssueItem(
                severity="blocker",
                category="correctness",
                description=f"Reviewer 輸出解析失敗，需人工確認。技術錯誤：{e}",
            )],
            approved_aspects=["無法評估"],
            one_line_summary=f"Reviewer 解析失敗（{e}），需人工確認",
        )

    new_retry_count = retry_count + 1 if result.verdict == "needs_changes" else retry_count

    return {
        "review_result": result.model_dump(),
        "retry_count":   new_retry_count,
    }


def route_after_reviewer(state: DaedalusState) -> str:
    rr            = state.get("review_result") or {}
    verdict       = rr.get("verdict",       "needs_changes")
    reviewer_risk = rr.get("reviewer_risk", "high")
    coder_risk    = (state.get("task_report") or {}).get("risk_level", "high")
    retry_count   = state.get("retry_count") or 0

    # Either party flagging high risk → always require human confirmation
    if coder_risk == "high" or reviewer_risk == "high":
        return "human_escalation"

    if verdict == "approve":
        return "end"

    # needs_changes: retry_count was already incremented in reviewer_agent
    if retry_count >= 3:
        return "human_escalation"
    return "context_surgeon"


_ENFORCER_MARKER = "[TOOL_ENFORCER]"

# Deterministic test hook: a task containing this token routes straight to
# human_escalation without an LLM call. Exists because a task-based hard-cap
# trigger is unreliable — the Coder controls the sandbox and can always make
# forge pass with a trivial version, so it can't be forced to fail 6 rounds.
# This lets an operator exercise the escalation UI (e.g. the 放棄此任務 button)
# on demand. The token is deliberately obscure and never appears in normal use.
FORCE_ESCALATION_MARKER = "__FORCE_ESCALATION__"
_FORCE_ESCALATION_ACK = "[FORCE_ESCALATION]"


def _latest_user_task(messages: list) -> str:
    """Return the most recent genuine user task text (skips synthetic
    enforcer / reviewer-feedback HumanMessages injected by the graph)."""
    for m in reversed(messages):
        if not isinstance(m, HumanMessage):
            continue
        c = m.content or ""
        if c.startswith(_ENFORCER_MARKER) or c.startswith("[已完成操作摘要") \
                or "[Reviewer 審查結果" in c:
            continue
        return c
    return ""


def _force_escalation_requested(state: DaedalusState) -> bool:
    return FORCE_ESCALATION_MARKER in _latest_user_task(state["messages"])


def tool_enforcer(state: DaedalusState) -> dict:
    """
    Fires when the agent outputs a Python code block without calling any tool.
    Injects a correction HumanMessage and routes back to agent for one retry.
    Only fires once per conversation segment (marker prevents infinite loop).
    """
    return {
        "messages": [HumanMessage(content=(
            f"{_ENFORCER_MARKER} ⚠️ 系統偵測到你以 markdown 程式碼區塊回應，未呼叫任何工具。\n\n"
            "根據強制規則【建立／修改工具的強制流程】：\n"
            "- 建立或修改函式必須呼叫 forge_and_test_tool，不得以純文字展示代替\n"
            "- 在訊息裡貼程式碼不等於「已建立工具」\n\n"
            "請重新執行：呼叫 forge_and_test_tool(tool_code=..., test_code=...) 提交程式碼。"
        ))]
    }


def route_after_agent(state: DaedalusState) -> str:
    # Deterministic test hook — force escalation on demand.
    if _force_escalation_requested(state):
        return "human_escalation"

    # Hard iteration cap — catches forge-fail loops that bypass retry_count.
    if (state.get("total_iterations") or 0) >= 6:
        return "human_escalation"

    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        content = getattr(last, "content", "") or ""
        # Detect "code block without tool call" — catch the most common bypass pattern.
        # Only enforce once: if the enforcer marker already exists in recent history,
        # the agent had its chance and still refused; let it end rather than loop.
        has_python_block = "```python" in content or "```Python" in content
        already_enforced = any(
            isinstance(m, HumanMessage) and _ENFORCER_MARKER in (m.content or "")
            for m in state["messages"][-8:]
        )
        if has_python_block and not already_enforced:
            return "tool_enforcer"
        return "end"
    tool_names = [tc["name"] for tc in last.tool_calls]
    if SUBMIT_TOOL_NAME in tool_names:
        return "task_reporter"
    if any(name in DANGEROUS_TOOL_NAMES for name in tool_names):
        return "pre_tool_check"
    return "tools"


def route_after_task_reporter(state: DaedalusState) -> str:
    status = state.get("task_reporter_status") or ""
    if status == "escalate":
        return "human_escalation"
    if status != "ok":
        return "agent"            # "retry": schema error, give Coder another chance

    # Skip Reviewer when there are no code changes to review.
    # Pure action tasks (e.g. delete/upload with no file writes) have nothing
    # for the Reviewer to audit — routing them through reviewer_agent would cause
    # stale fallback risk scores to misfire human_escalation.
    tr = state.get("task_report") or {}
    has_code_changes = bool(tr.get("modified_files")) or bool(tr.get("test_executed"))
    if not has_code_changes:
        return "end"

    return "reviewer_agent"


# ──────────────────────────────────────────────────────────────
# Compile graph
# ──────────────────────────────────────────────────────────────

workflow = StateGraph(DaedalusState)
workflow.add_node("agent",            call_model)
workflow.add_node("pre_tool_check",   pre_tool_check)
workflow.add_node("tools",            tool_node)
workflow.add_node("task_reporter",    task_reporter)
workflow.add_node("reviewer_agent",   reviewer_agent)
workflow.add_node("context_surgeon",  context_surgeon)
workflow.add_node("human_escalation", human_escalation)
workflow.add_node("tool_enforcer",    tool_enforcer)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent", route_after_agent,
    {
        "pre_tool_check":  "pre_tool_check",
        "tools":           "tools",
        "task_reporter":   "task_reporter",
        "tool_enforcer":   "tool_enforcer",
        "human_escalation":"human_escalation",
        "end":             END,
    },
)
workflow.add_edge("tool_enforcer", "agent")
workflow.add_edge("pre_tool_check", "tools")
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges(
    "task_reporter", route_after_task_reporter,
    {
        "agent":            "agent",
        "human_escalation": "human_escalation",
        "reviewer_agent":   "reviewer_agent",
        "end":              END,
    },
)
workflow.add_conditional_edges(
    "reviewer_agent", route_after_reviewer,
    {
        "context_surgeon":  "context_surgeon",
        "human_escalation": "human_escalation",
        "end":              END,
    },
)
workflow.add_edge("context_surgeon",  "agent")
workflow.add_edge("human_escalation", END)

checkpointer = MemorySaver()

app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["pre_tool_check", "human_escalation"],
)
