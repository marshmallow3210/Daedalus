from langchain_core.runnables import RunnableConfig
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import encyclopedia  # noqa: E402

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

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
    tool_code 禁止含 import / exec / eval / open。
    """
    import ast, unittest, importlib
    from io import StringIO

    BLOCKED = {"exec", "eval", "open", "__import__"}
    try:
        tree = ast.parse(tool_code)
    except SyntaxError as e:
        return f"AST 解析失敗：{e}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "安全掃描失敗：tool_code 禁止使用 import。"
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name in BLOCKED:
                return f"安全掃描失敗：禁止呼叫 {name}()。"

    sb = {"__builtins__": __import__("builtins")}
    try:
        exec(tool_code, sb); exec(test_code, sb)
    except Exception as e:
        return f"沙盒載入失敗：{e}"

    loader = unittest.TestLoader(); suite = unittest.TestSuite()
    for obj in sb.values():
        if isinstance(obj, type) and issubclass(obj, unittest.TestCase) and obj is not unittest.TestCase:
            suite.addTests(loader.loadTestsFromTestCase(obj))

    buf = StringIO()
    result = unittest.TextTestRunner(stream=buf, verbosity=2).run(suite)
    if not result.wasSuccessful():
        return f"單元測試失敗，拒絕寫入：\n{buf.getvalue()}"

    path = os.getenv("CUSTOM_TOOLS_PATH", "/app/custom_tools.py")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n\n" + tool_code)
    except Exception as e:
        return f"寫入失敗：{e}"
    try:
        import custom_tools; importlib.reload(custom_tools)
        return f"工具已成功寫入並動態載入！\n{buf.getvalue()}"
    except Exception as e:
        return f"動態 reload 失敗：{e}"


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

    def _make_frame(word: dict, idx: int) -> str:
        W, H = 1920, 1080
        has_etym = bool(word.get("etymology"))
        img  = Image.new("RGB", (W, H), (12, 12, 35))
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, 0), (W, 6)], fill=(80, 130, 255))
        draw.rectangle([(0, H - 6), (W, H)], fill=(80, 130, 255))

        jlpt = word.get("jlpt_level", "")
        if jlpt:
            draw.rounded_rectangle([(W - 130, 18), (W - 18, 58)], radius=8, fill=(80, 130, 255))
            draw.text((W - 74, 38), jlpt, fill=(240, 240, 240), font=_find_font(26), anchor="mm")

        emoji_str = word.get("emoji", "")
        if emoji_str:
            draw.text((90, 120), emoji_str, fill=(240, 240, 240), font=_find_font(100), anchor="lm")

        main = word.get("kanji") or word.get("katakana") or word.get("hiragana", "")
        cy   = 280 if has_etym else 330
        draw.text((W // 2, cy), main, fill=(240, 240, 240), font=_find_font(130), anchor="mm")

        reading = word.get("hiragana", "")
        if reading and reading != main:
            draw.text((W // 2, cy + 110), f"（{reading}）",
                      fill=(160, 190, 255), font=_find_font(55), anchor="mm")

        div_y = cy + 170
        draw.rectangle([(W // 2 - 180, div_y), (W // 2 + 180, div_y + 3)], fill=(80, 130, 255))

        cn = word.get("chinese_translation", "")
        draw.text((W // 2, div_y + 80), cn, fill=(255, 210, 80), font=_find_font(85), anchor="mm")

        etym = word.get("etymology", "")
        if etym:
            f_etym = _find_font(34)
            for li, chunk in enumerate([etym[i:i+42] for i in range(0, min(len(etym), 126), 42)]):
                draw.text((W // 2, 820 + li * 48), chunk, fill=(160, 160, 160), font=f_etym, anchor="mm")

        path = os.path.join(tmp_dir, "frames", f"{idx:03d}.png")
        img.save(path, "PNG")
        return path

    # ── 5. Per-word segment generation ─────────────────────────
    segment_paths = []
    tts_failures  = []
    consecutive_fails = 0

    for idx, word in enumerate(words):
        hiragana = word.get("hiragana", "")
        jp_say   = word.get("katakana") or word.get("kanji") or hiragana
        cn_text  = word.get("chinese_translation", "")
        etym     = word.get("etymology", "")

        jp_path   = os.path.join(tmp_dir, "audio", f"{idx:03d}_jp.mp3")
        cn_path   = os.path.join(tmp_dir, "audio", f"{idx:03d}_cn.mp3")
        etym_path = os.path.join(tmp_dir, "audio", f"{idx:03d}_etym.mp3")

        ok_jp = _tts_with_retry(jp_say, JP_VOICE, jp_path)
        ok_cn = cn_text and _tts_with_retry(cn_text, CN_VOICE, cn_path)
        ok_et = etym and _tts_with_retry(etym, JP_VOICE, etym_path)

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

        # Build audio: JP → CN → JP (reuse jp_path) → etymology
        audio_parts = [AudioFileClip(jp_path)]
        if ok_cn:
            audio_parts.append(AudioFileClip(cn_path))
        audio_parts.append(AudioFileClip(jp_path))   # second JP play reuses same file
        if ok_et:
            audio_parts.append(AudioFileClip(etym_path))

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
        for p in [jp_path, cn_path, etym_path]:
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
# Tool registry & routing
# ──────────────────────────────────────────────────────────────

DANGEROUS_TOOL_NAMES = frozenset({"upload_to_youtube", "delete_local_video"})

_safe_tools = [
    web_search, fetch_web_page, python_executor, forge_and_test_tool,
    init_encyclopedia_db, add_japanese_word,
    get_video_candidate_words, generate_japanese_learning_video,
]
_dangerous_tools = [upload_to_youtube, delete_local_video]

all_tools = _safe_tools + _dangerous_tools
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
).bind_tools(all_tools)

# ──────────────────────────────────────────────────────────────
# Graph nodes
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是配備多種工具的全能助理 Daedalus，專精日文學習影片製作。

【強制規則 — 違反即為系統錯誤】
1. 禁止用純文字假裝執行工具。所有操作必須透過實際 Tool Call 完成。
2. 若工具回傳錯誤，必須如實呈現完整錯誤訊息，不得宣稱操作成功。
3. 不確定的事實禁止臆測，應呼叫 web_search 查詢後再回答。
4. 回覆一律使用繁體中文。

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


async def call_model(state: MessagesState, config: RunnableConfig):
    sys_msg = {"role": "system", "content": SYSTEM_PROMPT}
    final = None
    async for chunk in llm.astream([sys_msg] + state["messages"], config=config):
        final = chunk if final is None else final + chunk
    return {"messages": [final]}


def pre_tool_check(state: MessagesState) -> dict:
    """No-op passthrough — interrupt_before fires here for dangerous tools."""
    return {}


def route_after_agent(state: MessagesState) -> str:
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return "end"
    if any(tc["name"] in DANGEROUS_TOOL_NAMES for tc in last.tool_calls):
        return "pre_tool_check"
    return "tools"


# ──────────────────────────────────────────────────────────────
# Compile graph
# ──────────────────────────────────────────────────────────────

workflow = StateGraph(MessagesState)
workflow.add_node("agent",          call_model)
workflow.add_node("pre_tool_check", pre_tool_check)
workflow.add_node("tools",          tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent", route_after_agent,
    {"pre_tool_check": "pre_tool_check", "tools": "tools", "end": END},
)
workflow.add_edge("pre_tool_check", "tools")
workflow.add_edge("tools", "agent")

checkpointer = MemorySaver()

app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["pre_tool_check"],
)
