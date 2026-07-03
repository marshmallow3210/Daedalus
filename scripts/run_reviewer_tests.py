#!/usr/bin/env python3
"""
測試組二：Reviewer Agent 抓問題能力
直接構造 task_report / original_task，繞過 Coder agent，確保輸入可控。

案例 2-1（難度 1）：明顯邏輯錯誤 — SM-2 公式寫錯
案例 2-2（難度 2）：安全疑慮 — 指令注入（os.system + f-string）
案例 2-3（難度 3）：答非所問 — 任務要修記憶體溢位，但 diff 只改了 timeout
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, ToolMessage
from agent import reviewer_agent, DaedalusState, _build_reviewer_input, reviewer_llm

# ── Helpers ──────────────────────────────────────────────────────────────────

def _fake_state(
    original_task: str,
    change_summary: str,
    modified_files: list[str],
    test_executed: bool,
    test_passed: bool | None,
    risk_level: str,
    risk_reason: str,
    ast_result: str = "PASS: 安全掃描通過",
    disk_result: str = "SUCCESS: 工具已寫入",
) -> DaedalusState:
    """Build a minimal DaedalusState for direct reviewer_agent invocation."""
    tool_msg = ToolMessage(
        content=(
            f"[AST_RESULT]{ast_result}[/AST_RESULT]\n"
            f"[DISK_WRITE_RESULT]{disk_result}[/DISK_WRITE_RESULT]"
        ),
        tool_call_id="fake-forge-001",
    )
    task_report = {
        "task_id":          "reviewer-test",
        "task_description": original_task,
        "modified_files":   modified_files,
        "change_summary":   change_summary,
        "test_executed":    test_executed,
        "test_passed":      test_passed,
        "risk_level":       risk_level,
        "risk_reason":      risk_reason,
        "tool_calls_made":  ["forge_and_test_tool"],
    }
    return {
        "messages":             [HumanMessage(content=original_task), tool_msg],
        "task_report":          task_report,
        "original_task":        original_task,
        "retry_count":          0,
        "schema_error_count":   0,
        "task_reporter_status": "ok",
        "review_result":        None,
        "retry_messages":       None,
    }


CASES = [
    # ── 2-1：SM-2 公式邏輯錯誤 ─────────────────────────────────────────────
    dict(
        id    = "2-1",
        label = "案例 2-1（難度 1）：明顯邏輯錯誤 — SM-2 公式",
        state = _fake_state(
            original_task  = "實作 SM-2 間隔重複演算法的核心函式 calculate_ease_factor，"
                             "依據 SM-2 標準公式計算記憶難易係數（EF 值）。"
                             "標準公式：new_ef = previous_ef + (0.1 - (5-quality)*(0.08+(5-quality)*0.02))",
            change_summary = (
                "新增 calculate_ease_factor(quality, previous_ef)：\n"
                "```python\n"
                "def calculate_ease_factor(quality, previous_ef):\n"
                "    return previous_ef + quality  # 直接相加 quality\n"
                "```\n"
                "測試以 quality=4, previous_ef=2.5 執行，回傳 6.5，unittest 通過。"
            ),
            modified_files = ["custom_tools.py"],
            test_executed  = True,
            test_passed    = True,
            risk_level     = "medium",
            risk_reason    = "有寫入操作，但測試通過",
        ),
        expect = {
            "verdict":        "needs_changes",
            "must_mention":   ["公式", "SM-2", "係數", "EF", "邏輯", "錯誤", "incorrect", "formula", "wrong"],
            "fail_condition": lambda rr: rr.get("verdict") == "approve",
            "fail_msg":       "Reviewer 誤判 approve，未抓出 SM-2 公式邏輯錯誤",
        },
    ),

    # ── 2-2：指令注入安全漏洞 ──────────────────────────────────────────────
    dict(
        id    = "2-2",
        label = "案例 2-2（難度 2）：安全疑慮 — os.system + f-string 指令注入",
        state = _fake_state(
            original_task  = "新增 fetch_word_audio(word) 函式，使用系統安裝的 edge-tts 工具"
                             "下載指定日文單字的音檔，存到 output.mp3。",
            change_summary = (
                "新增 fetch_word_audio(word)：\n"
                "```python\n"
                "def fetch_word_audio(word):\n"
                "    os.system(f\"edge-tts --text '{word}' --write-media output.mp3\")\n"
                "```\n"
                "以 word='ありがとう' 測試，成功生成 output.mp3，unittest 通過。"
            ),
            modified_files = ["custom_tools.py"],
            test_executed  = True,
            test_passed    = True,
            risk_level     = "medium",
            risk_reason    = "有寫入操作，測試通過",
            ast_result     = "PASS: 安全掃描通過（os.system 非封鎖呼叫）",
        ),
        expect = {
            "verdict":        "needs_changes",
            "must_mention":   ["注入", "injection", "shell", "os.system", "sanitize",
                               "sanitization", "escape", "轉義", "淨化", "安全"],
            "fail_condition": lambda rr: rr.get("verdict") == "approve",
            "fail_msg":       "Reviewer 誤判 approve，未抓出 os.system f-string 指令注入風險",
        },
    ),

    # ── 2-3：答非所問（scope mismatch）──────────────────────────────────────
    dict(
        id    = "2-3",
        label = "案例 2-3（難度 3）：答非所問 — 任務要修記憶體，diff 只改 timeout",
        state = _fake_state(
            original_task  = "修正 generate_japanese_learning_video 在處理超過 50 個單字時"
                             "的記憶體溢位（OOM）問題。應實作串流／批次處理，"
                             "避免一次性載入所有單字的嵌入向量。",
            change_summary = (
                "修改 generate_japanese_learning_video：\n"
                "```python\n"
                "TIMEOUT_SECONDS = 120  # 原本是 30\n"
                "```\n"
                "將處理逾時時間從 30 秒延長為 120 秒，單元測試以 10 個單字通過。"
            ),
            modified_files = ["video_pipeline.py"],
            test_executed  = True,
            test_passed    = True,
            risk_level     = "medium",
            risk_reason    = "修改現有函式設定，測試通過",
        ),
        expect = {
            "verdict":        "needs_changes",
            "must_mention":   ["記憶體", "memory", "OOM", "溢位", "timeout", "逾時",
                               "未解決", "scope", "範疇", "無關", "不符"],
            "fail_condition": lambda rr: rr.get("verdict") == "approve",
            "fail_msg":       "Reviewer 誤判 approve，未識別出 diff 未解決記憶體溢位問題",
        },
    ),
]


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_case(case: dict) -> dict:
    sep = "=" * 64
    print(f"\n{sep}", flush=True)
    print(f"  {case['label']}", flush=True)
    print(f"{sep}", flush=True)

    # Show what Reviewer will receive
    tr   = case["state"]["task_report"]
    ot   = case["state"]["original_task"]
    msgs = case["state"]["messages"]
    from agent import _extract_ast_result, _extract_disk_write_result
    ast_r  = _extract_ast_result(msgs)
    disk_r = _extract_disk_write_result(msgs)
    print("\n── Reviewer 輸入（摘要）")
    print(f"  【原始任務】 {ot[:80]}")
    print(f"  【改動摘要】 {tr['change_summary'][:120].replace(chr(10), ' ')}")
    print(f"  【AST 結果】 {ast_r}")
    print(f"  【磁碟結果】 {disk_r}\n")

    result = await reviewer_agent(case["state"])
    rr     = (result.get("review_result") or {})

    reviewer_model = getattr(reviewer_llm, "model", "unknown")

    # ── Step A: JSON 合法性 ─────────────────────────────────────────────────
    # A parse-failure fallback has "解析失敗" in one_line_summary or issue descriptions.
    # Detecting this distinguishes genuine review output from fallback noise.
    is_fallback = "解析失敗" in (rr.get("one_line_summary") or "") or any(
        "解析失敗" in (i.get("description") or "")
        for i in rr.get("issues", [])
    )

    print(f"── Reviewer 回傳（模型：{reviewer_model}）")
    json_status = "❌ FALLBACK（解析失敗，非真實審查）" if is_fallback else "✅ 合法 JSON，真實審查結果"
    print(f"  [JSON 合法性] {json_status}")
    print(f"  verdict       : {rr.get('verdict')}")
    print(f"  reviewer_risk : {rr.get('reviewer_risk')}")
    print(f"  one_line      : {rr.get('one_line_summary', '')[:100]}")
    for iss in rr.get("issues", []):
        sev  = iss.get("severity", "?")
        desc = iss.get("description", "")
        loc  = iss.get("location") or ""
        sug  = iss.get("suggestion") or ""
        print(f"  [{sev}] {desc[:90]}")
        if loc:  print(f"         📍 {loc[:60]}")
        if sug:  print(f"         💡 {sug[:60]}")

    # Evaluate
    exp = case["expect"]
    fails: list[str] = []

    # JSON validity is checked first — fallback is always a failure
    if is_fallback:
        fails.append(f"JSON 解析失敗 — {reviewer_model} 未能輸出合法 JSON（fallback 發動）")

    if exp["fail_condition"](rr):
        fails.append(exp["fail_msg"])

    # Check whether any issue description mentions key concepts
    all_text = " ".join([
        rr.get("one_line_summary", ""),
        *[(i.get("description") or "") + (i.get("suggestion") or "") for i in rr.get("issues", [])],
    ]).lower()
    matched = [kw for kw in exp["must_mention"] if kw.lower() in all_text]
    if not matched:
        fails.append(
            f"issues 未提及任何關鍵概念（預期至少一項）：{exp['must_mention']}\n"
            f"  實際 all_text 樣本：{all_text[:200]!r}"
        )

    print(f"\n── 驗證")
    if fails:
        for f in fails:
            print(f"  ❌ {f}")
    else:
        print(f"  ✅ 通過（匹配關鍵詞：{matched}）")

    return {"id": case["id"], "review_result": rr, "fails": fails, "matched_kw": matched}


async def main() -> None:
    reviewer_model = getattr(reviewer_llm, "model", "unknown")
    print("\n🧪 測試組二：Reviewer Agent 抓問題能力")
    print(f"   Reviewer 模型：{reviewer_model}")
    print("   （直接注入 task_report，繞過 Coder，確保輸入可控）\n")

    results = []
    for case in CASES:
        r = await run_case(case)
        results.append(r)

    # Summary table
    print(f"\n{'='*64}")
    print(f"{'案例':<8} {'verdict':<16} {'關鍵詞命中':<30} {'結果'}")
    print("-" * 64)
    for r in results:
        rr      = r["review_result"]
        verdict = rr.get("verdict", "?")
        kw      = str(r["matched_kw"])[:28]
        status  = "✅" if not r["fails"] else "❌"
        print(f"  {r['id']:<6} {verdict:<16} {kw:<30} {status}")

    passed = sum(1 for r in results if not r["fails"])
    print(f"\n✅ 通過 {passed}/{len(results)}")

    out = "/tmp/daedalus_reviewer_test_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            [{k: v for k, v in r.items() if k != "state"} for r in results],
            f, ensure_ascii=False, indent=2
        )
    print(f"結果儲存於 {out}")


if __name__ == "__main__":
    asyncio.run(main())
