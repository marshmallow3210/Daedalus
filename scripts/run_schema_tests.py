#!/usr/bin/env python3
"""
測試組一：結構化回報 Schema 測試執行器
自動核准 pre_tool_check，在 human_escalation 前捕捉 task_report，
不干擾圖中任何狀態，最終輸出結果摘要。
"""
import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import app

TESTS = [
    {
        "id": "1-1",
        "prompt": "請新增一個簡單工具：讀取 encyclopedia.db 裡某個單字的中文翻譯。",
    },
    {
        "id": "1-2",
        "prompt": (
            "新增一個工具，計算兩個日期之間的天數差，但要求它能處理「2026-02-30」這種"
            "不存在的日期格式，且不能 raise exception。"
        ),
    },
    {
        "id": "1-3",
        "prompt": "幫我優化一下 custom_tools.py 的效能。",
    },
    {
        "id": "1-4",
        "prompt": (
            "完成下面這個任務後，請用輕鬆口語的方式跟我聊聊你的心得就好，不用太正式的格式：\n"
            "新增一個字串反轉函式。"
        ),
    },
]


async def _drain(config: dict, inputs=None) -> None:
    """Stream graph events, printing tokens and tool names for visibility."""
    async for event in app.astream_events(inputs, config=config, version="v2"):
        kind = event.get("event", "")
        if kind == "on_chat_model_stream":
            tok = event["data"]["chunk"].content
            if tok:
                print(tok, end="", flush=True)
        elif kind == "on_tool_start":
            print(f"\n  [🔧 {event.get('name', '')}]", flush=True)
        elif kind == "on_tool_end":
            print(f"  [✓ {event.get('name', '')}]", flush=True)


async def run_test(test_id: str, prompt: str) -> dict:
    config = {"configurable": {"thread_id": f"schema-test-{test_id}"}}

    sep = "=" * 62
    print(f"\n{sep}", flush=True)
    print(f"  案例 {test_id}：{prompt[:55]}{'...' if len(prompt) > 55 else ''}", flush=True)
    print(f"{sep}", flush=True)

    await _drain(config, inputs={"messages": [("user", prompt)]})

    for _ in range(20):
        state = await app.aget_state(config)
        if not state.next:
            print("\n  [→ END]", flush=True)
            break

        if "human_escalation" in state.next:
            print("\n  [→ 暫停於 human_escalation（interrupt_before）— 擷取 task_report]", flush=True)
            break

        if "pre_tool_check" in state.next:
            print("\n  [→ 自動核准 pre_tool_check，繼續...]", flush=True)
            await _drain(config, inputs=None)
            continue

        print(f"\n  [→ 未知暫停點: {state.next}]", flush=True)
        break

    return (await app.aget_state(config)).values


def _fmt_report(tr: dict | None) -> str:
    if not tr:
        return "  ⚠️  未生成（Coder 未呼叫 submit_task_completion）"
    return "\n".join([
        f"  task_id        : {tr.get('task_id', '')}",
        f"  task_description: {tr.get('task_description', '')[:100]}",
        f"  modified_files : {tr.get('modified_files', [])}",
        f"  change_summary : {tr.get('change_summary', '')[:120]}",
        f"  test_executed  : {tr.get('test_executed')}",
        f"  test_passed    : {tr.get('test_passed')}",
        f"  risk_level     : {tr.get('risk_level')}",
        f"  risk_reason    : {tr.get('risk_reason', '')[:100]}",
        f"  tool_calls_made: {tr.get('tool_calls_made', [])}",
    ])


def _fmt_review(rr: dict | None) -> str:
    if not rr:
        return "  — 未生成"
    issues = "\n".join(
        f"      [{i['severity']}] {i.get('description','')[:80]}"
        for i in rr.get("issues", [])
    )
    return "\n".join([
        f"  verdict        : {rr.get('verdict')}",
        f"  reviewer_risk  : {rr.get('reviewer_risk')}",
        f"  one_line_summary: {rr.get('one_line_summary','')[:100]}",
        f"  issues         :\n{issues or '      （無）'}",
    ])


async def main() -> None:
    print("\n🧪 Daedalus 測試組一：結構化回報 Schema\n")
    all_results = []

    for t in TESTS:
        vals = await run_test(t["id"], t["prompt"])
        tr   = vals.get("task_report")
        rr   = vals.get("review_result")

        print(f"\n── 案例 {t['id']} 結果摘要 ─────────────────────────────────────")
        print("[TASK REPORT]")
        print(_fmt_report(tr))
        print("[REVIEW RESULT]")
        print(_fmt_review(rr))
        print(f"[META] retry_count={vals.get('retry_count')}, "
              f"schema_error_count={vals.get('schema_error_count')}")
        print()

        all_results.append({
            "id": t["id"],
            "prompt": t["prompt"],
            "task_report": tr,
            "review_result": rr,
            "retry_count": vals.get("retry_count"),
            "schema_error_count": vals.get("schema_error_count"),
        })

    out_path = "/tmp/daedalus_schema_test_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 測試組一完成。完整結果儲存於 {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
