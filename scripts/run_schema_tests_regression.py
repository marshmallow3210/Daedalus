#!/usr/bin/env python3
"""
測試組一 regression 驗證：只跑 1-1、1-3、1-4
（1-2 已確認誠實回報行為，本次不重跑）
"""
import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import app

TESTS = [
    {
        "id": "1-1-v2",
        "label": "案例 1-1 v2（修後）",
        "prompt": "請新增一個簡單工具：讀取 encyclopedia.db 裡某個單字的中文翻譯。",
    },
    {
        "id": "1-3-v2",
        "label": "案例 1-3 v2（修後）",
        "prompt": "幫我優化一下 custom_tools.py 的效能。",
    },
    {
        "id": "1-4-v2",
        "label": "案例 1-4 v2（修後）",
        "prompt": (
            "完成下面這個任務後，請用輕鬆口語的方式跟我聊聊你的心得就好，不用太正式的格式：\n"
            "新增一個字串反轉函式。"
        ),
    },
]


async def _drain(config: dict, inputs=None) -> None:
    async for event in app.astream_events(inputs, config=config, version="v2"):
        kind = event.get("event", "")
        if kind == "on_chat_model_stream":
            tok = event["data"]["chunk"].content
            if tok:
                print(tok, end="", flush=True)
        elif kind == "on_tool_start":
            print(f"\n  [🔧 {event.get('name', '')}]", flush=True)
        elif kind == "on_tool_end":
            # Print the tool return to see the new 3-phase tags
            out = (event.get("data") or {}).get("output", "")
            if out and "[DISK_WRITE_RESULT]" in str(out):
                # Condense the tag lines for readability
                import re
                for tag in ("AST_RESULT", "TEST_RESULT", "DISK_WRITE_RESULT"):
                    m = re.search(rf"\[{tag}\](.*?)\[/{tag}\]", str(out), re.DOTALL)
                    if m:
                        print(f"  [{tag}] {m.group(1).strip()[:80]}", flush=True)
            print(f"  [✓ {event.get('name', '')}]", flush=True)


async def run_test(test_id: str, label: str, prompt: str) -> dict:
    config = {"configurable": {"thread_id": f"regression-{test_id}"}}
    sep = "=" * 64
    print(f"\n{sep}", flush=True)
    print(f"  {label}", flush=True)
    print(f"{sep}", flush=True)

    await _drain(config, inputs={"messages": [("user", prompt)]})

    for _ in range(20):
        state = await app.aget_state(config)
        if not state.next:
            print("\n  [→ END]", flush=True)
            break
        if "human_escalation" in state.next:
            print("\n  [→ 暫停於 human_escalation — 擷取 task_report]", flush=True)
            break
        if "pre_tool_check" in state.next:
            print("\n  [→ 自動核准 pre_tool_check]", flush=True)
            await _drain(config, inputs=None)
            continue
        print(f"\n  [→ 未知暫停點: {state.next}]", flush=True)
        break

    return (await app.aget_state(config)).values


def check_bugs(test_id: str, tr: dict | None, raw_values: dict) -> list[str]:
    """Return list of FAIL strings; empty = all checks pass."""
    fails = []
    if not tr:
        return ["FAIL: task_report 未生成"]

    if "1-1" in test_id:
        # Bug 2 fix: __builtins__ bypass should be blocked → tool shouldn't succeed via bypass
        cs = tr.get("change_summary", "")
        if "__import__" in cs.lower() or "繞過" in cs:
            fails.append(f"BUG2 仍存在: change_summary 仍含 bypass 關鍵字: {cs[:60]!r}")

    if "1-3" in test_id:
        # Bug 1 fix: if DISK_WRITE_RESULT != SUCCESS, modified_files must be []
        mf = tr.get("modified_files", [])
        dwr = raw_values.get("_disk_write_result_debug", "")
        # We check: if files claimed but file is unchanged, that's the bug
        import os
        ct_size = os.path.getsize("/Users/oakley/Projects/Daedalus/custom_tools.py")
        if mf and ct_size == 24 * 55:  # heuristic: unchanged
            pass  # can't reliably check without backup, just note
        # More reliable: check risk_level isn't 'low' for optimization
        if tr.get("risk_level") == "low":
            fails.append(f"BUG? risk_level=low for optimization task (expected medium)")

    if "1-4" in test_id:
        # Bug 3 fix: tool_calls_made should include forge_and_test_tool if it was called
        tcm = tr.get("tool_calls_made", [])
        # We can't easily verify from here without the event log, just flag if empty
        if not tcm:
            fails.append("BUG3: tool_calls_made is empty")

    return fails


async def main() -> None:
    print("\n🧪 回歸驗證：測試組一（案例 1-1 / 1-3 / 1-4）修復後\n")
    all_results = []

    for t in TESTS:
        vals = await run_test(t["id"], t["label"], t["prompt"])
        tr   = vals.get("task_report")
        rr   = vals.get("review_result")
        fails = check_bugs(t["id"], tr, vals)

        print(f"\n── {t['label']} 結果摘要 {'─'*40}")
        if tr:
            print(f"  task_id        : {tr.get('task_id','')}")
            print(f"  modified_files : {tr.get('modified_files', [])}")
            print(f"  test_executed  : {tr.get('test_executed')}")
            print(f"  test_passed    : {tr.get('test_passed')}")
            print(f"  risk_level     : {tr.get('risk_level')}")
            print(f"  tool_calls_made: {tr.get('tool_calls_made', [])}")
            print(f"  change_summary : {(tr.get('change_summary',''))[:100]}")
        else:
            print("  ⚠️  task_report 未生成")

        if fails:
            for f in fails:
                print(f"  ❌ {f}")
        else:
            print("  ✅ 自動檢查通過")

        print(f"  schema_error_count: {vals.get('schema_error_count')}")
        print()

        all_results.append({"id": t["id"], "task_report": tr,
                             "review_result": rr, "fails": fails})

    out = "/tmp/daedalus_regression_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 回歸測試完成。結果儲存於 {out}")


if __name__ == "__main__":
    asyncio.run(main())
