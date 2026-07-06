#!/usr/bin/env python3
"""
Pre-requisite validation: DaedalusState (MessagesState + new fields) 和
Pydantic model dict 能正確跑過 MemorySaver checkpoint 存 / 讀 round-trip。

通過後才開始實作 Block 1。
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.checkpoint.memory import MemorySaver


# ── 模擬新 DaedalusState ─────────────────────────────────────────────────────

class DaedalusState(MessagesState):
    task_report: Optional[dict]
    review_result: Optional[dict]
    retry_count: int
    schema_error_count: int
    original_task: str
    task_reporter_status: str


# ── 模擬 Pydantic model → dict 序列化 ────────────────────────────────────────

class TaskCompletionReport(BaseModel):
    task_id: str
    task_description: str
    modified_files: list[str]
    change_summary: str
    test_executed: bool
    test_passed: Optional[bool] = None
    risk_level: str
    risk_reason: str
    tool_calls_made: list[str]


SAMPLE_REPORT = TaskCompletionReport(
    task_id="abc12345",
    task_description="新增 send_notification 工具",
    modified_files=["custom_tools.py"],
    change_summary="forge_and_test_tool 新增工具，AST 掃描通過，unittest 2/2 通過",
    test_executed=True,
    test_passed=True,
    risk_level="medium",
    risk_reason="寫入 custom_tools.py，但測試通過",
    tool_calls_made=["forge_and_test_tool", "submit_task_completion"],
)


# ── Graph nodes ───────────────────────────────────────────────────────────────

def set_fields_node(state: DaedalusState) -> dict:
    """模擬 task_reporter 寫入新 state 欄位"""
    return {
        "messages": [AIMessage(content="任務完成")],
        "task_report": SAMPLE_REPORT.model_dump(),
        "review_result": None,
        "retry_count": 0,
        "schema_error_count": 0,
        "original_task": SAMPLE_REPORT.task_description,
        "task_reporter_status": "ok",
    }


def read_fields_node(state: DaedalusState) -> dict:
    """模擬後續節點讀取 state"""
    tr = state.get("task_report")
    assert tr is not None,              "FAIL: task_report 是 None"
    assert tr.get("task_id") == "abc12345", "FAIL: task_id 不符"
    assert tr.get("test_passed") is True,   "FAIL: test_passed 不符"
    assert state.get("review_result") is None, "FAIL: review_result 應為 None"
    assert state.get("retry_count") == 0,      "FAIL: retry_count 應為 0"
    assert state.get("schema_error_count") == 0
    assert state.get("original_task") == SAMPLE_REPORT.task_description
    assert state.get("task_reporter_status") == "ok"
    return {"messages": [AIMessage(content="讀取驗證通過")]}


# ── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    workflow = StateGraph(DaedalusState)
    workflow.add_node("set_fields",  set_fields_node)
    workflow.add_node("read_fields", read_fields_node)
    workflow.set_entry_point("set_fields")
    workflow.add_edge("set_fields", "read_fields")
    workflow.add_edge("read_fields", END)

    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "validate-block1"}}

    # ── Test 1: 所有新欄位都傳入初始 state ──────────────────────────────────
    print("=== Test 1: 完整 initial state ===")
    result = await app.ainvoke(
        {
            "messages":             [HumanMessage(content="test")],
            "task_report":          None,
            "review_result":        None,
            "retry_count":          0,
            "schema_error_count":   0,
            "original_task":        "",
            "task_reporter_status": "",
        },
        config=config,
    )
    _check_result(result, "Test 1")

    # 從 checkpoint 讀回
    state = await app.aget_state(config)
    _check_result(state.values, "Test 1 checkpoint")

    # ── Test 2: 只傳 messages（其他欄位靠 node 寫入）─────────────────────────
    print("\n=== Test 2: 只傳 messages（其他欄位靠 node 寫入）===")
    config2 = {"configurable": {"thread_id": "validate-block1-min"}}
    result2 = await app.ainvoke(
        {"messages": [HumanMessage(content="minimal test")]},
        config=config2,
    )
    _check_result(result2, "Test 2")

    state2 = await app.aget_state(config2)
    _check_result(state2.values, "Test 2 checkpoint")

    print("\n✅  MemorySaver checkpoint 相容性驗證全部通過。可以開始實作 Block 1。")


def _check_result(vals: dict, label: str) -> None:
    tr = vals.get("task_report")
    assert tr is not None,                   f"{label}: task_report 是 None"
    assert tr.get("task_id") == "abc12345",  f"{label}: task_id 不符"
    assert vals.get("review_result") is None, f"{label}: review_result 應為 None"
    assert vals.get("original_task") == SAMPLE_REPORT.task_description
    assert vals.get("task_reporter_status") == "ok"
    print(f"  {label}: task_id={tr.get('task_id')!r}, "
          f"review_result={vals.get('review_result')!r}, "
          f"retry_count={vals.get('retry_count')!r}  ✓")


if __name__ == "__main__":
    asyncio.run(main())
