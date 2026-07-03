import asyncio
import os
import chainlit as cl
from langchain_core.messages import ToolMessage
from agent import app as graph, DANGEROUS_TOOL_NAMES


# ──────────────────────────────────────────────────────────────
# Action callbacks (Chainlit 2.x requires explicit registration)
# ──────────────────────────────────────────────────────────────

def _fire_action(value: str) -> None:
    """Called from action callbacks to resolve the pending _ask_actions wait."""
    event: asyncio.Event = cl.user_session.get("_pending_action_event")
    if event:
        cl.user_session.set("_pending_action_value", value)
        event.set()


@cl.action_callback("approve")
async def _cb_approve(action: cl.Action):
    _fire_action("approve")


@cl.action_callback("reject")
async def _cb_reject(action: cl.Action):
    _fire_action("reject")


@cl.action_callback("resume")
async def _cb_resume(action: cl.Action):
    _fire_action("resume")


@cl.action_callback("abandon")
async def _cb_abandon(action: cl.Action):
    _fire_action("abandon")


async def _ask_actions(content: str, actions: list, timeout: int = 900) -> str | None:
    """
    Send a message with action buttons and wait for one to be clicked.
    Returns the action name that was clicked, or None on timeout.
    """
    event = asyncio.Event()
    cl.user_session.set("_pending_action_event", event)
    cl.user_session.set("_pending_action_value", None)
    await cl.Message(content=content, actions=actions).send()
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        return cl.user_session.get("_pending_action_value")
    except asyncio.TimeoutError:
        return None
    finally:
        cl.user_session.set("_pending_action_event", None)


@cl.on_chat_start
async def start():
    # Each browser session gets its own LangGraph thread so state is isolated.
    cl.user_session.set("thread_id", f"daedalus_{cl.context.session.id}")
    await cl.Message(
        content="🕶️ **Daedalus 系統已上線**。特務 marshmallow3210，請下達指令。"
    ).send()


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _config() -> dict:
    return {"configurable": {"thread_id": cl.user_session.get("thread_id")}}


async def _display_inline_reviewer_card(rr: dict, retry_count: int) -> None:
    """
    Real-time reviewer summary card — fires after each reviewer_agent run
    without pausing the loop. Shown as an informational message only;
    the loop continues automatically to context_surgeon → agent.
    """
    verdict       = rr.get("verdict", "unknown")
    reviewer_risk = rr.get("reviewer_risk", "?")
    _risk = {"low": "🟢", "medium": "🟡", "high": "🔴"}
    _sev  = {"blocker": "🔴", "major": "🟠", "minor": "🟡"}

    if verdict == "approve":
        aspects = rr.get("approved_aspects") or []
        fallback = rr.get("one_line_summary") or "程式碼審查通過"
        aspects_txt = "\n".join(f"- {a}" for a in aspects) or f"- {fallback}"
        minor_issues = [i for i in rr.get("issues", []) if i.get("severity") == "minor"]
        minor_txt = ""
        if minor_issues:
            minor_txt = "\n\n**次要建議（不影響核准）：**\n" + "\n".join(
                f"- {i['description']}" for i in minor_issues
            )
        card = (
            f"## ✅ 第 {retry_count} 輪 Reviewer 審查通過\n\n"
            f"**Reviewer 風險評估**：{_risk.get(reviewer_risk,'❓')} {reviewer_risk}\n\n"
            f"**確認 OK 的面向：**\n{aspects_txt}"
            f"{minor_txt}"
        )
    else:
        issues = rr.get("issues", [])
        issues_txt = "\n\n".join(
            f"{_sev.get(i['severity'],'⚪')} **[{i['severity']}]** `{i.get('category','')}` — {i['description']}"
            + (f"\n   📍 {i['location']}" if i.get("location") else "")
            + (f"\n   💡 {i['suggestion']}" if i.get("suggestion") else "")
            for i in issues
        ) or "（無具體問題清單）"
        card = (
            f"## 🔄 第 {retry_count} 輪 Reviewer 要求修正\n\n"
            f"**結論**：{rr.get('one_line_summary','')}\n\n"
            f"**問題清單：**\n\n{issues_txt}\n\n"
            f"*Coder 正在自動修正，無需等待……*"
        )

    await cl.Message(content=card).send()


async def _stream_graph(inputs, config: dict, msg: cl.Message) -> None:
    """Stream graph events into msg, with real-time forge progress and reviewer cards."""
    _coder_round = 0
    async for event in graph.astream_events(inputs, config=config, version="v2"):
        kind = event.get("event", "")
        name = event.get("name", "")

        if kind == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            if token:
                await msg.stream_token(token)

        elif kind == "on_chain_start" and name == "agent":
            _coder_round += 1
            label = "初始分析" if _coder_round == 1 else f"第 {_coder_round} 輪修正"
            await cl.Message(content=f"---\n🤖 **Coder {label}**").send()

        elif kind == "on_chain_start" and name == "reviewer_agent":
            await cl.Message(
                content="🔍 **Reviewer 開始審查** — 評估面向：邏輯正確性、安全性、邊界條件、是否符合任務需求"
            ).send()

        elif kind == "on_chain_start" and name == "tools":
            await cl.Message(content="🔧 **Coder 正在執行工具**，請稍候…").send()

        elif kind == "on_chain_end" and name == "tools":
            tool_msgs = ((event.get("data") or {}).get("output") or {}).get("messages") or []
            for tm in tool_msgs:
                content = getattr(tm, "content", "") or ""
                if "[DISK_WRITE_RESULT]" not in content:
                    continue
                ast_icon  = "✅" if "[AST_RESULT]PASS"          in content else "❌"
                test_icon = "✅" if "[TEST_RESULT]PASS"         in content else "❌"
                if "[DISK_WRITE_RESULT]SUCCESS" in content:
                    disk_icon, disk_label = "✅", "寫入成功"
                elif "[DISK_WRITE_RESULT]FAIL" in content:
                    disk_icon, disk_label = "❌", "寫入失敗"
                else:
                    disk_icon, disk_label = "⏭️", "跳過（測試未通過）"
                await cl.Message(
                    content=f"🔍 **forge 結果** — AST {ast_icon} | 測試 {test_icon} | 磁碟 {disk_icon} {disk_label}"
                ).send()

        elif kind == "on_chain_end" and name == "reviewer_agent":
            output = (event.get("data") or {}).get("output") or {}
            rr     = output.get("review_result")
            rc     = output.get("retry_count", 0)
            if rr and isinstance(rr, dict):
                await _display_inline_reviewer_card(rr, rc)


async def _display_pipeline_cards(config: dict) -> None:
    """
    After each graph run, display structured cards if a new review result exists.
    Shows: task report summary card + reviewer result card.
    Deduplicates by task_id to avoid re-displaying on subsequent turns.
    """
    state  = await graph.aget_state(config)
    vals   = state.values
    rr     = vals.get("review_result")
    tr     = vals.get("task_report") or {}
    if not rr or not tr:
        return

    task_id = tr.get("task_id", "")
    if cl.user_session.get("_last_reviewed_task_id") == task_id:
        return
    cl.user_session.set("_last_reviewed_task_id", task_id)

    _risk  = {"low": "🟢", "medium": "🟡", "high": "🔴"}
    _sev   = {"blocker": "🔴", "major": "🟠", "minor": "🟡"}

    # ── Task report summary card ──────────────────────────────────────────────
    files_txt = "\n".join(f"  - `{f}`" for f in tr.get("modified_files", [])) or "  - （無改動）"
    test_icon = "✅" if tr.get("test_passed") else ("❌" if tr.get("test_executed") else "—")
    coder_risk = tr.get("risk_level", "?")
    report_card = (
        f"## 📋 任務回報\n\n"
        f"**目標**：{tr.get('task_description', '')}\n\n"
        f"**改動檔案**\n{files_txt}\n\n"
        f"**測試狀態**：{test_icon}　"
        f"**風險等級**：{_risk.get(coder_risk, '❓')} {coder_risk} — {tr.get('risk_reason', '')}\n\n"
        f"**使用工具**：{', '.join(tr.get('tool_calls_made', []))}"
    )
    await cl.Message(content=report_card).send()

    # ── Reviewer result card ──────────────────────────────────────────────────
    verdict       = rr.get("verdict", "unknown")
    reviewer_risk = rr.get("reviewer_risk", "?")
    retry_count   = vals.get("retry_count", 0)

    if verdict == "approve":
        aspects      = rr.get("approved_aspects") or []
        fallback     = rr.get("one_line_summary") or "程式碼審查通過"
        approved_txt = "\n".join(f"- {a}" for a in aspects) or f"- {fallback}"
        minor_issues = [i for i in rr.get("issues", []) if i.get("severity") == "minor"]
        minor_txt = ""
        if minor_issues:
            minor_txt = "\n\n**次要建議（不影響核准）：**\n" + "\n".join(
                f"- [minor] {i['description']}" for i in minor_issues
            )
        review_card = (
            f"## ✅ Reviewer 審查通過\n\n"
            f"**Reviewer 評估風險**：{_risk.get(reviewer_risk, '❓')} {reviewer_risk}　"
            f"**Coder 自評**：{_risk.get(coder_risk, '❓')} {coder_risk}\n\n"
            f"**確認 OK 的面向：**\n{approved_txt}"
            f"{minor_txt}"
        )
    else:
        issues = rr.get("issues", [])
        issues_txt = "\n\n".join(
            f"{_sev.get(i['severity'], '⚪')} **[{i['severity']}]** `{i.get('category','')}` — {i['description']}"
            + (f"\n   📍 {i['location']}" if i.get("location") else "")
            + (f"\n   💡 {i['suggestion']}" if i.get("suggestion") else "")
            for i in issues
        )
        review_card = (
            f"## ⚠️ Reviewer 要求修正（第 {retry_count} 次 / 共 3 次）\n\n"
            f"**結論**：{rr.get('one_line_summary', '')}\n\n"
            f"**問題清單：**\n\n{issues_txt}"
        )

    await cl.Message(content=review_card).send()


async def _display_completion_summary(config: dict) -> None:
    """Final one-card summary shown after a successful task (task_reporter_status=ok)."""
    state  = await graph.aget_state(config)
    vals   = state.values
    tr     = vals.get("task_report") or {}
    rr     = vals.get("review_result") or {}
    status = vals.get("task_reporter_status", "")
    if status != "ok":
        return

    retry_count = vals.get("retry_count") or 0
    total_iter  = vals.get("total_iterations") or 0
    verdict     = rr.get("verdict", "")
    files       = tr.get("modified_files") or []
    files_txt   = "、".join(f"`{f}`" for f in files) if files else "（無磁碟改動）"

    if verdict == "approve":
        result_line = "✅ Reviewer 核准，任務成功完成"
    else:
        result_line = "⚠️ 任務結束（Reviewer 最後狀態：未核准）"

    rounds_txt = f"共 {total_iter} 輪 Coder 執行"
    if retry_count:
        rounds_txt += f"，Reviewer 審查 {retry_count} 次後核准"
    else:
        rounds_txt += "，Reviewer 首輪即核准"

    summary = (
        f"## 🏁 任務完成總結\n\n"
        f"**完成內容**：{tr.get('task_description', '（無記錄）')}\n\n"
        f"**改動檔案**：{files_txt}\n\n"
        f"**執行過程**：{rounds_txt}\n\n"
        f"**最終結果**：{result_line}"
    )
    await cl.Message(content=summary).send()


async def _handle_escalation(config: dict, state) -> None:
    """
    Handle human_escalation interrupt: show audit info and let operator decide
    to resume (graph runs human_escalation → END, user then re-issues task)
    or abandon (same path, different label).
    """
    vals             = state.values
    tr               = vals.get("task_report") or {}
    rr               = vals.get("review_result") or {}
    retry_count      = vals.get("retry_count") or 0
    schema_errors    = vals.get("schema_error_count") or 0
    total_iterations = vals.get("total_iterations") or 0

    if total_iterations >= 6:
        reason = f"任務迭代次數超過上限（{total_iterations} 次 Coder 輪次），強制中止以防無限迴圈"
    elif schema_errors >= 2:
        reason = f"submit_task_completion schema 連續解析失敗（{schema_errors} 次）"
    elif retry_count >= 3:
        reason = f"Reviewer 審查連續不通過（{retry_count} 次重試）"
    else:
        coder_risk    = tr.get("risk_level", "?")
        reviewer_risk = rr.get("reviewer_risk", "?")
        reason        = f"高風險任務（Coder 自評：{coder_risk}，Reviewer 評估：{reviewer_risk}）"

    task_desc      = tr.get("task_description", "")
    review_summary = rr.get("one_line_summary", "")

    content = (
        f"🚨 **需要人工介入**\n\n"
        f"**原因**：{reason}\n"
        + (f"**任務**：{task_desc}\n"          if task_desc      else "")
        + (f"**審查結論**：{review_summary}\n" if review_summary else "")
        + "\n請選擇後續動作："
    )

    chosen = await _ask_actions(
        content=content,
        actions=[
            cl.Action(name="resume",  payload={"value": "resume"},  label="▶️ 繼續（人工修正後重新下達指令）"),
            cl.Action(name="abandon", payload={"value": "abandon"}, label="🛑 放棄此任務"),
        ],
        timeout=900,
    )

    if chosen is None:
        await cl.Message(
            content=(
                "⏰ **人工介入確認逾時（15 分鐘未回應）**\n\n"
                "基於安全原則，逾時一律**視為放棄**，任務已終止。\n"
                "如需繼續，請重新下達指令。"
            )
        ).send()

    # Advance through human_escalation → END regardless of choice.
    resume_msg = cl.Message(content="")
    await _stream_graph(None, config, resume_msg)
    if resume_msg.content:
        await resume_msg.update()

    if chosen == "resume":
        await cl.Message(content="▶️ 已轉交人工，請重新下達指令。").send()
    elif chosen == "abandon":
        await cl.Message(content="🛑 任務已放棄。請重新下達新任務。").send()


async def _handle_interrupt(config: dict) -> None:
    """
    After a graph run, check whether the graph paused at pre_tool_check
    (dangerous tool pending) or human_escalation (review/risk escalation).
    """
    state = await graph.aget_state(config)
    if not state.next:
        return

    if "human_escalation" in state.next:
        await _handle_escalation(config, state)
        return

    if "pre_tool_check" not in state.next:
        return

    last_msg   = state.values["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", []) or []
    tool_names = [tc["name"] for tc in tool_calls]

    chosen = await _ask_actions(
        content=(
            f"⚠️ **需要人工核准**\n\n"
            f"即將執行敏感操作：**{'**、**'.join(tool_names)}**\n\n"
            f"請確認後選擇是否執行。"
        ),
        actions=[
            cl.Action(name="approve", payload={"value": "approve"}, label="✅ 核准執行"),
            cl.Action(name="reject",  payload={"value": "reject"},  label="❌ 拒絕取消"),
        ],
        timeout=900,
    )

    if chosen is None:
        await cl.Message(
            content=(
                "⏰ **危險操作確認逾時（15 分鐘未回應）**\n\n"
                "基於安全原則，逾時一律**視為拒絕**，操作已取消。\n"
                "如需執行，請重新下達指令。"
            )
        ).send()

    if chosen == "approve":
        # Resume — the graph will execute the pending tool(s) and continue.
        resume_msg = cl.Message(content="")
        await _stream_graph(None, config, resume_msg)
        await resume_msg.update()
        # Recurse: there may be another dangerous tool further in the chain.
        await _handle_interrupt(config)

    else:
        # Explicit reject or timeout — both treated as cancel (safe default).
        # Immediately confirm to the user so they know their choice was registered,
        # regardless of what the agent says next.
        await cl.Message(
            content="❌ **已拒絕執行** — 危險操作已取消，不會執行。"
        ).send()

        cancel_msgs = [
            ToolMessage(
                content=(
                    "⛔ 此操作已被使用者明確拒絕，禁止重試此工具。\n"
                    "請停止嘗試執行此危險操作，並等候使用者的全新指令。"
                ),
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
        await graph.aupdate_state(
            config,
            {"messages": cancel_msgs},
            as_node="tools",
        )
        cancel_reply = cl.Message(content="")
        await _stream_graph(None, config, cancel_reply)
        if cancel_reply.content:
            await cancel_reply.update()
        # Handle any follow-up interrupt (e.g. agent attempts another dangerous tool).
        await _handle_interrupt(config)


# ──────────────────────────────────────────────────────────────
# Main message handler
# ──────────────────────────────────────────────────────────────

@cl.on_message
async def main(message: cl.Message):
    config = _config()
    inputs = {"messages": [("user", message.content)]}
    msg    = cl.Message(content="")

    try:
        await _stream_graph(inputs, config, msg)
        await msg.update()
        await _handle_interrupt(config)
        await _display_pipeline_cards(config)
        await _display_completion_summary(config)

    except Exception as e:
        msg.content = f"❌ 系統錯誤：\n```\n{e}\n```"
        await msg.update()
