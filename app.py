import re
import chainlit as cl
from langchain_core.messages import ToolMessage
from agent import app as graph, FORCE_ESCALATION_MARKER

# Per-task bookkeeping fields reset at the start of every new user turn so a
# previous task's report/review/counters can't leak into the next one.
_PER_TASK_RESET = {
    "task_report":          None,
    "review_result":        None,
    "retry_count":          0,
    "schema_error_count":   0,
    "total_iterations":     0,
    "last_task_iterations": 0,
    "original_task":        "",
    "task_reporter_status": "",
    "retry_messages":       None,
}


# ──────────────────────────────────────────────────────────────
# Human-in-the-loop action prompt
# ──────────────────────────────────────────────────────────────

async def _ask_actions(content: str, actions: list, timeout: int = 900) -> str | None:
    """
    Ask the user to pick an action and block until they click.

    Uses Chainlit's native ask protocol (AskActionMessage), NOT a plain
    cl.Message with actions. This is load-bearing: the escalation prompt fires
    while the @cl.on_message handler is still running (we're paused at a graph
    interrupt), and Chainlit renders action buttons on an ordinary message as
    DISABLED for the whole duration of an active run — so the previous
    asyncio.Event approach left the 放棄/核准 buttons greyed out and un-clickable
    in the real UI (no POST /project/action ever fired). AskActionMessage marks
    the buttons as an awaited question, which the frontend keeps enabled mid-run.

    Returns the chosen action's name, or None on timeout.
    """
    res = await cl.AskActionMessage(
        content=content,
        actions=actions,
        timeout=timeout,
    ).send()
    return res.get("name") if res else None


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


_CJK_RE = re.compile(r"[一-鿿぀-ヿ가-힯]")


def _clean_model_text(content, has_tool_calls: bool) -> str:
    """
    Normalise a Coder message for display and drop stray template fragments.
    Local models (gemma via Ollama) occasionally leak end-of-turn artifacts —
    lone tokens like 'end' / <end_of_turn> / short foreign-language junk —
    during tool-call rounds; rendered raw they show up as "Raw code" boxes.
    """
    if isinstance(content, list):
        content = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    text = (content or "").replace("<end_of_turn>", "").replace("<start_of_turn>", "")
    text = text.strip()
    if not text:
        return ""
    core = text.strip("`'\" \n\t")
    if core.lower() in {"end", "'end'", "end_of_turn", "eot"}:
        return ""
    # Tool-call rounds with a short non-CJK fragment → leaked template noise.
    if has_tool_calls and len(core) < 12 and not _CJK_RE.search(core):
        return ""
    return text


async def _display_inline_reviewer_card(rr: dict, review_round: int) -> None:
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
            f"## ✅ 第 {review_round} 輪 Reviewer 審查通過\n\n"
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
            f"## 🔄 第 {review_round} 輪 Reviewer 要求修正\n\n"
            f"**結論**：{rr.get('one_line_summary','')}\n\n"
            f"**問題清單：**\n\n{issues_txt}\n\n"
            f"*Coder 正在自動修正，無需等待……*"
        )

    await cl.Message(content=card).send()


async def _stream_graph(inputs, config: dict) -> None:
    """
    Stream graph events into per-round messages, with real-time forge progress
    and reviewer cards.

    Each Coder round gets its OWN message directly under its round header —
    never a shared accumulator. Tokens are accepted only from the "agent" node
    (metadata filter), so Reviewer output or routing internals can never bleed
    into the chat, and each round's text is replaced with the sanitised final
    content when the round ends (drops leaked template fragments like 'end').
    """
    _coder_round  = 0
    _review_round = 0
    round_msg: cl.Message | None = None   # current Coder round's streaming target

    async for event in graph.astream_events(inputs, config=config, version="v2"):
        kind = event.get("event", "")
        name = event.get("name", "")
        meta = event.get("metadata") or {}

        if kind == "on_chat_model_stream":
            if meta.get("langgraph_node") != "agent":
                continue        # only the Coder's own tokens belong in the chat
            token = event["data"]["chunk"].content
            if isinstance(token, list):
                token = "".join(
                    b.get("text", "") if isinstance(b, dict) else str(b) for b in token
                )
            if token:
                if round_msg is None:
                    round_msg = cl.Message(content="")
                await round_msg.stream_token(token)

        elif kind == "on_chain_start" and name == "agent":
            _coder_round += 1
            label = "初始分析" if _coder_round == 1 else f"第 {_coder_round} 輪修正"
            await cl.Message(content=f"---\n🤖 **Coder {label}**").send()
            round_msg = None

        elif kind == "on_chain_end" and name == "agent":
            out_msgs = ((event.get("data") or {}).get("output") or {}).get("messages") or []
            last     = out_msgs[-1] if out_msgs else None
            tool_calls = getattr(last, "tool_calls", None) or []
            final_text = _clean_model_text(getattr(last, "content", "") if last else "", bool(tool_calls))
            if not final_text and tool_calls:
                final_text = "*（Coder 未附說明，直接呼叫工具：" + "、".join(
                    f"`{tc['name']}`" for tc in tool_calls
                ) + "）*"
            if round_msg is not None:
                if final_text:
                    round_msg.content = final_text
                    await round_msg.update()
                else:
                    await round_msg.remove()
                round_msg = None
            elif final_text:
                await cl.Message(content=final_text).send()

        elif kind == "on_chain_start" and name == "reviewer_agent":
            _review_round += 1
            st = (event.get("data") or {}).get("input") or {}
            tr = (st.get("task_report") if isinstance(st, dict) else None) or {}
            files_txt = "、".join(f"`{f}`" for f in (tr.get("modified_files") or [])) or "（無磁碟改動）"
            await cl.Message(content=(
                f"🔍 **Reviewer 第 {_review_round} 輪審查開始**\n\n"
                f"- **評估目標**：{tr.get('task_description') or '（未提供）'}\n"
                f"- **宣告改動檔案**：{files_txt}\n"
                f"- **Coder 自評風險**：{tr.get('risk_level', '?')} — {tr.get('risk_reason', '')}\n"
                f"- **檢查項目**：邏輯正確性、安全漏洞、邊界條件、"
                f"測試真實性（對照 AST / TEST / DISK 機器標籤）、是否符合原始任務範圍\n\n"
                f"*獨立審查中，完成後自動繼續，無需人工介入……*"
            )).send()

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
            if rr and isinstance(rr, dict):
                await _display_inline_reviewer_card(rr, _review_round)


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
    # total_iterations is reset to 0 on submit; the display value is snapshotted
    total_iter  = vals.get("last_task_iterations") or vals.get("total_iterations") or 0
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

    msgs        = vals.get("messages") or []
    last_ai     = getattr(msgs[-1], "content", "") if msgs else ""
    forced_test = "[FORCE_ESCALATION]" in (last_ai or "")

    if forced_test:
        reason = f"測試開關 {FORCE_ESCALATION_MARKER} 觸發（人工驗證用，非真實風險）"
    elif total_iterations >= 6:
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
    await _stream_graph(None, config)

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
        await _stream_graph(None, config)
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
        await _stream_graph(None, config)
        # Handle any follow-up interrupt (e.g. agent attempts another dangerous tool).
        await _handle_interrupt(config)


# ──────────────────────────────────────────────────────────────
# Main message handler
# ──────────────────────────────────────────────────────────────

@cl.on_message
async def main(message: cl.Message):
    config = _config()
    # Reset per-task bookkeeping alongside the new message so the previous
    # task's report/review/counters can't leak (stale completion card, wrong
    # original_task anchor, premature hard-cap/escalation from carried counts).
    # These channels use last-value semantics; messages uses add_messages, so
    # the new turn is appended while the rest is overwritten atomically at entry.
    inputs = {"messages": [("user", message.content)], **_PER_TASK_RESET}
    # Fresh task → allow its completion/review cards to render again.
    cl.user_session.set("_last_reviewed_task_id", None)

    try:
        await _stream_graph(inputs, config)
        await _handle_interrupt(config)
        await _display_pipeline_cards(config)
        await _display_completion_summary(config)

    except Exception as e:
        await cl.Message(
            content=f"❌ 系統錯誤（{type(e).__name__}）：\n```\n{e}\n```"
        ).send()
