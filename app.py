import os
import chainlit as cl
from langchain_core.messages import ToolMessage
from agent import app as graph, DANGEROUS_TOOL_NAMES


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


async def _stream_graph(inputs, config: dict, msg: cl.Message) -> None:
    """Stream graph events into msg, token by token."""
    async for event in graph.astream_events(inputs, config=config, version="v2"):
        if event.get("event") == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            if token:
                await msg.stream_token(token)


async def _handle_interrupt(config: dict) -> None:
    """
    After a graph run, check whether the graph paused at pre_tool_check
    (i.e., a dangerous tool is pending).  Show an AskActionMessage and
    either resume or inject a cancellation ToolMessage.
    """
    state = await graph.aget_state(config)
    if not state.next or "pre_tool_check" not in state.next:
        return

    last_msg    = state.values["messages"][-1]
    tool_calls  = getattr(last_msg, "tool_calls", []) or []
    tool_names  = [tc["name"] for tc in tool_calls]

    res = await cl.AskActionMessage(
        content=(
            f"⚠️ **需要人工核准**\n\n"
            f"即將執行敏感操作：**{'**、**'.join(tool_names)}**\n\n"
            f"請確認後選擇是否執行。"
        ),
        actions=[
            cl.Action(name="approve", value="approve", label="✅ 核准執行"),
            cl.Action(name="reject",  value="reject",  label="❌ 拒絕取消"),
        ],
        timeout=300,
    ).send()

    if res and res.get("value") == "approve":
        # Resume — the graph will execute the pending tool(s) and continue.
        resume_msg = cl.Message(content="")
        await _stream_graph(None, config, resume_msg)
        await resume_msg.update()
        # Recurse: there may be another dangerous tool further in the chain
        # (e.g., delete_local_video after upload_to_youtube).
        await _handle_interrupt(config)

    else:
        # Cancel: inject ToolMessage(s) so the agent sees "operation cancelled"
        # as the tool output and can respond gracefully.
        cancel_msgs = [
            ToolMessage(
                content="操作已被使用者取消。請繼續等候指示。",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
        # Update state as-if the tools node ran (so graph continues to agent).
        await graph.aupdate_state(
            config,
            {"messages": cancel_msgs},
            as_node="tools",
        )
        cancel_reply = cl.Message(content="")
        await _stream_graph(None, config, cancel_reply)
        await cancel_reply.update()


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
        # Check whether the graph paused waiting for human approval.
        await _handle_interrupt(config)

    except Exception as e:
        msg.content = f"❌ 系統錯誤：\n```\n{e}\n```"
        await msg.update()
