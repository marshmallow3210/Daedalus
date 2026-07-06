"""Drive the running Chainlit app over socket.io exactly like the browser would,
and dump every message the server sends back — to reproduce UI-only bugs."""
import asyncio
import sys
import uuid

import socketio


URL = "http://localhost:8000"
TASK = sys.argv[1] if len(sys.argv) > 1 else "請建立一個回傳兩數之和的工具函式 add(a, b)"
TIMEOUT = float(sys.argv[2]) if len(sys.argv) > 2 else 300


async def main():
    sio = socketio.AsyncClient()
    done = asyncio.Event()
    outputs = []

    @sio.event
    async def connect():
        await sio.emit("connection_successful")

    @sio.on("new_message")
    async def new_message(data):
        content = (data.get("output") or "")
        outputs.append(content)
        print(f"\n===== NEW MESSAGE =====\n{content}", flush=True)

    @sio.on("update_message")
    async def update_message(data):
        content = (data.get("output") or "")
        print(f"\n===== UPDATE MESSAGE =====\n{content}", flush=True)

    @sio.on("stream_start")
    async def stream_start(data):
        print("\n===== STREAM START =====", flush=True)

    @sio.on("stream_token")
    async def stream_token(data):
        print(data.get("token", ""), end="", flush=True)

    sent = False

    @sio.on("task_end")
    async def task_end(data=None):
        if sent:
            done.set()

    session_id = uuid.uuid4().hex
    await sio.connect(
        URL,
        socketio_path="/ws/socket.io",
        auth={
            "sessionId": session_id,
            "clientType": "webapp",
            "threadId": None,
            "userEnv": "{}",
            "chatProfile": None,
        },
    )

    await asyncio.sleep(2)  # let on_chat_start finish

    msg_id = uuid.uuid4().hex
    sent = True
    await sio.emit("client_message", {
        "message": {
            "id": msg_id,
            "threadId": session_id,
            "name": "User",
            "type": "user_message",
            "output": TASK,
            "createdAt": "2026-07-06T00:00:00.000Z",
        },
        "fileReferences": [],
    })

    try:
        await asyncio.wait_for(done.wait(), timeout=TIMEOUT)
        # allow trailing messages (interrupt prompts arrive after task_end sometimes)
        await asyncio.sleep(3)
    except asyncio.TimeoutError:
        print("\n[driver] TIMEOUT waiting for task_end", flush=True)

    await sio.disconnect()
    print("\n[driver] done.", flush=True)


asyncio.run(main())
