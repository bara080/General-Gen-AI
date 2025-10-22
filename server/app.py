# server/app.py
import os
import dotenv
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from sse_starlette.sse import EventSourceResponse

# import your compiled LangGraph app
# ensure agenticai.py exposes `ai_app` (the compiled graph object)

from genai.agenticai import ai_app as graph_app

from server.utils_agent_build import compiled_app as graph_app


# env defaults
url = os.environ.setdefault("MONGODB_URI", os.getenv("MONGODB_URI"))


print(url)
exit(1)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskIn(BaseModel):
    query: str
    thread_id: Optional[str] = None

def _run_once(query: str, thread_id: Optional[str] = None) -> str:
    cfg = {"configurable": {"thread_id": thread_id}} if thread_id else {}
    last = ""
    for state in graph_app.stream({"messages": [{"role": "user", "content": query}]}, cfg, stream_mode="values"):
        msg = state["messages"][-1]
        last = getattr(msg, "content", "") if isinstance(msg, dict) else msg.content
    return last

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/qa")
def qa(inp: AskIn):
    return {"answer": _run_once(inp.query, inp.thread_id)}

@app.post("/summarize")
def summarize(inp: AskIn):
    return {"summary": _run_once(f"Summarize: {inp.query}", inp.thread_id)}

@app.post("/hybrid")
def hybrid(inp: AskIn):
    return {"answer": _run_once(f"Use hybrid_search then answer: {inp.query}", inp.thread_id)}

@app.post("/stream")
def stream(inp: AskIn):
    def gen():
        cfg = {"configurable": {"thread_id": inp.thread_id}} if inp.thread_id else {}
        for state in graph_app.stream({"messages": [{"role": "user", "content": inp.query}]}, cfg, stream_mode="values"):
            msg = state["messages"][-1]
            yield {"event": "token", "data": getattr(msg, "content", "")}
    return EventSourceResponse(gen())
