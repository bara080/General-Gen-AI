# server/app.py
import os
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from sse_starlette.sse import EventSourceResponse

# reuse same env/graph build as your script
os.environ.setdefault("MONGODB_URI", "<your-atlas-uri>")
os.environ.setdefault("LEXICAL_INDEX_NAME", "lexical")  # optional

# --- minimal inline build mirroring ai-agent.py (no imports from your demo-run code) ---
# If you prefer, paste your compiled-graph setup here (agent, tools, route, app = graph.compile()).
# For brevity assume you placed a small helper in utils to return `compiled_app` and `get_embeddings`.
from utils_agent_build import compiled_app as graph_app  # <- create once, returns compiled LangGraph app

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to your Vercel domain in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskIn(BaseModel):
    query: str
    thread_id: Optional[str] = None  # to enable memory

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
