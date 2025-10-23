# server/app.py

import os
import sys
from dotenv import load_dotenv
from agenticai import ai_app

BASE_DIR = os.path.dirname(__file__)          
SRC_DIR  = os.path.join(BASE_DIR, "src")     
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ... rest of your FastAPI code unchanged ...


# -----------------------------------------------------------------------------
# Load environment variables BEFORE importing the graph app, so agenticai
# picks up the right config (e.g., MONGODB_URL / keys).
# -----------------------------------------------------------------------------
#load
load_dotenv()



# -----------------------------------------------------------------------------
# Normalize MongoDB connection URL: support both MONGODB_URL and MONGODB_URI
# and ensure the one your graph code expects (MONGODB_URL) is set.
# -----------------------------------------------------------------------------
uri = os.getenv("MONGODB_URL") or os.getenv("MONGODB_URI")
if not uri:
    raise RuntimeError("Missing MONGODB_URL (or MONGODB_URI) in environment")
os.environ["MONGODB_URL"] = uri

# -----------------------------------------------------------------------------
# Disable data seeding and demo runs by default on the web server process.
# (You can override these per-process via environment variables.)
# -----------------------------------------------------------------------------
os.environ.setdefault("SEED_DATA", "0")
os.environ.setdefault("RUN_DEMO", "0")
os.environ.setdefault("RUN_MEMORY_DEMO", "0")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from sse_starlette.sse import EventSourceResponse
# from genai.agenticai import ai_app as graph_app
# -----------------------------------------------------------------------------
# Import the compiled LangGraph app ONCE. (Avoid importing utils_agent_build.)
# -----------------------------------------------------------------------------
# try:
#     from src.genai.agenticai import ai_app as graph_app
# except ImportError:
#     from genai.agenticai import ai_app as graph_app
    

# -----------------------------------------------------------------------------
# FastAPI setup
# -----------------------------------------------------------------------------
#app = FastAPI(root_path="/api")



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class AskIn(BaseModel):
    """Request schema for QA/summarize/hybrid endpoints."""
    query: str
    thread_id: Optional[str] = None

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _run_once(query: str, thread_id: Optional[str] = None) -> str:
    """
    Run a single-turn interaction against the graph app and return the last
    message content. If a thread_id is provided, it is passed through LangGraph
    config to enable checkpointing/memory.
    """
    cfg = {"configurable": {"thread_id": thread_id}} if thread_id else {}
    last = ""
    for state in graph_app.stream(
        {"messages": [{"role": "user", "content": query}]},
        cfg,
        stream_mode="values",
    ):
        msg = state["messages"][-1]
        last = getattr(msg, "content", "") if isinstance(msg, dict) else msg.content
    return last

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    """Simple liveness check."""
    return {"ok": True}

@app.post("/qa")
def qa(inp: AskIn):
    """Answer a direct question using the graph app."""
    return {"answer": _run_once(inp.query, inp.thread_id)}

@app.post("/summarize")
def summarize(inp: AskIn):
    """Summarize a query/prompt via the graph app."""
    return {"summary": _run_once(f"Summarize: {inp.query}", inp.thread_id)}

@app.post("/hybrid")
def hybrid(inp: AskIn):
    """Example hybrid mode hint (preserves structure/behavior)."""
    return {"answer": _run_once(f"Use hybrid_search then answer: {inp.query}", inp.thread_id)}

@app.post("/stream")
def stream(inp: AskIn):
    """
    Server-Sent Events streaming endpoint.
    Streams incremental message content from the graph app.
    """
    def gen():
        cfg = {"configurable": {"thread_id": inp.thread_id}} if inp.thread_id else {}
        for state in graph_app.stream(
            {"messages": [{"role": "user", "content": inp.query}]},
            cfg,
            stream_mode="values",
        ):
            msg = state["messages"][-1]
            yield {"event": "token", "data": getattr(msg, "content", "")}
    return EventSourceResponse(gen())
