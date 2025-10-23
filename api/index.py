import os
import sys
from dotenv import load_dotenv

# --- FIX: Add 'src' to system path for correct module imports ---
# Vercel needs to find the 'src' directory, which contains 'utils/agenticai.py'.
if 'src' not in sys.path:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    src_path = os.path.join(project_root, 'src')
    sys.path.append(src_path)
# ---------------------------------------------------------------

# Corrected Import: Now imports from src.utils.agenticai
try:
    from src.utils.agenticai import ai_app as graph_app 
except ImportError as e:
    print(f"FATAL ERROR: Could not import graph app: {e}")
    graph_app = None


# -----------------------------------------------------------------------------
# Load environment variables
# -----------------------------------------------------------------------------
load_dotenv()


# -----------------------------------------------------------------------------
# Normalize MongoDB connection URL: FIX to prevent Vercel build failure
# -----------------------------------------------------------------------------
uri = os.getenv("MONGODB_URL") or os.getenv("MONGODB_URI")
if not uri:
    # SOFT FIX: This prevents the Vercel build from crashing.
    print("WARNING: Missing MONGODB_URL/MONGODB_URI in environment. Setting placeholder.")
    uri = "mongodb://localhost:27017/placeholder" 
    
os.environ["MONGODB_URL"] = uri

# -----------------------------------------------------------------------------
# Disable data seeding and demo runs by default
# -----------------------------------------------------------------------------
os.environ.setdefault("SEED_DATA", "0")
os.environ.setdefault("RUN_DEMO", "0")
os.environ.setdefault("RUN_MEMORY_DEMO", "0")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from sse_starlette.sse import EventSourceResponse
    
# -----------------------------------------------------------------------------
# FastAPI setup
# We use root_path="/api" because Vercel routes all /api/* requests to this function.
# -----------------------------------------------------------------------------
app = FastAPI(root_path="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    message content.
    """
    if graph_app is None:
        print("ERROR: LangGraph application is not loaded.")
        return "Internal Error: Server function could not initialize."

    cfg = {"configurable": {"thread_id": thread_id}} if thread_id else {}
    last = ""
    for state in graph_app.stream(
        {"messages": [{"role": "user", "content": query}]},
        cfg,
        stream_mode="values",
    ):
        msg = state["messages"][-1]
        last = getattr(msg, "content", msg.get("content", ""))
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
        if graph_app is None:
            yield {"event": "error", "data": "Internal Error: Server function could not initialize."}
            return
            
        cfg = {"configurable": {"thread_id": inp.thread_id}} if inp.thread_id else {}
        for state in graph_app.stream(
            {"messages": [{"role": "user", "content": inp.query}]},
            cfg,
            stream_mode="values",
        ):
            msg = state["messages"][-1]
            content = getattr(msg, "content", msg.get("content", ""))
            if content:
                 yield {"event": "token", "data": content}
                 
    return EventSourceResponse(gen())
