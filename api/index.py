import os
import sys
from dotenv import load_dotenv

# Import the graph app. This path assumes 'utils' is in the root directory.
from utils.agenticai import ai_app as graph_app

# -----------------------------------------------------------------------------
# Load environment variables
# -----------------------------------------------------------------------------
load_dotenv()

# -----------------------------------------------------------------------------
# Normalize MongoDB connection URL: Support both URL/URI and prevent build crash
# -----------------------------------------------------------------------------
uri = os.getenv("MONGODB_URL") or os.getenv("MONGODB_URI")
if not uri:
    # Set a placeholder to prevent the build from failing, and print a warning.
    print("WARNING: Missing MONGODB_URL/MONGODB_URI in environment. Setting placeholder.")
    os.environ["MONGODB_URL"] = "mongodb://localhost:27017/placeholder" 
else:
    # Ensure it is set to the name the graph code expects
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
    message content.
    """
    # Defensive check: if graph_app failed to load due to missing dependency/path, 
    # return an error message instead of crashing the function
    if 'graph_app' not in globals() or not graph_app:
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
        # Use .get() for safety and assume content is the field we want
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
        if 'graph_app' not in globals() or not graph_app:
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
                 # Vercel's EventSourceResponse requires data to be a dict that is JSON serializable
                 yield {"event": "token", "data": content}
                 
    # EventSourceResponse expects an iterable of dicts or strings with event/data format
    return EventSourceResponse(gen())
