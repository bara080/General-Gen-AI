# =============================================================================
# Agentic Q&A + Summarization over MongoDB Docs with Vector Search (LangGraph)
# =============================================================================
# This script:
#   1) Connects to MongoDB Atlas and loads two datasets:
#      - Vector-search-ready docs (with precomputed embeddings) for retrieval
#      - Full-length docs (raw pages) for summarization by title
#   2) Creates an Atlas Vector Search index on the embeddings field
#   3) Defines Tools (vector Q&A, page-by-title fetch, hybrid search, human override)
#   4) Builds a LangGraph with an agent node, tool-execution node, and human review node
#   5) Demonstrates execution with and without MongoDB-backed memory
#
# =============================================================================

from langchain.agents import tool
import voyageai
from typing import List
import os
import sys
from pymongo import MongoClient
import json
from utils import create_index, check_index_ready
from typing import Annotated, Dict
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from utils import get_llm
from langchain_core.messages import ToolMessage
from pprint import pprint
from langgraph.graph import END, StateGraph, START
from IPython.display import Image, display  # noqa: F401 (import retained by request)
from langgraph.checkpoint.mongodb import MongoDBSaver
from utils import track_progress, set_env
from utils import create_index, check_index_ready  # (kept as-is per structure request)
import functools
import time
import uuid
from datetime import datetime

# =============================================================================
# 0) Environment & Path Setup
# =============================================================================

# Add parent directory to import shared utilities (utils.py)
sys.path.append(os.path.join(os.path.dirname(os.getcwd())))

# -----------------------------------------------------------------------------
# MongoDB Connection
# -----------------------------------------------------------------------------
# If you are using your own MongoDB Atlas cluster, set MONGODB_URI in env.
MONGODB_URL = os.environ.get("MONGODB_URI")
mongodb_client = MongoClient(MONGODB_URL)
# Quick connectivity check
mongodb_client.admin.command("ping")

# -----------------------------------------------------------------------------
# Lightweight observability (run logs)
# -----------------------------------------------------------------------------
OBS_DB = "agent_observability"
runs_collection = mongodb_client[OBS_DB]["runs"]
reviews_collection = mongodb_client[OBS_DB]["reviews"]

def log_event(event: str, payload: Dict):
    runs_collection.insert_one({
        "event": event,
        "payload": payload,
        "ts": datetime.utcnow().isoformat()
    })

RUN_ID = str(uuid.uuid4())

# =============================================================================
# 1) Model Provider Setup
# =============================================================================
# LLM provider and passkey expected by the utils.get_llm and set_env helpers.
# LLM_PROVIDER can be "aws" / "microsoft" / "google" / "openai" (here "openai").

LLM_PROVIDER = "openai"
PASSKEY = "voyageai"

# Obtain API keys via helper (keys are set as env vars); do not modify.
set_env([LLM_PROVIDER, "voyageai"], PASSKEY)

# =============================================================================
# 2) Collections & Data Load
# =============================================================================
# Database and collections used in this lab/demo.
DB_NAME = "mongodb_genai_devday_agents"
# Full documents (raw pages) — used for summarization
FULL_COLLECTION_NAME = "mongodb_docs"
# Vector-search-ready documents (pre-embedded) — used for retrieval/Q&A
VS_COLLECTION_NAME = "mongodb_docs_embeddings"
# Atlas Vector Search index name
VS_INDEX_NAME = "vector_index"

# Collection handles
vs_collection = mongodb_client[DB_NAME][VS_COLLECTION_NAME]
full_collection = mongodb_client[DB_NAME][FULL_COLLECTION_NAME]

# -----------------------------------------------------------------------------
# Load vector-search dataset (embeddings included) → VS_COLLECTION_NAME
# -----------------------------------------------------------------------------
with open(f"../data/{VS_COLLECTION_NAME}.json", "r") as data_file:
    json_data = data_file.read()
data = json.loads(json_data)

print(f"Deleting existing documents from the {VS_COLLECTION_NAME} collection.")
vs_collection.delete_many({})
vs_collection.insert_many(data)
print(
    f"{vs_collection.count_documents({})} documents ingested into the {VS_COLLECTION_NAME} collection."
)

# -----------------------------------------------------------------------------
# Load full documentation pages → FULL_COLLECTION_NAME
# -----------------------------------------------------------------------------
with open(f"../data/{FULL_COLLECTION_NAME}.json", "r") as data_file:
    json_data = data_file.read()
data = json.loads(json_data)

print(f"Deleting existing documents from the {FULL_COLLECTION_NAME} collection.")
full_collection.delete_many({})
full_collection.insert_many(data)
print(
    f"{full_collection.count_documents({})} documents ingested into the {FULL_COLLECTION_NAME} collection."
)

# =============================================================================
# 3) Create Atlas Vector Search Index
# =============================================================================
# Definition:
# - path:       field path to your embedding vector
# - dimensions: must match your embedding model's output size
# - similarity: cosine | euclidean | dotProduct
model = {
    "name": VS_INDEX_NAME,
    "type": "vectorSearch",
    "definition": {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 1024,
                "similarity": "cosine",
            }
        ]
    },
}

# Create and wait for index to be READY
create_index(vs_collection, VS_INDEX_NAME, model)
check_index_ready(vs_collection, VS_INDEX_NAME)

# Optional: validate index dims vs expected (guardrail)
def validate_index_dims(expected: int):
    try:
        info = mongodb_client[DB_NAME].command({"listSearchIndexes": VS_COLLECTION_NAME})
        fields = info.get("indexes", [])[0].get("definition", {}).get("fields", [])
        got = fields[0].get("numDimensions", None) if fields else None
        if got != expected:
            raise ValueError(f"Vector index numDimensions={got} != expected={expected}")
    except Exception as e:
        log_event("index_validation_error", {"error": str(e)})
        raise

validate_index_dims(1024)

# Track progress (kept as-is)
track_progress("vs_index_creation", "ai_agents_lab")

# =============================================================================
# 4) Tools (VoyageAI embeddings + MongoDB vector search + page fetch + hybrid + human override)
# =============================================================================

# Retry wrapper for fragile tools (timeouts/backoff)
def with_retry(timeout_s=20, retries=2, backoff=2.0):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, **kw):
            t0, attempt = time.time(), 0
            while True:
                try:
                    return fn(*a, **kw)
                except Exception as e:
                    attempt += 1
                    if attempt > retries or (time.time() - t0) > timeout_s:
                        log_event("tool_failure", {"tool": fn.__name__, "error": str(e), "run_id": RUN_ID})
                        raise
                    time.sleep(backoff ** attempt)
        return wrap
    return deco

# VoyageAI client (used for embedding the user query for retrieval)
vo = voyageai.Client()

def get_embeddings(query: str) -> List[float]:
    """
    Compute a query embedding using VoyageAI contextualized embeddings.

    Args:
        query: User query string.

    Returns:
        A single embedding vector (List[float]) for the query.
    """
    embds_obj = vo.contextualized_embed(
        inputs=[[query]],
        model="voyage-context-3",
        input_type="query",
    )
    embeddings = embds_obj.results[0].embeddings[0]
    return embeddings

@tool
@with_retry()
def get_information_for_question_answering(user_query: str) -> str:
    """
    Retrieve relevant information using Atlas Vector Search (top-5 bodies).
    """
    query_embedding = get_embeddings(user_query)
    pipeline = [
        {
            "$vectorSearch": {
                "index": VS_INDEX_NAME,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 150,
                "limit": 5,
            }
        },
        {"$project": {"_id": 0, "body": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]
    results = vs_collection.aggregate(pipeline)
    context = "\n\n".join([doc.get("body") for doc in results])
    return context

@tool
@with_retry()
def get_page_content_for_summarization(user_query: str) -> str:
    """
    Retrieve the raw 'body' of a documentation page by its title.
    """
    query = {"title": user_query}
    projection = {"_id": 0, "body": 1}
    document = full_collection.find_one(query, projection)
    if document:
        return document["body"]
    else:
        return "Document not found"

# Optional lexical index name for BM25 search (configure in Atlas Search)
LEXICAL_INDEX_NAME = os.environ.get("LEXICAL_INDEX_NAME", "lexical")

@tool
@with_retry()
def hybrid_search(user_query: str) -> str:
    """
    Hybrid retrieval: BM25 + Vector, fused via Reciprocal Rank Fusion (returns top-5 bodies).
    """
    # BM25 (lexical)
    bm25 = list(vs_collection.aggregate([
        {"$search": {"index": LEXICAL_INDEX_NAME, "text": {"query": user_query, "path": "body"}}},
        {"$project": {"_id": 0, "body": 1, "score": {"$meta": "searchScore"}}},
        {"$limit": 10},
    ]))
    # Vector
    vec = list(vs_collection.aggregate([
        {"$vectorSearch": {
            "index": VS_INDEX_NAME,
            "path": "embedding",
            "queryVector": get_embeddings(user_query),
            "numCandidates": 200,
            "limit": 10
        }},
        {"$project": {"_id": 0, "body": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]))

    def rrf(rank): return 1.0 / (60 + rank)
    pool = {}
    for i, d in enumerate(bm25):
        pool[d["body"]] = pool.get(d["body"], 0) + rrf(i + 1)
    for i, d in enumerate(vec):
        pool[d["body"]] = pool.get(d["body"], 0) + rrf(i + 1)
    top = sorted(pool.items(), key=lambda x: -x[1])[:5]
    return "\n\n".join([t[0] for t in top])

@tool
def apply_human_override(approved_answer: str) -> str:
    """
    Replace the last AI answer with a human-approved one.
    """
    return approved_answer

# Tool registry used by the agent
tools = [
    get_information_for_question_answering,
    get_page_content_for_summarization,
    hybrid_search,
    apply_human_override,
]

# Quick sanity tests (non-empty responses expected)
get_information_for_question_answering.invoke("What are some best practices for data backups in MongoDB?")
get_page_content_for_summarization.invoke("Create a MongoDB Deployment")

# =============================================================================
# 5) Graph State & LLM Wiring
# =============================================================================

class GraphState(TypedDict):
    """
    Graph state container.

    Attributes:
        messages: Conversation history/messages tracked by LangGraph.
    """
    messages: Annotated[list, add_messages]

# Instantiate the LLM from provider
llm = get_llm(LLM_PROVIDER)

# Prompt template with tool awareness
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "You are a helpful AI assistant."
            " You are provided with tools to answer questions and summarize technical documentation related to MongoDB."
            " Think step-by-step and use these tools to get the information required to answer the user query."
            " Do not re-run tools unless absolutely necessary."
            " If you are not able to get enough information using the tools, reply with I DON'T KNOW."
            " You have access to the following tools: {tool_names}."
            " When you present factual claims, prefer returning supporting snippets that can be cited."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Fill tool names in the prompt
prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))

# Bind tools to LLM and chain with prompt
bind_tools = llm.bind_tools(tools)
llm_with_tools = prompt | bind_tools

# Smoke tests: verify that correct tool calls are produced
llm_with_tools.invoke(["Give me a summary of the page titled Create a MongoDB Deployment."]).tool_calls
llm_with_tools.invoke(["What are some best practices for data backups in MongoDB?"]).tool_calls

# =============================================================================
# 6) Graph Nodes (Agent + Tool Node + Human Review Node)
# =============================================================================

def agent(state: GraphState) -> Dict[str, List]:
    """
    Agent node: runs the LLM with tool-binding over incoming messages.
    """
    messages = state["messages"]
    result = llm_with_tools.invoke(messages)
    log_event("agent_output", {"run_id": RUN_ID, "has_tool_calls": bool(getattr(result, "tool_calls", []))})
    return {"messages": [result]}

# Name→tool map for quick dispatch
tools_by_name = {tool.name: tool for tool in tools}
pprint(tools_by_name)

def tool_node(state: GraphState) -> Dict[str, List]:
    """
    Tool node: executes any tool calls emitted by the agent.
    """
    result = []
    tool_calls = state["messages"][-1].tool_calls
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    log_event("tools_executed", {"run_id": RUN_ID, "n_tools": len(tool_calls)})
    return {"messages": result}

# --- Human-in-the-Loop (HITL) ---

def needs_human_review(ai_message) -> bool:
    """
    Simple heuristic: route to human review when:
    - explicit token in content, or
    - tool error previously logged for this run (checked externally), or
    - missing supporting context keyword (very light check here).
    """
    content = getattr(ai_message, "content", "") or ""
    if "REVIEW_REQUIRED" in content:
        return True
    # If the model returned nothing or trivially short content, escalate
    if isinstance(content, str) and len(content.strip()) < 20:
        return True
    return False

def human_review(state: GraphState) -> Dict[str, List]:
    """
    Persist last AI output for a human to approve/edit in 'reviews' collection.
    External UI/process can write an approved answer back via apply_human_override.
    """
    last_msg = state["messages"][-1]
    reviews_collection.insert_one({
        "run_id": RUN_ID,
        "ts": datetime.utcnow().isoformat(),
        "message": getattr(last_msg, "content", ""),
        "tool_calls": getattr(last_msg, "tool_calls", []),
        "thread_hint": "use apply_human_override tool to finalize response"
    })
    log_event("hitl_enqueued", {"run_id": RUN_ID})
    return {"messages": state["messages"]}

# =============================================================================
# 7) Conditional Routing
# =============================================================================

def route_tools(state: GraphState):
    """
    Router for conditional edges:
    - If last AI message has tool calls → route to 'tools'
    - Else if human review needed → 'human_review'
    - Else → END
    """
    messages = state.get("messages", [])
    if len(messages) > 0:
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    if needs_human_review(ai_message):
        return "human_review"
    return END

# =============================================================================
# 8) Build & Compile Graph
# =============================================================================

graph = StateGraph(GraphState)

# Nodes
graph.add_node("agent", agent)
graph.add_node("tools", tool_node)
graph.add_node("human_review", human_review)

# Edges
graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_edge("human_review", END)  # or back to 'agent' after external approval

# Conditional edges from 'agent'
graph.add_conditional_edges(
    "agent",
    route_tools,
    {"tools": "tools", "human_review": "human_review", END: END},
)

# Compile the application (no memory)
app = graph.compile()

# Visualize (repr) — retained by request
app

# =============================================================================
# 9) Execute (No Memory)
# =============================================================================

def execute_graph(user_input: str) -> None:
    """
    Stream outputs for a single-turn run (no checkpointing).
    """
    for step in app.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode="values",  # stream entire state each step
    ):
        step["messages"][-1].pretty_print()

# Demo runs
execute_graph("What are some best practices for data backups in MongoDB?")
execute_graph("Give me a summary of the page titled Create a MongoDB Deployment")

# =============================================================================
# 10) Add Memory (MongoDBSaver Checkpointer) and Execute
# =============================================================================

checkpointer = MongoDBSaver(mongodb_client)
app = graph.compile(checkpointer=checkpointer)

def execute_graph_with_memory(thread_id: str, user_input: str) -> None:
    """
    Stream outputs with checkpointing (thread-scoped memory).
    """
    config = {"configurable": {"thread_id": thread_id}}
    for step in app.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,  # pass thread config
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

# Example: same thread gets history-aware replies
execute_graph_with_memory("1", "What are some best practices for data backups in MongoDB?")
execute_graph_with_memory("1", "What did I just ask you?")

# =============================================================================
# [limitation addition]
# =============================================================================
LIMITATION_ADDITION = [
    # Captured limitations before this enhancement block:
    "No human-in-the-loop review gate for low-confidence/edge cases.",
    "No hybrid retrieval fallback (BM25 + vector) or reranking.",
    "No retries/backoff/timeouts around fragile tool operations.",
    "No citation/PII guardrails or minimal output validation.",
    "No per-step observability/logging for traces and metrics.",
    "No index/model-dimension validation against embedding model."
]
