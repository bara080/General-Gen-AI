# =============================================================================
# Agentic Q&A + Summarization over MongoDB Docs with Vector Search (LangGraph)
# =============================================================================
# This script:
#   1) Connects to MongoDB Atlas and loads two datasets:
#      - Vector-search-ready docs (with precomputed embeddings) for retrieval
#      - Full-length docs (raw pages) for summarization by title
#   2) Creates an Atlas Vector Search index on the embeddings field
#   3) Defines two Tools (vector Q&A + page-by-title fetch) for an LLM agent
#   4) Builds a LangGraph with an agent node and a tool-execution node
#   5) Demonstrates execution with and without MongoDB-backed memory
#
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

# =============================================================================
# 0) Environment & Path Setup
# =============================================================================

# Add parent directory to import shared utilities (utils.py)
sys.path.append(os.path.join(os.path.dirname(os.getcwd())))

# -----------------------------------------------------------------------------
# MongoDB Connection
# -----------------------------------------------------------------------------
# If you are using your own MongoDB Atlas cluster, set MONGODB_URI in env.
MONGODB_URI = os.environ.get("MONGODB_URI")
mongodb_client = MongoClient(MONGODB_URI)
# Quick connectivity check
mongodb_client.admin.command("ping")

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

# Track progress (kept as-is)
track_progress("vs_index_creation", "ai_agents_lab")

# =============================================================================
# 4) Tools (VoyageAI embeddings + MongoDB vector search + page fetch)
# =============================================================================

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
    # Use contextualized_embed with input_type="query"
    embds_obj = vo.contextualized_embed(
        inputs=[[query]],
        model="voyage-context-3",
        input_type="query",
    )
    # Extract the single query embedding
    embeddings = embds_obj.results[0].embeddings[0]
    return embeddings

@tool
def get_information_for_question_answering(user_query: str) -> str:
    """
    Retrieve relevant information using Atlas Vector Search.

    Vector search over VS_COLLECTION_NAME with top-5 results returned as a
    concatenated string (bodies only).

    Args:
        user_query: Freeform user question.

    Returns:
        Concatenated 'body' fields (string) of the retrieved docs.
    """
    # 1) Embed the user query
    query_embedding = get_embeddings(user_query)

    # 2) Build the vector search pipeline
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
        {
            "$project": {
                "_id": 0,
                "body": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    # 3) Run aggregation and return concatenated bodies
    results = vs_collection.aggregate(pipeline)
    context = "\n\n".join([doc.get("body") for doc in results])
    return context

@tool
def get_page_content_for_summarization(user_query: str) -> str:
    """
    Retrieve the raw 'body' of a documentation page by its title.

    Args:
        user_query: Document title.

    Returns:
        Page content (string) if found; otherwise a not-found message.
    """
    query = {"title": user_query}
    projection = {"_id": 0, "body": 1}
    document = full_collection.find_one(query, projection)
    if document:
        return document["body"]
    else:
        return "Document not found"

# Tool registry used by the agent
tools = [
    get_information_for_question_answering,
    get_page_content_for_summarization,
]

# Quick sanity tests (non-empty responses expected)
get_information_for_question_answering.invoke(
    "What are some best practices for data backups in MongoDB?"
)
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
llm_with_tools.invoke(
    ["Give me a summary of the page titled Create a MongoDB Deployment."]
).tool_calls
llm_with_tools.invoke(
    ["What are some best practices for data backups in MongoDB?"]
).tool_calls

# =============================================================================
# 6) Graph Nodes (Agent + Tool Node)
# =============================================================================

def agent(state: GraphState) -> Dict[str, List]:
    """
    Agent node: runs the LLM with tool-binding over incoming messages.

    Args:
        state: Graph state with 'messages'.

    Returns:
        Dict updating 'messages' with the agent's latest output.
    """
    messages = state["messages"]
    result = llm_with_tools.invoke(messages)
    return {"messages": [result]}

# Name→tool map for quick dispatch
tools_by_name = {tool.name: tool for tool in tools}
pprint(tools_by_name)

def tool_node(state: GraphState) -> Dict[str, List]:
    """
    Tool node: executes any tool calls emitted by the agent.

    Args:
        state: Graph state with 'messages' (last message may contain tool_calls).

    Returns:
        Dict updating 'messages' with ToolMessage observations.
    """
    result = []
    tool_calls = state["messages"][-1].tool_calls

    # Example tool_call structure:
    # {
    #   "name": "get_information_for_question_answering",
    #   "args": {"user_query": "What are Atlas Triggers"},
    #   "id": "call_123...",
    #   "type": "tool_call",
    # }
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))

    return {"messages": result}

# =============================================================================
# 7) Conditional Routing
# =============================================================================

def route_tools(state: GraphState):
    """
    Router for conditional edges:
    - If last AI message has tool calls → route to 'tools'
    - Else → END
    """
    messages = state.get("messages", [])
    if len(messages) > 0:
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

# =============================================================================
# 8) Build & Compile Graph
# =============================================================================

graph = StateGraph(GraphState)

# Nodes
graph.add_node("agent", agent)
graph.add_node("tools", tool_node)

# Edges
graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")

# Conditional edges from 'agent'
graph.add_conditional_edges(
    "agent",
    route_tools,
    {"tools": "tools", END: END},
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

    Args:
        user_input: User question/prompt.
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

    Args:
        thread_id: Unique thread identifier for the conversation.
        user_input: User question/prompt.
    """
    config = {"configurable": {"thread_id": thread_id}}
    for step in app.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,  # pass thread config
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

# Example: same thread gets history-aware replies
execute_graph_with_memory(
    "1",
    "What are some best practices for data backups in MongoDB?",
)
execute_graph_with_memory(
    "1",
    "What did I just ask you?",
)
