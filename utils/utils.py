from pymongo.errors import OperationFailure
from pymongo.collection import Collection
from langchain_aws import ChatBedrock
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
from typing import Dict, List
import time
import os
from dotenv import dotenv_values, find_dotenv

SLEEP_TIMER = 5

def create_index(collection: Collection, index_name: str, model: Dict) -> None:
    """
    Create a search index.

    Args:
        collection (Collection): Target MongoDB collection.
        index_name (str): Name of the search index.
        model (Dict): Index definition document.
    """
    try:
        print(f"Creating the {index_name} index")
        collection.create_search_index(model=model)
    except OperationFailure:
        print(f"{index_name} index already exists, recreating...")
        try:
            print(f"Dropping {index_name} index")
            collection.drop_search_index(name=index_name)

            # Poll for index deletion to complete
            while True:
                indexes = list(collection.list_search_indexes())
                index_exists = any(idx.get("name") == index_name for idx in indexes)
                if not index_exists:
                    print(f"{index_name} index deletion complete")
                    break
                print(f"Waiting for {index_name} index deletion to complete...")
                time.sleep(SLEEP_TIMER)

            print(f"Creating new {index_name} index")
            collection.create_search_index(model=model)
            print(f"Successfully recreated the {index_name} index")
        except Exception as e:
            raise Exception(f"Error during index recreation: {str(e)}")


def check_index_ready(collection: Collection, index_name: str) -> None:
    """
    Poll until the specified search index is READY.

    Args:
        collection (Collection): Target MongoDB collection.
        index_name (str): Name of the search index to check.
    """
    while True:
        indexes = list(collection.list_search_indexes())
        matching_indexes = [idx for idx in indexes if idx.get("name") == index_name]

        if not matching_indexes:
            print(f"{index_name} index not found")
            time.sleep(SLEEP_TIMER)
            continue

        index = matching_indexes[0]
        status = index["status"]
        if status == "READY":
            print(f"{index_name} index status: READY")
            print(f"{index_name} index definition: {index['latestDefinition']}")
            break

        print(f"{index_name} index status: {status}")
        time.sleep(SLEEP_TIMER)


def track_progress(task: str, workshop_id: str) -> None:
    """
    Track progress of a task.

    Args:
        task (str): Task identifier.
        workshop_id (str): Workshop or lab identifier.
    """
    print(f"Tracking progress for task {task}")
    #payload = {"task": task, "workshop_id": workshop_id, "sandbox_id": SANDBOX_NAME}
    #requests.post(url=PROXY_ENDPOINT, json={"task": "track_progress", "data": payload})



def set_env(providers: List[str], passkey: str = "", env_path: str | None = None) -> Dict[str, str]:
    """
    Load env vars from a local .env file and export only those
    matching a provider-specific prefix: <PROVIDER>_*

    Example keys in .env:
      OPENAI_API_KEY=...
      AZURE_OPENAI_ENDPOINT=...
      GOOGLE_API_KEY=...
      OPENAI_API_KEY=..
      VOYAGEAI_API_KEY=

    Usage:
      set_env(["openai", "azure_openai", "google"])
    """
    path = env_path or find_dotenv(usecwd=True)
    env = dotenv_values(path)
    applied: Dict[str, str] = {}

    for provider in providers:
        prefix = f"{provider.upper()}_"
        for k, v in env.items():
            if k.startswith(prefix) and v is not None:
                os.environ[k] = v
                applied[k] = v

    return applied


def get_llm(provider: str):
    """
    Return a chat LLM client for the requested provider.

    Args:
        provider (str): One of 'openai', 'aws', 'google', 'microsoft'.

    Returns:
        A LangChain chat model compatible client.
    """
    if provider == "openai":
        return ChatOpenAI(model="gpt-4o", temperature=0)
    elif provider == "aws":
        return ChatBedrock(
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
            model_kwargs=dict(temperature=0),
            region_name="us-west-2",
        )
    elif provider == "google":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0,
        )
    elif provider == "microsoft":
        return AzureChatOpenAI(
            azure_endpoint="https://gai-326.openai.azure.com/",
            azure_deployment="gpt-4o",
            api_version="2023-06-01-preview",
            temperature=0,
        )
    else:
        raise Exception("Unsupported provider. provider can be one of 'openai', 'aws', 'google', 'microsoft'.")
