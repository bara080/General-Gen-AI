# test.py
# Atlas Vector Search sanity test: env, embeddings, index (create+READY), KNN query.
# Works on PyMongo versions WITHOUT search_indexes()/get_search_index helpers.

import os, time, argparse, uuid
from typing import Dict, Any, List
from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from voyageai import Client as Voyage

INDEX_NAME = "vector_index"
EMBED_PATH = "embedding"
MODEL_TEXT = "voyage-3"  # 1024-d

def env() -> Dict[str, str]:
    load_dotenv(find_dotenv(usecwd=True))
    cfg = {
        "MONGODB_URI": os.getenv("MONGODB_URI", ""),
        "DB_NAME": os.getenv("DB_NAME", "zinga-General-purpose"),
        "COLLECTION": os.getenv("COLLECTION", "books"),
        "VOYAGE_API_KEY": (os.getenv("VOYAGE_API_KEY") or "").strip(),
    }
    assert cfg["MONGODB_URI"], "MONGODB_URI missing"
    assert cfg["VOYAGE_API_KEY"], "VOYAGE_API_KEY missing"
    return cfg

def mongo(cfg: Dict[str, str]):
    client = MongoClient(cfg["MONGODB_URI"])
    client.admin.command("ping")
    db = client[cfg["DB_NAME"]]
    col = db[cfg["COLLECTION"]]
    return client, db, col

def _list_search_indexes(col) -> List[Dict[str, Any]]:
    try:
        res = col.database.command({"listSearchIndexes": col.name})
        return res.get("indexes", [])
    except Exception:
        return []

def _drop_search_index(col, name: str):
    try:
        col.database.command({"dropSearchIndex": col.name, "name": name})
    except OperationFailure:
        pass

def ensure_index(col, dims: int) -> Dict[str, Any]:
    desired = {
        "name": INDEX_NAME,
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {"type": "vector", "path": EMBED_PATH, "numDimensions": dims, "similarity": "cosine"}
            ]
        },
    }

    # If exists with wrong dims/path, drop and recreate
    existing = {i["name"]: i for i in _list_search_indexes(col)}
    if INDEX_NAME in existing:
        fld = (existing[INDEX_NAME].get("definition", {}).get("fields") or [{}])[0]
        if fld.get("path") != EMBED_PATH or int(fld.get("numDimensions", -1)) != int(dims):
            _drop_search_index(col, INDEX_NAME)
            existing.pop(INDEX_NAME, None)

    if INDEX_NAME not in existing:
        try:
            col.database.command({"createSearchIndexes": col.name, "indexes": [desired]})
        except OperationFailure:
            pass

    # Wait READY
    t0 = time.time()
    while True:
        idxs = _list_search_indexes(col)
        for idx in idxs:
            if idx.get("name") == INDEX_NAME:
                # Some clusters return status="READY"; others return queryable=True
                status = idx.get("status")
                queryable = idx.get("queryable")
                if status == "READY" or queryable is True:
                    return idx
        if time.time() - t0 > 180:
            raise TimeoutError("index not READY after 180s")
        time.sleep(2)

def embed_text(vo: Voyage, text: str) -> List[float]:
    return vo.embed(texts=[text], model=MODEL_TEXT).embeddings[0]

def upsert_sample(col, vec: List[float], tag: str) -> str:
    _id = f"avs-test:{uuid.uuid4()}"
    col.insert_one({
        "_id": _id,
        "title": "Sample: King Arthur",
        "description": "A man wearing a golden crown holds Excalibur.",
        EMBED_PATH: vec,
        "tag": tag,
    })
    return _id

def knn(col, vec: List[float], k: int = 3) -> List[Dict[str, Any]]:
    pipeline = [
        {
            "$vectorSearch": {
                "index": INDEX_NAME,
                "path": EMBED_PATH,
                "queryVector": vec,
                "numCandidates": 200,
                "limit": k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "title": 1,
                "description": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(col.aggregate(pipeline))

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--query", default="A man wearing a golden crown")
    args = parser.parse_args()

    cfg = env()
    client, db, col = mongo(cfg)
    vo = Voyage(api_key=cfg["VOYAGE_API_KEY"])

    probe_vec = embed_text(vo, "probe")
    dims = len(probe_vec)

    info = ensure_index(col, dims)
    f0 = (info.get("definition", {}).get("fields") or [{}])[0]
    print("INDEX:", info.get("name"), info.get("status", "READY" if info.get("queryable") else ""), "dims:", f0.get("numDimensions"))

    qvec = embed_text(vo, args.query)
    doc_id = upsert_sample(col, qvec, "avs-test")
    print("UPSERTED:", doc_id)

    hits = knn(col, qvec, k=3)
    for i, h in enumerate(hits, 1):
        print(f"{i}. {h.get('title','')} | score={h.get('score'):.5f}")

if __name__ == "__main__":
    main()
