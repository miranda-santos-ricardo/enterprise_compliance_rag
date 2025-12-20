# src/ingestion/embedder.py
import os
from typing import List

from openai import OpenAI
import chromadb

from ingestion.chunking import TextChunk


def build_or_load_chroma(persist_dir: str) -> chromadb.api.models.Collection.Collection:
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(name="policies")


def index_chunks(
    collection,
    chunks: List[TextChunk],
    model: str,
) -> int:
    """
    Stores chunks in Chroma with deterministic IDs and embeddings from OpenAI.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Cannot embed.")

    oai = OpenAI(api_key=api_key)

    texts = [c.text for c in chunks]
    ids = [f"{c.policy_id}:{c.section_id}" for c in chunks]
    metadatas = [c.metadata for c in chunks]

    # OpenAI embeddings call (batched)
    resp = oai.embeddings.create(
        model=model,
        input=texts,
    )
    embeddings = [d.embedding for d in resp.data]

    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return len(ids)