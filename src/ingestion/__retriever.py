import os

from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Any

load_dotenv()

def embed_query(query: str, model:str) -> List[float]:
    api_key = os.environ.get("OPENAI_API_KEY","")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Cannot embed")
    
    oai = OpenAI(api_key=api_key)
    resp = oai.embeddings.create(model=model, input=query)
    return resp.data[0].embedding

def retrieve_top_k(collection, question: str, embed_model: str, k: int = 5) -> List[Dict[str, Any]]:
    
    q_emb = embed_query(question, embed_model)

    res = collection.query(
        query_embeddings = [q_emb],
        n_results = k,
        include=["documents","metadatas","distances"]
    )
    
    hits = []

    for i in range(len(res["ids"][0])):
        chunk_id = resp["ids"][0][i]
        hits.append(
            {
                "id": chunk_id,
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i]
            }
        )
        
    return hits