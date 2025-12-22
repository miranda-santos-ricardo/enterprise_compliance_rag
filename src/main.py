# src/main.py
import os
import traceback
import sys

from dotenv import load_dotenv

from agents.answer_agent import AnswerAgent
from ingestion.loader import load_policies
from ingestion.chunking import chunk_text
from ingestion.embedder import build_or_load_chroma, index_chunks
from retrieval.retriever import retrieve_top_k, dedup_hits


load_dotenv()


def ensure_indexed(data_dir: str, persist_dir: str, embed_model: str) -> None:
    collection = build_or_load_chroma(persist_dir)

    # If already has data, skip (simple heuristic)
    #if collection.count() > 0:
    #    return

    docs = load_policies(data_dir)

    #print("--")
    #print(docs)

    all_chunks = []
    for d in docs:
        #print('222')
        #print(d)
        #print('333')
        chunks = chunk_text(d.policy_id, d.text, d.metadata)
        all_chunks.extend(chunks)

    inserted = index_chunks(collection, all_chunks, model=embed_model)
    print(f"[Index] Inserted {inserted} chunks into Chroma at {persist_dir}")


def main() -> None:
    try:
        if len(sys.argv) < 2:
            print("Usage: python src/main.py \"<question>\"")
            raise SystemExit(1)

        question = sys.argv[1]

        data_dir = os.environ.get("POLICY_DATA_DIR", "data/policies")
        persist_dir = os.environ.get("CHROMA_DIR", "vectorstore/index")
        embed_model = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
        top_k = int(os.environ.get("TOP_K", "5"))

        ensure_indexed(data_dir, persist_dir, embed_model)

        collection = build_or_load_chroma(persist_dir)
        hits = retrieve_top_k(collection, question, embed_model, k=top_k)
        hits = dedup_hits(hits, max_results=top_k)

        print("\n=== Retrieved Policy Excerpts ===")
        context_lines = []
        for h in hits:
            preview = (h["text"] or "").replace("\n", " ")
            preview = preview

            pid = h["metadata"].get("policy_id")
            sid = h["metadata"].get("section_id")
            src = h["metadata"].get("file_name")
            print(f"- [{pid}:{sid}] dist={h['distance']:.4f} src={src} | {preview}")

            cid = h["id"]
            txt = (h["text"] or "").strip()
            txt = " ".join(txt.split()) #normalize whitespace
            context_lines.append(f"[{cid}] {txt}")

            #preview = preview[:220] + ("..." if len(preview) > 220 else "")
#            print(f"- [{h['id']}] dist={h['distance']:.4f} | {preview}")
        agent = AnswerAgent.from_env()
        proposal = agent.run(question, context_lines)

        #get fake citations
        retrived_ids = {h["id"] for h in hits}
        bad = [c for c in proposal.citations if c not in retrived_ids]
        if bad:
            print(f"[Warning] Some citations were not in retrieved excerpts: {bad}")

        print("\n=== Answer Proposal ===")
        print(proposal.answer)
        print("\nCitations:", proposal.citations)
        if proposal.assumptions:
            print("Assumptions:", proposal.assumptions)

    except Exception as e:
        print(e)
        traceback.print_exc()

if __name__ == "__main__":
    main()