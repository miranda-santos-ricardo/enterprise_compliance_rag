# src/main.py
import os
import traceback
import sys

from dotenv import load_dotenv

from agents.answer_agent import AnswerAgent
from agents.policy_agent import PolicyAgent
from ingestion.loader import load_policies
from ingestion.chunking import chunk_text
from ingestion.embedder import build_or_load_chroma, index_chunks
from retrieval.retriever import retrieve_top_k, dedup_hits
from control.grounding_checks import citations_in_retrieved, citation_relevance_heuristic
from control.evaluator import evaluate
from control.hard_gates import run_hard_gates
from control.assumption_gate import classify_assumptions
from control.assumption_detector import detect_assumptions

load_dotenv()


def ensure_indexed(data_dir: str, persist_dir: str, embed_model: str) -> None:
    collection = build_or_load_chroma(persist_dir)

    # Re-index each run while you're iterating (optional)
    # If you want caching later, re-enable the count() early return.
    # if collection.count() > 0:
    #     return

    docs = load_policies(data_dir)

    all_chunks = []
    for d in docs:
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

            pid = h["metadata"].get("policy_id")
            sid = h["metadata"].get("section_id")
            src = h["metadata"].get("file_name")
            print(f"- [{pid}:{sid}] dist={h['distance']:.4f} src={src} | {preview}")

            cid = h["id"]
            txt = (h["text"] or "").strip()
            txt = " ".join(txt.split())
            context_lines.append(f"[{cid}] {txt}")

        # map of retrieved chunk texts
        retrieved_map = {h["id"]: (h["text"] or "") for h in hits}
        retrieved_ids = set(retrieved_map.keys())

        # Generate answer proposal
        agent = AnswerAgent.from_env()
        proposal = agent.run(question, context_lines)

        # Lightweight pre-checks (soft warnings)
        print("\n Verify issues in the answer")
        issues = []
        for i, c in enumerate(proposal.claims):
            bad = citations_in_retrieved(c, retrieved_ids)
            if bad:
                issues.append(f"CLAIM_{i}_CITES_UNKNOWN_IDS:{bad}")

            cited_texts = [retrieved_map[x] for x in c["citations"] if x in retrieved_map]
            if not citation_relevance_heuristic(c["text"], cited_texts):
                issues.append(f"CLAIM_{i}_CITATIONS_LOOK_WEAK")

        if issues:
            print("\n[PreCheck Issues]")
            for it in issues:
                print("-", it)

        #print("\n=== Answer Proposal ===")
        #print(proposal.final_answer)
        #print("\nPROPOSAL:", proposal)
        #if proposal.assumptions:
        #    print("Assumptions:", proposal.assumptions)

        print("\n=== Answer (Summary) ===")
        summary_claim_ids = [0, 1, 3, 4]  # choose the minimal set
        for i in summary_claim_ids:
            c = proposal.claims[i]
            cites = " ".join([f"[{x}]" for x in c.get("citations", [])])
            print(f"- {c['text']} {cites}")

        ###########################################################        
        #print("\n=== DEBUG: Show full text of cited chunks ===")
        #for c in proposal.claims:
        #    for cid in c["citations"]:
        #        txt = retrieved_map.get(cid, "")
        #        print(f"\n--- {cid} ---\n{txt[:1200]}\n")
        ###########################################################


        # HARD GATES (must block SAFE)
        hard_issues = []
        for i, c in enumerate(proposal.claims):
            gate_issues = run_hard_gates(c, retrieved_map)
            for it in gate_issues:
                hard_issues.append(f"CLAIM_{i}:{it}")

        if hard_issues:
            print("\n=== Hard Gate Failures ===")
            for it in hard_issues:
                print("-", it)

        # Policy verification (LLM)
        policy_agent = PolicyAgent.from_env()
        assessment = policy_agent.run(question, context_lines, proposal.claims)

        #detect assumptions
        detected = detect_assumptions(question, proposal.claims, context_lines)
        # Merge + dedup by (type,text)
        merged = proposal.assumptions if isinstance(proposal.assumptions, list) else []
        all_assumptions = merged + detected
        dedup = {}
        for a in all_assumptions:
            key = (a.get("type"), a.get("text"))
            dedup[key] = a
        proposal.assumptions = list(dedup.values())

        #classify assumptions
        override, cap, a_issues = classify_assumptions(proposal.assumptions)
        if assessment.confidence > cap:
            assessment.confidence = cap
        if a_issues:
            assessment.issues.extend(a_issues)

        print("\n=== Assumptions ===")
        if proposal.assumptions:
            for a in proposal.assumptions:
                print(f"- {a['type']} ({a['impact']}): {a['text']}")
        else:
            print("- None")
        print("Override:", override or "(none)")

        # override final decision
        if override == "BLOCK":
            print("\n=== Final Decision Override (Assumptions) ===")
            print("do_not_use")
            for it in a_issues:
                print("-", it)
            return
        

        decision = evaluate(assessment)
        
        print(f"Confidence: {decision.assessment.confidence:.2f}")

        if override == "REVIEW":# and decision.status.value == "safe_to_use":
            print("\n=== Final Decision Override (Assumptions) ===")
            print("review_required")
            for it in a_issues:
                print("-", it)
            return

        # DECISION OVERRIDE: hard gates > LLM verifier
        print("\n=== Hard Gates Summary ===")
        print("Hard issues count:", len(hard_issues))
        if hard_issues:
            print("\n=== Final Decision Override (Hard Gates) ===")
            print("do_not_use")
            for it in hard_issues:
                print("-", it)
            return
        
        #print("\n=== Claims ===")
        #for i, c in enumerate(proposal.claims):
        #    print(f"{i}. {c['text']}")
        #    print(f"   citations: {c.get('citations', [])}")
            
        print("\n=== Policy Verification ===")
        print("Compliant:", assessment.is_compliant)
        print("Risk:", assessment.risk_level.value)
        print("Confidence:", assessment.confidence)
        print("Issues:", assessment.issues)

        print("\n=== Final Decision ===")
        print(decision.status.value)

        if assessment.issues:
            print("Notes:")
            for it in assessment.issues:
                print("-", it)
        else:
            print("- No issues detected. Answer is compliant.")

    except Exception as e:
        print(e)
        traceback.print_exc()


if __name__ == "__main__":
    main()