# src/control/grounding_checks.py
import re
from typing import List, Dict, Any

STOPWORDS = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are", "by", "with"}

def keyword_set(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    return {w for w in words if w not in STOPWORDS}

def citations_in_retrieved(claim: Dict[str, Any], retrieved_ids: set[str]) -> List[str]:
    bad = [c for c in claim.get("citations", []) if c not in retrieved_ids]
    return bad

def citation_relevance_heuristic(claim_text: str, cited_texts: List[str], min_overlap: int = 2) -> bool:
    """
    Very simple: claim must share at least N keywords with at least one cited chunk text.
    This kills obvious citation spam.
    """
    ck = keyword_set(claim_text)
    if not ck:
        return False
    for t in cited_texts:
        tk = keyword_set(t)
        if len(ck.intersection(tk)) >= min_overlap:
            return True
    return False