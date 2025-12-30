# src/control/hard_gates.py
import re
from typing import List, Dict, Any, Tuple


def extract_numbers(text: str) -> List[str]:
    return re.findall(r"\b\d+\b", text)


def normalize_ws(s: str) -> str:
    return " ".join((s or "").split()).lower()

def canonicalize(text: str) -> str:
    t = normalize_ws(text)
    # normalize hyphens/spaces for common variants
    t = t.replace("pro-rated", "prorated")
    t = t.replace("pro rated", "prorated")
    return t


def must_contain_phrases_gate(claim_text: str, cited_texts: List[str], phrases: List[str]) -> Tuple[bool, str]:
    c = canonicalize(claim_text)
    combined = canonicalize(" ".join(cited_texts))

    needed = [p for p in phrases if p in c]
    missing = [p for p in needed if p not in combined]

    if missing:
        return False, f"UNSUPPORTED_KEY_PHRASE:missing={missing}"
    return True, ""

def must_contain_any_numbers_gate(claim_text: str, cited_texts: List[str]) -> Tuple[bool, str]:
    """
    If the claim includes any numbers (15, 30, etc.), at least one cited chunk must contain them.
    This prevents "15 days" claims being supported by irrelevant citations.
    """
    nums = extract_numbers(claim_text)
    if not nums:
        return True, ""

    combined = normalize_ws(" ".join(cited_texts))
    missing = [n for n in nums if n not in combined]
    if missing:
        return False, f"UNSUPPORTED_NUMBER:missing={missing}"
    return True, ""


def must_contain_numbers_gate(claim_text: str, cited_texts: List[str]) -> Tuple[bool, str]:
    """
    Stronger version for number-heavy claims: if there are 3+ numbers, require ALL present.
    """
    nums = extract_numbers(claim_text)
    if len(nums) < 3:
        return True, ""

    combined = normalize_ws(" ".join(cited_texts))
    missing = [n for n in nums if n not in combined]
    if missing:
        return False, f"UNSUPPORTED_NUMERIC_DETAIL:missing={missing[:8]}{'...' if len(missing)>8 else ''}"
    return True, ""


#def must_contain_phrases_gate(claim_text: str, cited_texts: List[str], phrases: List[str]) -> Tuple[bool, str]:
#    c = normalize_ws(claim_text)
#    combined = normalize_ws(" ".join(cited_texts))#

#    needed = [p for p in phrases if p in c]
#    missing = [p for p in needed if p not in combined]
#    if missing:
#        return False, f"UNSUPPORTED_KEY_PHRASE:missing={missing}"
#    return True, ""

def breadth_gate(claim_text: str) -> tuple[bool, str]:
    t = normalize_ws(claim_text)
    # crude but effective: too many conjunctions makes auditing harder
    count = t.count(" and ") + t.count(" including ") + t.count(" as well as ")
    if count >= 2:
        return False, "CLAIM_TOO_BROAD"
    return True, ""

def run_hard_gates(claim: Dict[str, Any], retrieved_map: Dict[str, str]) -> List[str]:
    issues: List[str] = []

    citations = claim.get("citations") or []
    if not citations:
        issues.append("MISSING_CITATIONS")
        return issues

    cited_texts = [retrieved_map.get(c, "") for c in citations if c in retrieved_map]
    if not any(t.strip() for t in cited_texts):
        issues.append("CITATIONS_NOT_IN_CONTEXT")
        return issues

    # Gate 0: any numbers in claim must exist in cited text
    ok, msg = must_contain_any_numbers_gate(claim.get("text", ""), cited_texts)
    if not ok:
        issues.append(msg)

    # Gate 1: numeric-heavy claims (3+ numbers) must be fully supported
    ok, msg = must_contain_numbers_gate(claim.get("text", ""), cited_texts)
    if not ok:
        issues.append(msg)

    # Gate 2: policy anchor phrases
    #phrases = [
    #    "pro-rated", "prorated",
    #    "appendix a", "vacation schedule",
    #    "calendar year", "january 1", "december 31"
    #]
    phrases = [
        "prorated",
        "appendix a", "vacation schedule",
        "calendar year", "january 1", "december 31"
    ]
    ok, msg = must_contain_phrases_gate(claim.get("text", ""), cited_texts, phrases)
    if not ok:
        issues.append(msg)

    # Gate 3: validate claim BROAD
    ok, msg = breadth_gate(claim.get("text", ""))
    if not ok:
        issues.append(msg)

    return issues