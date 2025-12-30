# src/control/assumption_detector.py
from typing import List, Dict, Any

def detect_assumptions(question: str, claims: List[Dict[str, Any]], context_lines: List[str]) -> List[Dict[str, Any]]:
    """
    Deterministically add assumptions that are logically required but often omitted.
    """
    ctx = "\n".join(context_lines).lower()
    assumptions: List[Dict[str, Any]] = []

    # A1_SCOPE: if policy mentions exclusions/applicability and answer gives entitlement rules,
    # assume subject is eligible.
    scope_triggers = ["applies to", "this policy applies", "exception", "with the exception", "does not apply", "excluded"]
    entitlement_triggers = ["entitlement", "vacation", "days per", "pro-rated", "prorated", "carry-over"]

    if any(t in ctx for t in scope_triggers) and any(
        any(et in (c.get("text","").lower()) for et in entitlement_triggers) for c in claims
    ):
        assumptions.append({
            "type": "A1_SCOPE",
            "impact": "low",
            "text": "The employee is eligible under this policy (i.e., not in an excluded category listed in the policy scope/applicability section)."
        })

    return assumptions