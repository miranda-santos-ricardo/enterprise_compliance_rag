from typing import List, Dict, Any, Tuple

def classify_assumptions(assumptions: List[Dict[str,Any]]) -> Tuple[str,float,list[str]]:
    """
    Returns: (status_override, confidence_cap, issues)
    status_override: "BLOCK"|"REVIEW"|""  ("" means no override)
    confidence_cap: 1.0 means no cap
    """
    #print("AQUi")
    issues = []
    if not assumptions:
        return "", 1.0, issues
    
    types = {a.get("type") for a in assumptions if isinstance(a, dict)}

    if "A3_MISSING_CONTEXT" in types:
        issues.append("ASSUMPTION_A3_MISSING_CONTEXT")
        return "BLOCK", 0.7, issues

    if "A2_INTERPRETATION" in types:
        issues.append("ASSUMPTION_A2_INTERPRETATION")
        return "REVIEW", 0.8, issues
    
    if "A1_SCOPE" in types:
        issues.append("ASSUMPTION_A1_SCOPE")
        return "", 0.85, issues

    issues.append("ASSUMPTION_UNKNOWN_TYPE")
    return "REVIEW", 0.8, issues