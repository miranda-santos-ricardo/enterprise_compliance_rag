import os
from typing import List
from models.types import PolicyAssessment, ComplianceDecision, DecisionStatus, RiskLevel

def evaluate(assessment: PolicyAssessment) -> ComplianceDecision:
    reasons: List[str] = []
    confidence_level = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))
    #hard blocks
    if not assessment.is_compliant:
        reasons.append("Policy verification marked answer as non-compliant.")
    if assessment.risk_level in {RiskLevel.CRITICAL}:
        reasons.append("Risk Level is CRITICAL")
    if assessment.confidence < confidence_level:
        reasons.append("Confidence {assessment.confidence:.2f} is below threshold({confidence_level}).")

    #make a decision
    if reasons:
        return ComplianceDecision(
            status=DecisionStatus.BLOCK,
            reasons=reasons,
            assessment=assessment
        ) 

    #Review bucket
    if assessment.risk_level in {RiskLevel.HIGH} or assessment.confidence < 0.8:
        return ComplianceDecision(
            status=DecisionStatus.REVIEW,
            reasons=["Risk level is High. Need human review."] + assessment.issues,
            assessment=assessment
        )   

    #Return SAFE in the last scenario
    return ComplianceDecision(
        status = DecisionStatus.SAFE,
        reasons = ["No issues detected. Answer is compliant."],
        assessment=assessment
    )