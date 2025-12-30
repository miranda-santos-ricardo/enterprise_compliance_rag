from dataclasses import dataclass
from typing import Any, Dict, List
from enum import Enum



class AssumptionType(str,Enum):
    A1_SCOPE = "A1_SCOPE"
    A2_INTERPRETATION = "A2_INTERPRETATION"
    A3_MISSING_CONTEXT = "A3_MISSING_CONTEXT"

class Impact(str,Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class Assumption:
    type: AssumptionType
    text: str
    impact: Impact

#######################################################################    

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DecisionStatus(str, Enum):
    SAFE = "safe_to_use"
    REVIEW = "review_required"
    BLOCK = "do_not_use"

@dataclass
class Chunk:
    id: str
    policy_id: str
    section_id: str
    text: str
    metadata: Dict [str, Any]

@dataclass
class RetrievedContext:
    chunks: List[Chunk]

@dataclass
class Claim:
    text: str
    citations: List[str]

@dataclass
class AnswerProposal:
    claims: List[Claim]
    assumptions: List[Assumption]
    final_answer: str

@dataclass
class PolicyAssessment:
    issues: List[str]
    risk_level: RiskLevel
    confidence: float
    is_compliant: bool

@dataclass
class ComplianceDecision:
    status: DecisionStatus
    reasons: List[str]
    assessment: PolicyAssessment

