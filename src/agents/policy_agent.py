import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any

from openai import OpenAI
from models.types import PolicyAssessment, RiskLevel

VERIFY_PROMPT = """You are the Policy Verification Agent for enterprise policy Q&A.

You receive:
- A user question
- Retrieved policy excerpts with IDs
- A proposed answer made of individual claims, each with citations

Your job:
1) Verify each claim is supported by the cited excerpts.
2) Flag unsupported or over-specific claims.
3) Flag weak/irrelevant citations (citation spam).
4) Identify compliance risk level based on ambiguity or missing evidence.

Return ONLY valid JSON:
{
  "claim_checks":[
    {"claim_index": 0, "supported": true, "issues":[]},
    {"claim_index": 1, "supported": false, "issues":["UNSUPPORTED_CLAIM","WEAK_CITATION"]}
  ],
  "issues": ["..."],
  "risk_level": "low"|"medium"|"high"|"critical",
  "confidence": 0.0-1.0,
  "is_compliant": true/false
}

Rules:
- If ANY claim is unsupported -> is_compliant MUST be false.
- If there are ANY issues -> confidence MUST be < 0.85.
- Confidence MUST NOT be 1.0 unless all claims are explicitly supported with strong citations.

Issue codes you MUST use when applicable:
- UNSUPPORTED_CLAIM
- WEAK_CITATION
- MISSING_CITATIONS
- OVER_SPECIFIC
- INSUFFICIENT_CONTEXT

Question:
{{question}}

Retrieved excerpts:
{{context}}

Claims (with citations):
{{claims}}
"""


@dataclass
class PolicyAgent:
    client: OpenAI
    model: str

    @classmethod
    def from_env(cls) -> "PolicyAgent":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI API KEY is missing.")
        model = os.environ.get("MODEL_ID", "gpt-4.1-mini")
        return cls(client=OpenAI(api_key=api_key), model=model)

    def run(self, question: str, context_lines: List[str], claims: List[Dict[str, Any]]) -> PolicyAssessment:
        context = "\n".join(context_lines)
        claims_json = json.dumps(claims, ensure_ascii=False)

        prompt = (
            VERIFY_PROMPT
            .replace("{{question}}", question)
            .replace("{{context}}", context)
            .replace("{{claims}}", claims_json)
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You verify claims against policy excerpts and enforce strict grounding."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)

        issues = data.get("issues", [])
        risk_level = data.get("risk_level", "medium")
        confidence = float(data.get("confidence", 0.0))
        is_compliant = bool(data.get("is_compliant", False))

        if not isinstance(issues, list):
            issues = []

        # Enforce claim_checks
        claim_checks = data.get("claim_checks", [])
        if isinstance(claim_checks, list) and claim_checks:
            for cc in claim_checks:
                supported = cc.get("supported", True)
                cc_issues = cc.get("issues", []) or []
                if supported is False:
                    is_compliant = False
                    if cc_issues:
                        issues.extend([str(x) for x in cc_issues])
                    else:
                        issues.append("UNSUPPORTED_CLAIM")

        # Normalize risk level
        risk_level = risk_level if risk_level in {"low", "medium", "high", "critical"} else "medium"

        # Confidence sanity
        if issues and confidence >= 0.85:
            confidence = 0.84
        if not is_compliant and confidence > 0.8:
            confidence = 0.8
        confidence = max(0.0, min(1.0, confidence))

        return PolicyAssessment(
            issues=[str(i) for i in issues],
            risk_level=RiskLevel(risk_level),
            confidence=confidence,
            is_compliant=is_compliant,
        )