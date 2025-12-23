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

Be strict. If a claim is not clearly supported, mark it as unsupported.

Return ONLY valid JSON:
{
  "issues": ["..."],
  "risk_level": "low"|"medium"|"high"|"critical",
  "confidence": 0.0-1.0,
  "is_compliant": true/false
}

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
      api_key = os.environ.get("OPENAI_API_KEY","")
      if not api_key:
        raise RuntimeError("OPENAI API KEY is missing.")
      model = os.environ.get("MODEL_ID","gpt-4.1-mini")
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
        model = self.model,
        messages = [
          {"role":"system", "content": "You verify claims against policy excerpts and enforce strict grounding."},
          {"role":"user", "content": prompt}
        ],
        response_format = {"type": "json_object"}
      )

      raw = resp.choices[0].message.content or {}
      data = json.loads(raw)

      issues = data.get("issues", [])
      riskLevel = data.get("risk_level","medium")
      confidence = float(data.get("confidence", 0.0))
      is_compliant = bool(data.get("is_compliant", False))

      #normalize
      if not isinstance(issues, list):
        issues = []

      risk_level = riskLevel if riskLevel in {"low", "medium", "high", "critical"} else "medium"

      return PolicyAssessment(
        issues=issues,
        risk_level=RiskLevel(risk_level),
        confidence=max(0.0, min(1.0, confidence)),
        is_compliant = is_compliant
      )
