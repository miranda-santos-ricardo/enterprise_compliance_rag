import os
import json
from dataclasses import dataclass
from typing import List

from openai import OpenAI

from models.types import AnswerProposal

ANSWER_PROMPT = """You are the Answer Agent for enterprise policy Q&A.

You will receive:
- A user question
- A set of policy excerpts, each with an ID like "vacation-time-policy:sec0003"

Rules (non-negotiable):
1) Use ONLY the provided excerpts. Do not use outside knowledge.
2) Every factual statement must be supported by at least one cited excerpt ID.
3) If the excerpts are insufficient to answer safely, say so explicitly.
4) Be concise. Avoid unnecessary text.
5) Each claim must express one policy rule only.
6) Avoid ‘and/including’ unless it’s a single tight clause.
7) If you mention another rule it must be a separate claim with its own citation.
8) final answer has to follow the structure: statement[section-id]. Example: Initial entitlement: typically 15 days. [sec002].
9) If a claim relies on a Schedule, Appendix, Table, or Matrix, the claim text MUST explicitly name it (e.g., ‘Appendix A (Vacation Schedule)’). Do not rely on citations alone.

Return ONLY valid JSON in this exact schema:
{
  "claims": [
    {"text":"...", "citations": ["policy-id:section-id", "..."]},
    {"text":"...", "citations": ["policy-id:section-id", "..."]},
  ],
  "assumptions": ["...", "..."]
  "final_answer": "string"
}

Question:
{{question}}

Policy excerpts:
{{context}}
"""

@dataclass
class AnswerAgent:
  client: OpenAI
  model: str

  @classmethod
  def from_env(cls) -> "AnswerAgent":
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
      raise RuntimeError("OPENAI_API_KEY is missing.")
    model = os.environ.get("MODEL_ID","gpt-4.1-mini")
    return cls(client=OpenAI(api_key=api_key),model=model)

  def run(self, question: str, context_lines: List[str]) -> AnswerProposal:
    context = "\n".join(context_lines)
    prompt = ANSWER_PROMPT.replace("{{question}}", question).replace("{{context}}", context)

    resp = self.client.chat.completions.create(
      model = self.model,
      messages = [
        {"role":"system","content":"You answer policy questions using ONLY provided excerpts."},
        {"role":"user","content":prompt}
      ],
      response_format = {"type":"json_object"},
    )

    raw = resp.choices[0].message.content or {}
    data = json.loads(raw)

    print(data)
    
    #defensive parsing
    claims = data.get("claims", [])
    assumptions = data.get("assumptions", [])
    final_answer = str(data.get("final_answer","")).strip()
    
    if not isinstance(claims, list):
      claims=[]
   
    if not isinstance(assumptions, list):
      assumptions = []
    
    return AnswerProposal(
      claims=claims,
      assumptions=assumptions,
      final_answer = final_answer
    )