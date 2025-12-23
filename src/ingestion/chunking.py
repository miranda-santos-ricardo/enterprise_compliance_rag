# src/ingestion/chunking.py
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class TextChunk:
    policy_id: str
    section_id: str
    text: str
    metadata: Dict[str, Any]

def looks_like_table(s:str) -> bool:
    digits = sum(ch.isdigit() for ch in s)
    return digits / max(len(s),1) > 0.18

def chunk_text(
    policy_id: str,
    text: str,
    base_metadata: Dict[str, Any],
    chunk_size: int = 1400,
    overlap: int = 80,
) -> List[TextChunk]:
    """
    Naive character-based chunking. Good enough for MVP.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    chunks: List[TextChunk] = []
    start = 0
    idx = 0
        
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)

        chunk = cleaned[start:end].strip()
        section_id = f"sec{idx:04d}"
        md = dict(base_metadata)
        md.update({"policy_id": policy_id, "section_id": section_id, "chunk_index": idx})

        #drop chunks that looks like table 
        if looks_like_table(chunk):
            continue

        chunks.append(TextChunk(policy_id=policy_id, section_id=section_id, text=chunk, metadata=md))

        idx += 1
        
        if  (end - start) < chunk_size:
            break
        start = end - overlap  # overlap for continuity
        start = max(start,0)
        
    return chunks