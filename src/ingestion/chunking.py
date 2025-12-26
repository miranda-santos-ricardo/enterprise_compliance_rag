# src/ingestion/chunking.py
import re
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class TextChunk:
    policy_id: str
    section_id: str
    text: str
    metadata: Dict[str, Any]


HEADING_REGEXES = [
    r"^\s*[A-Z][A-Za-z &/\-]{2,60}\s*$",                  # Title-case heading line
    r"^\s*[A-Z][A-Z &/\-]{3,60}\s*$",                     # ALL CAPS heading
    r"^\s*(Appendix|Schedule)\s+[A-Z0-9]+.*$",            # Appendix A / Schedule 1
    r"^\s*\d+(\.\d+)*\s+[A-Z].*$",                        # 1. Heading / 2.1 Heading
]


def is_heading(line: str) -> bool:
    s = line.strip()
    if len(s) < 4 or len(s) > 80:
        return False
    return any(re.match(rx, s) for rx in HEADING_REGEXES)


def split_into_sections(text: str) -> List[str]:
    """
    Generic structural split: starts a new section when a heading-like line appears.
    Keeps content grouped so Appendix/table blocks don't merge with previous section.
    """
    lines = (text or "").splitlines()
    sections: List[str] = []
    buf: List[str] = []

    for line in lines:
        if is_heading(line) and buf:
            section_text = "\n".join(buf).strip()
            if section_text:
                sections.append(section_text)
            buf = [line]
        else:
            buf.append(line)

    last = "\n".join(buf).strip()
    if last:
        sections.append(last)

    return sections


def naive_chunk(section_text: str, chunk_size: int, overlap: int) -> List[str]:
    cleaned = (section_text or "").strip()
    if not cleaned:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    out: List[str] = []
    start = 0
    n = len(cleaned)

    while start < n:
        end = min(n, start + chunk_size)
        out.append(cleaned[start:end].strip())

        if end == n:
            break  # IMPORTANT: stop at end, don't loop forever

        # advance start; ensure progress
        start = end - overlap
        if start < 0:
            start = 0

    return out


def chunk_text(
    policy_id: str,
    text: str,
    base_metadata: Dict[str, Any],
    chunk_size: int = 1400,
    overlap: int = 80,
) -> List[TextChunk]:
    """
    Section-aware chunking (generic). Prevents Frankenstein chunks.
    """
    if not text:
        return []

    sections = split_into_sections(text)
    chunks: List[TextChunk] = []

    idx = 0
    for s in sections:
        subchunks = naive_chunk(s, chunk_size=chunk_size, overlap=overlap)
        for sc in subchunks:
            section_id = f"sec{idx:04d}"
            md = dict(base_metadata)
            md.update({"policy_id": policy_id, "section_id": section_id, "chunk_index": idx})
            chunks.append(TextChunk(policy_id=policy_id, section_id=section_id, text=sc, metadata=md))
            idx += 1

    return chunks