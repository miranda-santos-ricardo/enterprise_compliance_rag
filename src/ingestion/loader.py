# src/ingestion/loader.py
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

from pypdf import PdfReader


@dataclass
class LoadedDoc:
    policy_id: str
    title: str
    text: str
    metadata: Dict[str, Any]

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s

def _load_pdf(path: Path) -> Tuple[str, Dict[str, Any]]:
    reader = PdfReader(str(path))
    pages = [p.extract_text() or "" for p in reader.pages]
    text = "\n\n".join(pages)

    meta = reader.metadata or {}
    title = getattr(meta, "title", None) or path.stem

    text = clean_pdf_text(text)

    md = {
        "source_path": os.path.abspath(str(path)),
        "file_name": path.name,
        "file_type": "pdf",
        "page_count": len(reader.pages),
        "pdf_title": getattr(meta, "title", None),
        "pdf_author": getattr(meta, "author", None),
        "pdf_subject": getattr(meta, "subject", None),
    }
    return text, {"title": title, **md}


def _load_txt(path: Path) -> Tuple[str, Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    md = {
        "source_path": os.path.abspath(str(path)),
        "file_name": path.name,
        "file_type": "txt",
    }
    return text, {"title": path.stem, **md}


def load_policies(data_dir: str) -> List[LoadedDoc]:
    """
    Loads all policy files from data_dir (pdf/txt/md).
    policy_id is derived from filename stem.
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    docs: List[LoadedDoc] = []
    for path in sorted(root.glob("*")):
        if path.is_dir():
            continue
        ext = path.suffix.lower()
        if ext == ".pdf":
            text, md = _load_pdf(path)
        elif ext in {".txt", ".md"}:
            text, md = _load_txt(path)
        else:
            continue  # ignore unknown formats

        policy_id = slugify(path.stem)
        title = md.get("title", path.stem)
        docs.append(LoadedDoc(policy_id=policy_id, title=title, text=text, metadata=md))

    if not docs:
        raise ValueError(f"No supported policy files found in: {data_dir}")

    return docs



def clean_pdf_text(text: str) -> str:
    if not text:
        return ""
    # normalize whitespace
    t = re.sub(r"\r\n", "\n", text)
    t = re.sub(r"[ \t]+", " ", t)

    # remove common “Page header/footer” patterns (customize later)
    patterns = [
        r"Vacation Time Policy \| Department of Human Resources\s*\d+",
        r"Department of Human Resources\s*\d+",
    ]
    for p in patterns:
        t = re.sub(p, "", t)

    # remove excessive blank lines
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()
