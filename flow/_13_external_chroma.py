from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ._2_models import ExternalDoc
from ._3_openai_client import OpenAIClient


@dataclass(frozen=True)
class ChromaExternal:
    persist_dir: str
    collection_name: str


def connect_chroma(persist_dir: str, *, collection_name: str = "default") -> ChromaExternal:
    p = Path(_normalize_windows_drive_path(persist_dir))
    if not p.exists():
        raise FileNotFoundError(
            f"Chroma persist dir not found: {persist_dir}. "
            "If you are running on WSL/Linux and this is a Windows drive path (e.g. G:\\...), "
            "make sure the drive is mounted under /mnt/<drive>/."
        )
    return ChromaExternal(persist_dir=str(p), collection_name=str(collection_name or "default"))

def _embed_local_384(text: str) -> list[float]:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("sentence-transformers is required for 384-dim Chroma retrieval.") from e

    if not hasattr(_embed_local_384, "_model"):
        _embed_local_384._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    vec = _embed_local_384._model.encode(text or "", normalize_embeddings=True)
    return [float(x) for x in vec]

def retrieve_top_n_chroma(
    chroma: ChromaExternal,
    *,
    oai: OpenAIClient,
    embed_model: str,
    question: str,
    top_n: int,
) -> list[ExternalDoc]:
    """
    Retrieves top_n chunks from a Chroma persistent DB.

    NOTE: requires the `chromadb` package at runtime. If missing, raises RuntimeError.
    """
    try:
        import chromadb  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("chromadb is not installed. Install it to enable External RAG via Chroma.") from e

    n = max(1, int(top_n))
    q_vec = _embed_local_384(question)

    client = chromadb.PersistentClient(path=chroma.persist_dir)
    try:
        col = client.get_collection(name=chroma.collection_name)
    except Exception as e:
        try:
            cols = client.list_collections()
            names = [c.name for c in (cols or []) if getattr(c, "name", None)]
        except Exception:
            names = []
        if len(names) == 1:
            col = client.get_collection(name=names[0])
        else:
            avail = ", ".join(names[:50]) if names else "<none>"
            raise RuntimeError(
                f"Chroma collection [{chroma.collection_name}] does not exist. "
                f"Available collections: {avail}. "
                "Set EXTERNAL_CHROMA_COLLECTION to a valid name."
            ) from e

    res = col.query(query_embeddings=[q_vec], n_results=n, include=["documents", "metadatas"])
    ids = (res.get("ids") or [[]])[0] or []
    docs = (res.get("documents") or [[]])[0] or []
    metas = (res.get("metadatas") or [[]])[0] or []

    out: list[ExternalDoc] = []
    for doc_id, text, meta in zip(ids, docs, metas):
        text_value = str(text or "")
        title, url = _meta_title_url(meta)
        if not title:
            title = _extract_title_from_text(text_value)
        
        out.append(
            ExternalDoc(
                doc_id=str(doc_id),
                title=title,
                text=text_value,
                url=url,
            )
        )
    return out


def _meta_title_url(meta: object) -> tuple[str, Optional[str]]:
    if not isinstance(meta, dict):
        return ("", None)
    title = str(meta.get("title") or "").strip()
    url = meta.get("url")
    if url is not None:
        url = str(url).strip() or None
    return (title, url)

def _extract_title_from_text(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    marker = "TITLE:"
    abstract_marker = "ABSTRACT:"
    upper = s.upper()
    if marker in upper:
        start = upper.find(marker) + len(marker)
        end = upper.find(abstract_marker, start)
        if end == -1:
            end = min(len(s), start + 180)
        return s[start:end].strip(" .:\n\t")
    return ""


def _normalize_windows_drive_path(p: str) -> str:
    s = (p or "").strip()
    if not s or os.name == "nt":
        return s
    # Minimal parsing to avoid extra imports/regex for this hot path.
    if len(s) >= 3 and s[1] == ":" and (s[2] == "\\" or s[2] == "/") and s[0].isalpha():
        drive = s[0].lower()
        rest = s[3:].replace("\\", "/")
        cand = f"/mnt/{drive}/{rest}"
        if Path(cand).exists():
            return cand
    return s
