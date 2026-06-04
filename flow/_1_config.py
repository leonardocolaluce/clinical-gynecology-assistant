from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_DEFAULT_CHROMA_DRIVE_PATH = r"G:\Il mio Drive\_rag_vector_db_MISSING_ONLY"


@dataclass(frozen=True)
class Settings:
    # NCBI / PubMed
    ncbi_api_key: str
    ncbi_tool: str
    ncbi_email: str
    pubmed_retmax: int = 50
    pubmed_timeout_s: int = 30

    # Retrieval
    top_k: int = 50
    min_distinct_citations: int = 2
    external_rag_db_path: str = ""
    external_chroma_collection: str = "default"
    external_candidates: int = 50
    final_pubmed_k: int = 50
    final_external_k: int = 50

    # OpenAI (required for this CLI: "only GPT answers")
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_chat_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"

    disclaimer: str = (
        "Nota: risposta a scopo informativo/educativo; non sostituisce il parere medico. "
        "Per diagnosi o terapia rivolgiti a un/una ginecologo/a."
    )


_KV_RE = re.compile(
    r"""(?ix) ^
    \s*
    (?P<k>[A-Za-z_][A-Za-z0-9_]*)
    \s*
    (?:=|:)
    \s*
    ["']?
    (?P<v>.*?)
    ["']?
    \s*
    $
"""
)


def _load_from_config_files() -> dict[str, str]:
    """
    Minimal config loader (secrets only).
    Looks for simple KEY=VALUE / key = "value" pairs in a few local paths.
    """
    base = Path(__file__).resolve().parents[1]  # .../pipeline_m1
    candidates = [
        os.getenv("PIPELINE_M1_CONFIG_PATH") or "",
        str(base / "config"),
        str(base / "config.txt"),
        str(base.parent / "config"),  # .../1_Milestone/config
    ]
    out: dict[str, str] = {}
    for p in candidates:
        path = Path(p) if p else None
        if not path or not path.is_file():
            continue
        try:
            for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = (raw or "").strip()
                if not line or line.startswith("#"):
                    continue
                m = _KV_RE.match(line)
                if not m:
                    continue
                k = (m.group("k") or "").strip()
                v = (m.group("v") or "").strip()
                if k and v:
                    out[k] = v
        except Exception:
            continue
    return out


def _get_conf(conf: dict[str, str], *keys: str) -> Optional[str]:
    for k in keys:
        v = conf.get(k)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return None


def load_settings() -> Settings:
    conf = _load_from_config_files()
    ncbi_api_key = (os.getenv("NCBI_API_KEY") or "").strip() or (_get_conf(conf, "NCBI_API_KEY", "ncbi_api_key") or "")
    openai_api_key = (os.getenv("OPENAI_API_KEY") or "").strip() or (
        _get_conf(conf, "OPENAI_API_KEY", "openai_api_key", "api_key", "openai_key") or ""
    )
    external_rag_db_path = (os.getenv("EXTERNAL_RAG_DB_PATH") or "").strip() or (
        _get_conf(conf, "EXTERNAL_RAG_DB_PATH") or ""
    )
    external_rag_db_path = _normalize_windows_drive_path(external_rag_db_path)
    if not external_rag_db_path:
        p = Path(_normalize_windows_drive_path(_DEFAULT_CHROMA_DRIVE_PATH))
        if p.exists() and p.is_dir():
            external_rag_db_path = str(p)
    external_chroma_collection = (os.getenv("EXTERNAL_CHROMA_COLLECTION") or "").strip() or (
        _get_conf(conf, "EXTERNAL_CHROMA_COLLECTION") or "default"
    )

    return Settings(
        ncbi_api_key=ncbi_api_key,
        ncbi_tool=(os.getenv("NCBI_TOOL") or "chatbot_gin_cli").strip() or "chatbot_gin_cli",
        ncbi_email=(os.getenv("NCBI_EMAIL") or "dev@example.com").strip() or "dev@example.com",
        pubmed_retmax=int((os.getenv("PUBMED_RETMAX") or "50").strip() or "50"),
        pubmed_timeout_s=int((os.getenv("PUBMED_TIMEOUT_S") or "30").strip() or "30"),
        top_k=int((os.getenv("TOP_K") or "50").strip() or "50"),
        min_distinct_citations=int((os.getenv("MIN_DISTINCT_CITATIONS") or "2").strip() or "2"),
        external_rag_db_path=external_rag_db_path,
        external_chroma_collection=external_chroma_collection,
        external_candidates=int((os.getenv("EXTERNAL_CANDIDATES") or "50").strip() or "50"),
        final_pubmed_k=int((os.getenv("FINAL_PUBMED_K") or "50").strip() or "50"),
        final_external_k=int((os.getenv("FINAL_EXTERNAL_K") or "50").strip() or "50"),
        openai_api_key=openai_api_key,
        openai_base_url=(os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip()
        or "https://api.openai.com/v1",
        openai_chat_model=(os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini").strip() or "gpt-4o-mini",
        openai_embed_model=(os.getenv("OPENAI_EMBED_MODEL") or "text-embedding-3-small").strip()
        or "text-embedding-3-small",
    )


_WIN_DRIVE_RE = re.compile(r"(?i)^(?P<drive>[a-z]):[\\/](?P<rest>.*)$")


def _normalize_windows_drive_path(p: str) -> str:
    """
    If running on non-Windows (e.g. WSL/Linux) and the user provides a Windows drive path like:
      G:\\Il mio Drive\\folder
    try to map it to a WSL mount path:
      /mnt/g/Il mio Drive/folder

    If the mapped path doesn't exist, keep the original string unchanged.
    """
    s = (p or "").strip()
    if not s or os.name == "nt":
        return s
    m = _WIN_DRIVE_RE.match(s)
    if not m:
        return s
    drive = (m.group("drive") or "").lower()
    rest = (m.group("rest") or "").replace("\\", "/")
    cand = f"/mnt/{drive}/{rest}"
    if Path(cand).exists():
        return cand
    return s
