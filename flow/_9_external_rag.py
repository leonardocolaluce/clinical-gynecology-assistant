from __future__ import annotations

import json
import math
import os
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from ._2_models import ExternalDoc


@dataclass(frozen=True)
class ExternalHit:
    doc: ExternalDoc
    score: float


def connect_external(db_path: str) -> sqlite3.Connection:
    path = Path(_normalize_windows_drive_path(db_path))
    if not path.is_file():
        raise FileNotFoundError(
            f"External RAG DB not found: {db_path}. "
            "If you are running on WSL/Linux and this is a Windows drive path (e.g. G:\\...), "
            "make sure the drive is mounted under /mnt/<drive>/."
        )
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def retrieve_top_n(
    conn: sqlite3.Connection,
    *,
    query_vec: list[float],
    top_n: int,
    table: str = "docs",
    id_col: str = "id",
    title_col: str = "title",
    text_col: str = "text",
    url_col: str = "url",
    embedding_col: str = "embedding",
) -> list[ExternalHit]:
    """
    Minimal external retriever for a pre-built embeddings DB.

    Expected schema by default:
      docs(id TEXT PRIMARY KEY, title TEXT, text TEXT, url TEXT, embedding BLOB|TEXT)

    Supported embedding formats:
      - BLOB of float32 little-endian (length = 4 * dim)
      - TEXT containing JSON list[float]
    """
    n = max(1, int(top_n))
    q = _as_floats(query_vec)
    qn = _norm(q)
    if qn <= 0:
        return []

    # Scan & keep best N (40k docs is acceptable in Python; relies on embeddings being readily decodable).
    best: list[ExternalHit] = []
    worst_score = -1.0

    sql = f"SELECT {id_col} AS id, {title_col} AS title, {text_col} AS text, {url_col} AS url, {embedding_col} AS emb FROM {table}"
    cur = conn.execute(sql)
    for row in cur:
        emb_raw = row["emb"]
        vec = _decode_embedding(emb_raw)
        if not vec:
            continue
        score = _cosine_pre_normed(q, qn, vec)
        if len(best) < n:
            best.append(
                ExternalHit(
                    doc=ExternalDoc(
                        doc_id=str(row["id"]),
                        title=str(row["title"] or ""),
                        text=str(row["text"] or ""),
                        url=str(row["url"] or "") or None,
                    ),
                    score=float(score),
                )
            )
            if len(best) == n:
                best.sort(key=lambda h: h.score, reverse=True)
                worst_score = best[-1].score
            continue

        if score <= worst_score:
            continue
        # Insert in sorted list (n is small: 50).
        hit = ExternalHit(
            doc=ExternalDoc(
                doc_id=str(row["id"]),
                title=str(row["title"] or ""),
                text=str(row["text"] or ""),
                url=str(row["url"] or "") or None,
            ),
            score=float(score),
        )
        _insert_sorted(best, hit)
        best[:] = best[:n]
        worst_score = best[-1].score

    best.sort(key=lambda h: h.score, reverse=True)
    return best


def _insert_sorted(arr: list[ExternalHit], item: ExternalHit) -> None:
    lo = 0
    hi = len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if item.score > arr[mid].score:
            hi = mid
        else:
            lo = mid + 1
    arr.insert(lo, item)


def _decode_embedding(v: object) -> Optional[list[float]]:
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray, memoryview)):
        b = bytes(v)
        if len(b) < 8 or (len(b) % 4) != 0:
            return None
        try:
            out = [x[0] for x in struct.iter_unpack("<f", b)]
        except Exception:
            return None
        return out
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            data = json.loads(s)
        except Exception:
            return None
        if not isinstance(data, list):
            return None
        try:
            return [float(x) for x in data]
        except Exception:
            return None
    return None


def _as_floats(v: Iterable[float]) -> list[float]:
    return [float(x) for x in (v or [])]


def _norm(v: list[float]) -> float:
    s = 0.0
    for x in v:
        s += x * x
    return math.sqrt(s)


def _cosine_pre_normed(q: list[float], qn: float, d: list[float]) -> float:
    if not d or len(d) != len(q):
        return -1.0
    dot = 0.0
    dn2 = 0.0
    for a, b in zip(q, d):
        dot += a * b
        dn2 += b * b
    if dn2 <= 0:
        return -1.0
    return float(dot / (qn * math.sqrt(dn2)))


def _normalize_windows_drive_path(p: str) -> str:
    s = (p or "").strip()
    if not s or os.name == "nt":
        return s
    if len(s) >= 3 and s[1] == ":" and (s[2] == "\\" or s[2] == "/") and s[0].isalpha():
        drive = s[0].lower()
        rest = s[3:].replace("\\", "/")
        cand = f"/mnt/{drive}/{rest}"
        if Path(cand).exists():
            return cand
    return s
