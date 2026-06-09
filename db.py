from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .flow._2_models import Paper


DEFAULT_DB_PATH = os.getenv("PIPELINE_M1_DB_PATH") or "1_Milestone/pipeline_m1/data/pipeline_m1.sqlite3"


@dataclass(frozen=True)
class RetrievalRun:
    id: int
    created_at: int
    query: str
    found_count: int


def connect(db_path: str | None = None) -> sqlite3.Connection:
    path = Path(db_path or DEFAULT_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS papers (
          pmid TEXT PRIMARY KEY,
          title TEXT NOT NULL,
          abstract TEXT NOT NULL,
          year TEXT,
          journal TEXT,
          doi TEXT,
          fetched_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS retrieval_runs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_at INTEGER NOT NULL,
          query TEXT NOT NULL,
          found_count INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS retrieval_run_pmids (
          run_id INTEGER NOT NULL,
          pmid TEXT NOT NULL,
          ord INTEGER NOT NULL,
          PRIMARY KEY (run_id, ord),
          FOREIGN KEY (run_id) REFERENCES retrieval_runs(id)
        );

        CREATE TABLE IF NOT EXISTS messages (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_at INTEGER NOT NULL,
          mode TEXT NOT NULL,
          question TEXT NOT NULL,
          session_id TEXT,
          answer TEXT,
          error TEXT,
          retrieval_run_id INTEGER,
          cited_pmids_json TEXT,
          FOREIGN KEY (retrieval_run_id) REFERENCES retrieval_runs(id)
        );
        """
    )
    conn.commit()


def create_retrieval_run(conn: sqlite3.Connection, *, query: str, found_count: int, pmids: list[str]) -> RetrievalRun:
    now = int(time.time())
    cur = conn.execute(
        "INSERT INTO retrieval_runs(created_at, query, found_count) VALUES(?,?,?)",
        (now, query, int(found_count)),
    )
    run_id = int(cur.lastrowid)
    conn.executemany(
        "INSERT INTO retrieval_run_pmids(run_id, pmid, ord) VALUES(?,?,?)",
        [(run_id, pmid, idx) for idx, pmid in enumerate(pmids)],
    )
    conn.commit()
    return RetrievalRun(id=run_id, created_at=now, query=query, found_count=int(found_count))


def create_message(conn: sqlite3.Connection, *, mode: str, question: str, session_id: str | None = None) -> int:
    now = int(time.time())
    cur = conn.execute(
        "INSERT INTO messages(created_at, mode, question, session_id) VALUES(?,?,?,?)",
        (now, mode, question, session_id),
    )
    conn.commit()
    return int(cur.lastrowid)


def finalize_message_ok(
    conn: sqlite3.Connection,
    *,
    message_id: int,
    answer: str,
    retrieval_run_id: int,
    cited_pmids: list[str],
) -> None:
    conn.execute(
        "UPDATE messages SET answer=?, error=NULL, retrieval_run_id=?, cited_pmids_json=? WHERE id=?",
        (answer, int(retrieval_run_id), json.dumps(cited_pmids), int(message_id)),
    )
    conn.commit()


def finalize_message_error(conn: sqlite3.Connection, *, message_id: int, error: str) -> None:
    conn.execute(
        "UPDATE messages SET error=? WHERE id=?",
        (error, int(message_id)),
    )
    conn.commit()

def get_recent_messages(conn: sqlite3.Connection, *, session_id: str, limit: int = 10) -> list[dict[str, str]]:
    rows = conn.execute(
        """
        SELECT question, answer
        FROM messages
        WHERE session_id=? AND answer IS NOT NULL
        ORDER BY id DESC
        LIMIT ?
        """,
        (session_id, int(limit)),
    ).fetchall()

    out: list[dict[str, str]] = []
    for row in reversed(rows):
        out.append({"question": str(row["question"] or ""), "answer": str(row["answer"] or "")})
    return out

def get_cached_papers(conn: sqlite3.Connection, pmids: Iterable[str]) -> dict[str, Paper]:
    pmid_list = [p for p in (pmids or []) if p]
    if not pmid_list:
        return {}

    placeholders = ",".join("?" for _ in pmid_list)
    rows = conn.execute(
        f"SELECT pmid,title,abstract,year,journal,doi FROM papers WHERE pmid IN ({placeholders})",
        pmid_list,
    ).fetchall()
    out: dict[str, Paper] = {}
    for r in rows:
        out[str(r["pmid"])] = Paper(
            pmid=str(r["pmid"]),
            title=str(r["title"] or ""),
            abstract=str(r["abstract"] or ""),
            year=str(r["year"]) if r["year"] is not None else None,
            journal=str(r["journal"]) if r["journal"] is not None else None,
            doi=str(r["doi"]) if r["doi"] is not None else None,
        )
    return out


def upsert_papers(conn: sqlite3.Connection, papers: list[Paper]) -> None:
    now = int(time.time())
    rows = [
        (
            p.pmid,
            p.title or "",
            p.abstract or "",
            p.year,
            p.journal,
            p.doi,
            now,
        )
        for p in (papers or [])
        if p and p.pmid
    ]
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO papers(pmid,title,abstract,year,journal,doi,fetched_at)
        VALUES(?,?,?,?,?,?,?)
        ON CONFLICT(pmid) DO UPDATE SET
          title=excluded.title,
          abstract=excluded.abstract,
          year=excluded.year,
          journal=excluded.journal,
          doi=excluded.doi,
          fetched_at=excluded.fetched_at
        """,
        rows,
    )
    conn.commit()


def get_papers_in_order(conn: sqlite3.Connection, pmids: list[str]) -> list[Paper]:
    cached = get_cached_papers(conn, pmids)
    out: list[Paper] = []
    for pmid in pmids:
        p = cached.get(pmid)
        if p is not None:
            out.append(p)
    return out


def get_paper(conn: sqlite3.Connection, pmid: str) -> Optional[Paper]:
    row = conn.execute(
        "SELECT pmid,title,abstract,year,journal,doi FROM papers WHERE pmid=?",
        (pmid,),
    ).fetchone()
    if not row:
        return None
    return Paper(
        pmid=str(row["pmid"]),
        title=str(row["title"] or ""),
        abstract=str(row["abstract"] or ""),
        year=str(row["year"]) if row["year"] is not None else None,
        journal=str(row["journal"]) if row["journal"] is not None else None,
        doi=str(row["doi"]) if row["doi"] is not None else None,
    )

