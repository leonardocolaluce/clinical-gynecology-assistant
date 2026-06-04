from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class GynSuggestion:
    name: str
    address: str
    phone: Optional[str] = None
    website: Optional[str] = None
    emails: Optional[str] = None
    rating: Optional[float] = None
    reviews: Optional[int] = None


def suggest_top3(*, city: str, address_hint: str | None = None, db_path: str | None = None) -> list[GynSuggestion]:
    """
    Suggests up to 3 gynecologists.
    Current heuristic (no lat/long): match by city, optional address substring, rank by rating then reviews.
    """
    city_norm = _norm_city(city)
    if not city_norm:
        return []

    path = Path(
        db_path
        or os.getenv("GINECOLOGHE_DB_PATH")
        or (Path(__file__).resolve().parents[1] / "data" / "ginecologhe.sqlite3")
    )
    if not path.is_file():
        return []

    conn = sqlite3.connect(str(path))
    try:
        conn.row_factory = sqlite3.Row
        params: list[object] = [city_norm]
        where = "city_norm = ?"
        if address_hint and address_hint.strip():
            where += " AND lower(address) LIKE ?"
            params.append(f"%{address_hint.strip().lower()}%")

        rows = conn.execute(
            f"""
            SELECT business_name, address, phone, website, emails, rating, reviews
            FROM gynecologists
            WHERE {where}
            ORDER BY
              CASE WHEN rating IS NULL THEN 1 ELSE 0 END,
              rating DESC,
              CASE WHEN reviews IS NULL THEN 1 ELSE 0 END,
              reviews DESC
            LIMIT 3
            """,
            params,
        ).fetchall()
        return [
            GynSuggestion(
                name=str(r["business_name"] or "").strip(),
                address=str(r["address"] or "").strip(),
                phone=_opt_str(r["phone"]),
                website=_opt_str(r["website"]),
                emails=_opt_str(r["emails"]),
                rating=float(r["rating"]) if r["rating"] is not None else None,
                reviews=int(r["reviews"]) if r["reviews"] is not None else None,
            )
            for r in rows
            if str(r["business_name"] or "").strip() and str(r["address"] or "").strip()
        ]
    finally:
        conn.close()


def _opt_str(v: object) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _norm_city(city: str | None) -> str:
    c = (city or "").strip().lower()
    out: list[str] = []
    prev_space = False
    for ch in c:
        if ch.isalnum():
            out.append(ch)
            prev_space = False
        else:
            if not prev_space:
                out.append(" ")
                prev_space = True
    return " ".join("".join(out).split())

