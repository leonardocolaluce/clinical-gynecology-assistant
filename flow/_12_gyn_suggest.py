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


def suggest_top3(
    *,
    city: str | None = None,
    address_hint: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    xlsx_path: str | None = None,
) -> list[GynSuggestion]:
    """
    Suggests up to 3 gynecologists using city, address text and/or latitude-longitude.
    Works if the user provides at least one of: city, address_hint, latitude+longitude.
    """
    path = Path(xlsx_path or Path(__file__).resolve().parents[1] / "ginecologhe_donna_filtrate.xlsx")
    if not path.is_file():
        return []

    rows = _load_rows(path)
    if not rows:
        return []

    city_norm = _norm(city)
    address_norm = _norm(address_hint)
    has_coords = latitude is not None and longitude is not None

    if not city_norm and not address_norm and not has_coords:
        return []

    scored = []
    for item in rows:
        score = 0.0
        distance_km = None

        item_address_norm = _norm(item["address"])

        if city_norm and city_norm in item_address_norm:
            score += 100

        if address_norm:
            score += _token_overlap(address_norm, item_address_norm) * 80

        if has_coords and item["latitude"] is not None and item["longitude"] is not None:
            distance_km = _haversine_km(latitude, longitude, item["latitude"], item["longitude"])
            score += max(0, 150 - distance_km)

        rating = item["rating"] or 0
        reviews = item["reviews"] or 0
        score += rating * 5
        score += min(reviews, 200) / 20

        if score > 0:
            scored.append((score, distance_km, item))

    scored.sort(
        key=lambda x: (
            x[1] is None,
            x[1] if x[1] is not None else 999999,
            -x[0],
        )
    )

    return [
        GynSuggestion(
            name=item["name"],
            address=item["address"],
            phone=item["phone"],
            website=item["website"],
            emails=item["emails"],
            rating=item["rating"],
            reviews=item["reviews"],
            distance_km=round(distance_km, 1) if distance_km is not None else None,
        )
        for _, distance_km, item in scored[:3]
    ]


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

