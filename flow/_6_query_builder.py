from __future__ import annotations

import datetime as dt
import re


_STOP = {
    "il",
    "lo",
    "la",
    "i",
    "gli",
    "le",
    "un",
    "una",
    "uno",
    "di",
    "a",
    "da",
    "in",
    "su",
    "con",
    "per",
    "tra",
    "fra",
    "e",
    "o",
    "che",
    "del",
    "della",
    "dello",
    "dei",
    "delle",
    "nel",
    "nella",
    "sul",
    "sulla",
    "come",
    "cosa",
    "quali",
    "qual",
    "quanto",
    "quando",
    "perche",
    "perché",
}


_WORD = re.compile(r"[A-Za-z0-9]+", re.UNICODE)
_HRT_HINT = re.compile(r"(?i)\b(tos|hrt|hormone|ormon|estrogen|estradiol|progestin|progesterone|menopaus)\b")


def _build_hrt_pubmed_term(question: str, *, last_years: int) -> str:
    year_from = dt.date.today().year - max(0, int(last_years))

    therapy = (
        "("
        "menopausal hormone therapy[Title/Abstract] OR "
        "hormone replacement therapy[Title/Abstract] OR "
        "MHT[Title/Abstract] OR HRT[Title/Abstract] OR "
        "estrogen therapy[Title/Abstract] OR estradiol[Title/Abstract] OR "
        "progestin[Title/Abstract] OR progesterone[Title/Abstract]"
        ")"
    )
    menopause = "(menopause[Title/Abstract] OR menopausal[Title/Abstract] OR Menopause[MeSH Terms])"

    q = (question or "").lower()
    risk_terms: list[str] = []
    if any(k in q for k in ("tromb", "vte", "embol", "venous", "stroke", "ictus")):
        risk_terms.append(
            "(thrombosis[Title/Abstract] OR thromboembolism[Title/Abstract] OR venous thromboembolism[Title/Abstract] OR VTE[Title/Abstract] OR stroke[Title/Abstract])"
        )
    if any(k in q for k in ("oncolog", "canc", "tumor", "carcin", "breast", "mamm", "endometr")):
        risk_terms.append(
            "(cancer[Title/Abstract] OR neoplasm[Title/Abstract] OR breast cancer[Title/Abstract] OR endometrial cancer[Title/Abstract])"
        )

    parts = [therapy, menopause, f"{year_from}:3000[dp]"]
    parts.extend(risk_terms)
    return " AND ".join(parts)


def build_pubmed_term(question: str, *, last_years: int, require_review: bool) -> str:
    q = (question or "").strip()
    if not q:
        return "gynecology[Title/Abstract]"

    tokens = [m.group(0).lower() for m in _WORD.finditer(q)]
    keywords = [t for t in tokens if t not in _STOP and len(t) >= 3][:8]
    if not keywords:
        keywords = [t for t in tokens if len(t) >= 2][:6]

    kw_query = " AND ".join(f"{_esc(k)}[Title/Abstract]" for k in keywords if _esc(k))
    if not kw_query:
        kw_query = "gynecology[Title/Abstract]"

    year_from = dt.date.today().year - max(0, int(last_years))
    parts = [kw_query, f"{year_from}:3000[dp]"]
    if require_review:
        parts.append("review[Publication Type]")

    # Gentle domain anchor.
    return f"({' AND '.join(parts)}) AND (gynecology[Title/Abstract] OR gynaecology[Title/Abstract] OR obstetrics[Title/Abstract])"


def _esc(t: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", " ", t).strip()


def build_pubmed_term_candidates(question: str) -> list[str]:
    """
    Returns a list of progressively more relaxed PubMed queries.
    Goal: reach up to 50 results whenever possible, even for non-medical inputs.
    """
    q = (question or "").strip()
    if not q:
        return ["gynecology[Title/Abstract] OR obstetrics[Title/Abstract]"]

    # Special-case: menopausal hormone therapy / TOS / HRT questions.
    # These often underperform with the generic "gynecology anchor", so we try a dedicated query first.
    if _HRT_HINT.search(q):
        hrt_strict = _build_hrt_pubmed_term(q, last_years=10)
        hrt_broad = _build_hrt_pubmed_term(q, last_years=30)
    else:
        hrt_strict = ""
        hrt_broad = ""

    # 1) Strict-ish (recent, anchored to domain).
    strict = build_pubmed_term(q, last_years=10, require_review=False)

    # 2) Broader time window.
    broad_time = build_pubmed_term(q, last_years=30, require_review=False)

    # 3) Drop year filter entirely but keep keywords.
    tokens = [m.group(0).lower() for m in _WORD.finditer(q)]
    keywords = [t for t in tokens if t not in _STOP and len(t) >= 3][:8]
    if not keywords:
        keywords = [t for t in tokens if len(t) >= 2][:6]

    if keywords:
        kw_and = " AND ".join(f"{_esc(k)}[Title/Abstract]" for k in keywords if _esc(k))
        kw_or = " OR ".join(f"{_esc(k)}[Title/Abstract]" for k in keywords if _esc(k))
    else:
        kw_and = ""
        kw_or = ""

    domain = "(gynecology[Title/Abstract] OR gynaecology[Title/Abstract] OR obstetrics[Title/Abstract])"
    no_year_and = f"({kw_and}) AND {domain}" if kw_and else domain
    no_year_or = f"({kw_or}) AND {domain}" if kw_or else domain

    # 4) Last-resort: pure domain query (guarantees lots of results).
    fallback = "gynecology[Title/Abstract] OR gynaecology[Title/Abstract] OR obstetrics[Title/Abstract]"

    # Keep order, remove empties/duplicates.
    out: list[str] = []
    for term in (hrt_strict, hrt_broad, strict, broad_time, no_year_and, no_year_or, fallback):
        t = (term or "").strip()
        if t and t not in out:
            out.append(t)
    return out
