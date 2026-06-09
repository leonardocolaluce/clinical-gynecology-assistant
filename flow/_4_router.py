from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from ._3_openai_client import OpenAIClient


@dataclass(frozen=True)
class RouteDecision:
    route: str  # "direct" | "pubmed"
    term: Optional[str] = None
    last_years: int = 10
    require_review: bool = False


_SYSTEM = """
Sei un router per un chatbot ginecologico.
Decidi se serve consultare PubMed prima di rispondere.

Regole:
- Se la domanda e' generale/conversazionale (saluti, "chi sei?", cose non mediche, data/ora), route="direct".
- Se la domanda e' medica/ginecologica e beneficia di evidenze/citazioni, route="pubmed".
- Se route="pubmed", scrivi "term" in inglese: traduci/normalizza la domanda utente in termini biomedicali inglesi adatti a PubMed".
- Se e' una domanda medica ma non richiede letteratura (es. chiarimenti sul funzionamento), route="direct".
- Se route="pubmed", il campo "term" deve essere una query PubMed in inglese, specifica per la domanda utente, anche se la domanda e' in italiano. Non usare query generiche come "gynecology OR obstetrics" se la domanda contiene un tema specifico.

Esempi:
- "cosa e' la menopausa" -> "menopause[Title/Abstract] OR menopausal[Title/Abstract] OR Menopause[MeSH Terms]"
- "cosa e' endometriosi" -> "endometriosis[Title/Abstract] OR Endometriosis[MeSH Terms]"
- "perdite vaginali cause" -> "vaginal discharge[Title/Abstract] AND causes[Title/Abstract]"

Rispondi SOLO con JSON valido, senza testo extra:
{
  "route": "direct" | "pubmed",
  "term": "pubmed query (solo se pubmed)",
  "last_years": 10,
  "require_review": false
}
""".strip()


def decide_route(client: OpenAIClient, *, model: str, question: str) -> RouteDecision:
    content = client.chat(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": question},
        ],
    )
    data = _safe_json(content)
    route = str((data.get("route") or "direct")).strip().lower()
    if route not in {"direct", "pubmed"}:
        route = "direct"

    term = data.get("term")
    if term is not None:
        term = str(term).strip() or None

    last_years = _clamp_int(data.get("last_years"), lo=0, hi=30, default=10)
    require_review = bool(data.get("require_review") or False)

    if route == "pubmed" and not term:
        term = None
    return RouteDecision(route=route, term=term, last_years=last_years, require_review=require_review)

def contextualize_question(
    client: OpenAIClient,
    *,
    model: str,
    question: str,
    history: list[dict[str, str]] | None,
) -> str:
    if not history:
        return question

    recent = []
    for item in history[-5:]:
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        recent.append(f"Utente: {q}\nAssistente: {a[:500]}")

    content = client.chat(
        model=model,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Riscrivi la domanda attuale rendendola autonoma usando il contesto. "
                    "Se la domanda attuale è già autonoma, restituiscila invariata. "
                    "Rispondi solo con la domanda riscritta, senza spiegazioni."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Conversazione precedente:\n"
                    + "\n\n".join(recent)
                    + f"\n\nDomanda attuale:\n{question}"
                ),
            },
        ],
    )
    return (content or question).strip() or question

def allow_direct_without_sources(question: str) -> bool:
    """
    Deterministic safety gate.
    Only allows the "direct" route for clearly non-medical / meta questions,
    so we don't accidentally skip PubMed on medical queries.
    """
    q = (question or "").strip().lower()
    if not q:
        return True

    # Greetings / small talk
    for tok in ("ciao", "buongiorno", "buonasera", "hey", "salve", "grazie", "thanks"):
        if tok in q:
            return True

    # Meta / how it works / identity
    for tok in (
        "chi sei",
        "cosa sei",
        "come funzioni",
        "come funziona",
        "come posso usare",
        "istruzioni",
        "help",
        "aiuto",
        "privacy",
        "dati personali",
        "account",
        "login",
        "registr",
        "password",
    ):
        if tok in q:
            return True

    # Time/date questions (non-medical)
    for tok in ("che ore", "che giorno", "che data", "oggi che", "data di oggi", "ora corrente"):
        if tok in q:
            return True

    return False


def _safe_json(s: str) -> dict[str, Any]:
    try:
        return json.loads((s or "").strip())
    except Exception:
        return {}


def _clamp_int(v: Any, *, lo: int, hi: int, default: int) -> int:
    try:
        x = int(v)
    except Exception:
        return default
    return max(lo, min(hi, x))
