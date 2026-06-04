from __future__ import annotations

import math
from dataclasses import dataclass

from ._2_models import Paper
from ._3_openai_client import OpenAIClient


@dataclass(frozen=True)
class RetrievalResult:
    papers: list[Paper]
    scores: list[float]


def select_top_k(oai: OpenAIClient, *, embed_model: str, question: str, papers: list[Paper], top_k: int) -> RetrievalResult:
    if not papers:
        return RetrievalResult(papers=[], scores=[])
    k = max(1, min(int(top_k), len(papers)))

    q_vec = oai.embed(model=embed_model, text=question)
    scored: list[tuple[Paper, float]] = []
    for p in papers:
        vec = oai.embed(model=embed_model, text=(p.title or "") + "\n" + (p.abstract or ""))
        scored.append((p, _cosine(q_vec, vec)))
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:k]
    return RetrievalResult(papers=[p for p, _ in top], scores=[float(s) for _, s in top])


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return float(dot / (math.sqrt(na) * math.sqrt(nb)))
