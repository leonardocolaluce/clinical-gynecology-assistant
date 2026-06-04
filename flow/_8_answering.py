from __future__ import annotations

import textwrap
from dataclasses import dataclass
import re

from ._2_models import ExternalDoc, Paper
from ._3_openai_client import OpenAIClient
from ._10_prompts import (
    direct_system_prompt,
    pubmed_external_system_prompt,
    pubmed_system_prompt,
    revise_system_prompt,
)


@dataclass(frozen=True)
class Answer:
    text: str


_PMID_RE = re.compile(r"\[PMID:\s*(\d+)\]", re.IGNORECASE)


def answer_direct(oai: OpenAIClient, *, model: str, question: str, mode: str) -> Answer:
    sys = direct_system_prompt(mode=mode)
    text = oai.chat(
        model=model,
        temperature=0.4,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": question},
        ],
    )
    return Answer(text=(text or "").strip())


def answer_with_pubmed(
    oai: OpenAIClient,
    *,
    model: str,
    question: str,
    mode: str,
    disclaimer: str,
    papers: list[Paper],
) -> Answer:
    system = pubmed_system_prompt(mode=mode, disclaimer=disclaimer)

    ctx = _format_context(papers)
    user = f"Domanda utente:\n{question}\n\nFonti (abstract PubMed):\n{ctx}"

    text = oai.chat(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return Answer(text=(text or "").strip())


def _format_context(papers: list[Paper]) -> str:
    blocks: list[str] = []
    for p in papers:
        header = f"PMID {p.pmid} | {p.year or 'n.d.'} | {p.title}".strip()
        meta = []
        if p.journal:
            meta.append(p.journal)
        if p.doi:
            meta.append(f"DOI: {p.doi}")
        meta_line = " | ".join(meta)
        abs_text = p.abstract.strip() if p.abstract else "<no abstract>"
        block = header
        if meta_line:
            block += "\n" + meta_line
        block += "\n" + abs_text
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)


def answer_with_pubmed_and_external(
    oai: OpenAIClient,
    *,
    model: str,
    question: str,
    mode: str,
    disclaimer: str,
    pubmed_papers: list[Paper],
    external_docs: list[ExternalDoc],
) -> Answer:
    system = pubmed_external_system_prompt(mode=mode, disclaimer=disclaimer)

    pubmed_ctx = _format_context(pubmed_papers or [])
    ext_ctx = _format_external_context(external_docs or [])
    user = (
        f"Domanda utente:\n{question}\n\n"
        f"Fonti PubMed (abstract):\n{pubmed_ctx or '<nessuna fonte PubMed>'}\n\n"
        f"Fonti Dataset (documenti):\n{ext_ctx or '<nessuna fonte Dataset>'}"
    )

    text = oai.chat(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return Answer(text=(text or "").strip())


def _format_external_context(docs: list[ExternalDoc]) -> str:
    blocks: list[str] = []
    for d in docs:
        header = f"DOC {d.doc_id} | {d.title}".strip()
        link = (d.url or "").strip()
        body = (d.text or "").strip() or "<no text>"
        block = header
        if link:
            block += "\n" + f"LINK: {link}"
        block += "\n" + body
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)


def extract_cited_pmids(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in _PMID_RE.finditer(text or ""):
        pmid = m.group(1)
        if pmid and pmid not in seen:
            seen.add(pmid)
            out.append(pmid)
    return out


def revise_to_meet_min_citations(
    oai: OpenAIClient,
    *,
    model: str,
    question: str,
    mode: str,
    disclaimer: str,
    papers: list[Paper],
    draft_answer: str,
    min_distinct_pmids: int,
) -> Answer:
    """
    Best-effort second pass to increase citation coverage *without inventing*.
    If sources can't support min distinct PMIDs, the revision should say so explicitly.
    """
    min_n = max(0, int(min_distinct_pmids))
    if min_n <= 1:
        return Answer(text=(draft_answer or "").strip())

    cited = extract_cited_pmids(draft_answer or "")
    if len(set(cited)) >= min_n:
        return Answer(text=(draft_answer or "").strip())
    if len(papers or []) < min_n:
        return Answer(text=(draft_answer or "").strip())

    allowed_pmids = [p.pmid for p in (papers or []) if p and p.pmid]
    allowed_pmids_str = ", ".join(allowed_pmids[:200])
    system = revise_system_prompt(mode=mode, disclaimer=disclaimer, min_n=min_n, allowed_pmids_str=allowed_pmids_str)

    ctx = _format_context(papers)
    user = (
        "Domanda utente:\n"
        f"{question}\n\n"
        "Bozza da revisionare:\n"
        f"{(draft_answer or '').strip()}\n\n"
        "Fonti (abstract PubMed):\n"
        f"{ctx}"
    )

    text = oai.chat(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return Answer(text=(text or "").strip())
