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
_DOC_RE = re.compile(r"\[DOC:\s*([^\]]+)\]", re.IGNORECASE)

def _format_history(history: list[dict[str, str]] | None) -> str:
    if not history:
        return "<nessuna conversazione precedente>"
    blocks = []
    for item in history[-10:]:
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        blocks.append(f"Utente: {q}\nAssistente: {a}")
    return "\n\n".join(blocks)

def answer_direct(oai: OpenAIClient, *, model: str, question: str, mode: str, history: list[dict[str, str]] | None = None) -> Answer:
    sys = direct_system_prompt(mode=mode)
    text = oai.chat(
        model=model,
        temperature=0.4,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": f"Conversazione precedente:\n{_format_history(history)}\n\nDomanda attuale:\n{question}"},
        ],
    )
    return Answer(text=(text or "").strip())


def answer_with_pubmed(
    oai: OpenAIClient,
    *,
    model: str,
    question: str,
    mode: str,
    history: list[dict[str, str]] | None = None,
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
    history: list[dict[str, str]] | None = None,
    disclaimer: str,
    pubmed_papers: list[Paper],
    external_docs: list[ExternalDoc],
) -> Answer:
    system = pubmed_external_system_prompt(mode=mode, disclaimer=disclaimer)

    pubmed_ctx = _format_context(pubmed_papers or [])
    ext_ctx = _format_external_context(external_docs or [])
    user = (
        f"Conversazione precedente:\n{_format_history(history)}\n\n"
        f"Domanda attuale:\n{question}\n\n"
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

def extract_cited_doc_ids(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in _DOC_RE.finditer(text or ""):
        doc_id = (m.group(1) or "").strip()
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
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


def answer_clarification(
    oai: OpenAIClient,
    *,
    model: str,
    question: str,
    mode: str,
    reason: str,
    history: list[dict[str, str]] | None = None,
) -> Answer:
    if reason == "no_sources":
        task = (
            "La pipeline non ha trovato fonti scientifiche sufficienti o citazioni affidabili. "
            "Non rispondere nel merito clinico. Spiega in modo breve che servono più dettagli "
            "per cercare meglio nelle fonti, e fai 3-4 controdomande utili."
        )
    else:
        task = (
            "La domanda è troppo generica per formulare una ricerca scientifica affidabile. "
            "Non rispondere nel merito clinico. Fai 3-4 controdomande utili per chiarire il sintomo."
        )

    text = oai.chat(
        model=model,
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": (
                    "Sei Chatbot Gin, un assistente informativo in ambito ginecologico. "
                    "Rispondi in italiano, con tono umano, naturale e sintetico. "
                    "Non fare diagnosi o terapia personalizzata. "
                    "Non chiedere età, nome, indirizzo, residenza o dati personali. "
                    "Puoi chiedere durata, localizzazione del sintomo, andamento, sintomi associati, "
                    "relazione con ciclo mestruale, menopausa, gravidanza, rapporti sessuali o terapie in corso. "
                    "Quando inviti a rivolgersi a una figura specialistica usa sempre il femminile: ginecologa, specialista, professionista. "
                    "Non usare mai ginecologo, dottore o medico per riferirti alla professionista a cui rivolgersi."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Modalità utente: {mode}\n"
                    f"Conversazione precedente:\n{_format_history(history)}\n\n"
                    f"Domanda attuale:\n{question}\n\n"
                    f"Istruzione:\n{task}"
                ),
            },
        ],
    )
    return Answer(text=(text or "").strip())


def answer_with_gyn_area_offer(
    oai: OpenAIClient,
    *,
    model: str,
    current_answer: str,
    mode: str,
) -> Answer:
    text = oai.chat(
        model=model,
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": (
                    "Sei Chatbot Gin. Devi riscrivere la risposta in italiano mantenendola sintetica e naturale. "
                    "Non aggiungere diagnosi o terapia. "
                    "Aggiungi in modo fluido una proposta: se l'utente indica un'area di interesse, ad esempio città o zona, "
                    "puoi provare a suggerire alcune ginecologhe in quell'area. "
                    "Non chiedere indirizzo, residenza, posizione geografica o dove vive. "
                    "Usa sempre il femminile: ginecologa, ginecologhe, specialista, professionista. "
                    "Non usare mai ginecologo, dottore o medico per riferirti alla professionista."
                ),
            },
            {
                "role": "user",
                "content": f"Modalità utente: {mode}\n\nRisposta da rendere più naturale:\n{current_answer}",
            },
        ],
    )
    return Answer(text=(text or "").strip())


def answer_gyn_suggestions_result(
    oai: OpenAIClient,
    *,
    model: str,
    area: str,
    count: int,
    mode: str,
) -> Answer:
    text = oai.chat(
        model=model,
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": (
                    "Sei Chatbot Gin. Rispondi in italiano con tono naturale, sintetico e umano. "
                    "Non fare diagnosi o terapia. "
                    "Devi introdurre il risultato della ricerca di ginecologhe per l'area indicata dall'utente. "
                    "Non chiedere indirizzo, residenza, posizione geografica o dove vive. "
                    "Usa sempre il femminile: ginecologa, ginecologhe, specialista, professionista. "
                    "Non usare mai ginecologo, dottore o medico per riferirti alla professionista."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Modalità utente: {mode}\n"
                    f"Area indicata: {area}\n"
                    f"Numero di risultati trovati: {count}\n\n"
                    "Scrivi una breve frase introduttiva. Se non ci sono risultati, dillo in modo gentile e suggerisci di provare con una zona più ampia."
                ),
            },
        ],
    )
    return Answer(text=(text or "").strip())
