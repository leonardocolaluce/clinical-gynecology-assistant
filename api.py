from __future__ import annotations

import random

from .flow._10_prompts import load_prompt_styles, save_prompt_styles
from .flow._12_gyn_suggest import suggest_top3
from typing import Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import db
from .flow._1_config import load_settings
from .flow._3_openai_client import OpenAIClient
from .flow._5_pubmed_client import PubMedClient
from .flow._6_query_builder import build_pubmed_term_candidates
from .flow._7_retrieval import select_top_k
from .flow._8_answering import (
    answer_clarification,
    answer_direct,
    answer_gyn_suggestions_result,
    answer_with_gyn_area_offer,
    answer_with_pubmed,
    answer_with_pubmed_and_external,
    extract_cited_pmids,
    extract_cited_doc_ids,
    revise_to_meet_min_citations,
)
from .flow._4_router import contextualize_question, decide_route
from .flow._9_external_rag import connect_external, retrieve_top_n
from .flow._13_external_chroma import connect_chroma, retrieve_top_n_chroma
from .flow._14_debug import dbg


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    mode: str = Field(default="patient", description="patient|doctor|menopause")
    session_id: Optional[str] = None
    area_of_interest: Optional[str] = None
    city: Optional[str] = None
    address_hint: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class GynSuggestionOut(BaseModel):
    name: str
    address: str
    phone: Optional[str] = None
    website: Optional[str] = None
    emails: Optional[str] = None
    rating: Optional[float] = None
    reviews: Optional[int] = None
    distance_km: Optional[float] = None

class Citation(BaseModel):
    source: str = "pubmed"
    pmid: Optional[str] = None
    doc_id: Optional[str] = None
    url: Optional[str] = None
    title: str
    year: Optional[str] = None
    journal: Optional[str] = None
    doi: Optional[str] = None


class RetrievalInfo(BaseModel):
    query: str
    found: int
    pmids: list[str]
    cached: int
    fetched: int


class ChatResponse(BaseModel):
    answer: str
    retrieval: RetrievalInfo
    citations: list[Citation]
    suggestions: list[GynSuggestionOut] = []

class StatusMessageResponse(BaseModel):
    message: str
    initial_delay_seconds: int = 5
    next_delay_seconds: int

class PromptConfig(BaseModel):
    patient: str
    menopause: str
    doctor: str

app = FastAPI(title="Pipeline M1 - Chatbot Gin", version="0.1")

STATUS_MESSAGES = [
    "Sto analizzando la domanda.",
    "Sto ricostruendo il contesto della conversazione.",
    "Sto preparando una ricerca più precisa.",
    "Sto consultando le fonti scientifiche disponibili.",
    "Sto interrogando PubMed.",
    "Sto verificando gli abstract più pertinenti.",
    "Sto confrontando le informazioni recuperate.",
    "Sto controllando che la risposta sia supportata dalle fonti.",
    "Sto selezionando i riferimenti più utili.",
    "Sto verificando le citazioni scientifiche.",
    "Sto consultando anche il database Europe PMC.",
    "Sto cercando documenti pertinenti nel database scientifico.",
    "Sto filtrando le informazioni meno rilevanti.",
    "Sto organizzando la risposta in modo chiaro.",
    "Sto controllando che non manchino passaggi importanti.",
    "Sto preparando una risposta sintetica e utile.",
    "Sto evitando conclusioni non supportate dalle fonti.",
    "Sto verificando se servono ulteriori dettagli.",
    "Sto controllando se la domanda richiede una risposta più specifica.",
    "Sto cercando di formulare una risposta prudente.",
    "Sto confrontando PubMed e il database scientifico interno.",
    "Sto controllando la coerenza tra fonti e risposta.",
    "Sto raccogliendo gli elementi più affidabili.",
    "Sto preparando una risposta con citazioni quando disponibili.",
    "Ancora qualche istante, sto verificando le fonti.",
    "Ancora qualche istante, sto completando il controllo.",
    "Sto finalizzando la risposta.",
    "Sto facendo l'ultima verifica sulle informazioni.",
    "Sto preparando il testo finale.",
    "Quasi pronto, sto ordinando le informazioni principali.",
]


@app.on_event("startup")
def _startup() -> None:
    conn = db.connect()
    db.init_db(conn)
    conn.close()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/chat/status-message", response_model=StatusMessageResponse)
def chat_status_message() -> StatusMessageResponse:
    return StatusMessageResponse(
        message=random.choice(STATUS_MESSAGES),
        next_delay_seconds=random.randint(8, 13),
    )


@app.get("/admin/prompts", response_model=PromptConfig)
def get_prompts() -> PromptConfig:
    return PromptConfig(**load_prompt_styles())


@app.put("/admin/prompts")
def save_prompts(payload: PromptConfig) -> dict[str, str]:
    save_prompt_styles(patient=payload.patient, menopause=payload.menopause, doctor=payload.doctor)
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    print(f"[CHAT] mode={req.mode!r} message={req.message[:80]!r}", flush=True)
    settings = load_settings()
    print(
        f"[SETTINGS] external_path={settings.external_rag_db_path!r} "
        f"collection={settings.external_chroma_collection!r} "
        f"external_candidates={settings.external_candidates} final_external_k={settings.final_external_k}",
        flush=True,
    )
    if not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")

    conn = db.connect()
    db.init_db(conn)

    session_id = (req.session_id or "").strip() or "default"
    history = db.get_recent_messages(conn, session_id=session_id, limit=20)
    print(f"[MEMORY] session_id={session_id!r} history_items={len(history)}", flush=True)
    for idx, item in enumerate(history, start=1):
        print(
            f"[MEMORY] {idx} Q={item.get('question', '')[:80]!r} A={item.get('answer', '')[:80]!r}",
            flush=True,
        )
    message_id = db.create_message(conn, mode=req.mode, question=req.message, session_id=session_id)
    oai = OpenAIClient(api_key=settings.openai_api_key, base_url=settings.openai_base_url)

    if not _is_doctor_mode(req.mode) and _asked_for_gyn_area(history):
        area = (req.area_of_interest or req.message or "").strip()
    
        if _vague_gyn_area(area):
            run = db.create_retrieval_run(conn, query="gyn_area_request", found_count=0, pmids=[])
            answer_text = "Mi serve una città o una zona precisa per cercare nel database. Quale area preferisci?"
            db.finalize_message_ok(conn, message_id=message_id, answer=answer_text, retrieval_run_id=run.id, cited_pmids=[])
            return ChatResponse(
                answer=answer_text,
                retrieval=RetrievalInfo(query="gyn_area_request", found=0, pmids=[], cached=0, fetched=0),
                citations=[],
                suggestions=[],
            )
        suggestions = build_gyn_suggestions(
            ChatRequest(message=req.message, mode=req.mode, session_id=session_id, city=area, address_hint=area)
        )
        run = db.create_retrieval_run(conn, query="gyn_suggestions", found_count=0, pmids=[])
        ans = answer_gyn_suggestions_result(
            oai,
            model=settings.openai_chat_model,
            area=area,
            count=len(suggestions),
            mode=req.mode,
        )
        answer_text = (ans.text or "").strip()
        db.finalize_message_ok(conn, message_id=message_id, answer=answer_text, retrieval_run_id=run.id, cited_pmids=[])
        return ChatResponse(
            answer=answer_text,
            retrieval=RetrievalInfo(query="gyn_suggestions", found=0, pmids=[], cached=0, fetched=0),
            citations=[],
            suggestions=suggestions,
        )

    explicit_area = None if _is_doctor_mode(req.mode) else _extract_area_from_gyn_request(req.message)
    if explicit_area and not _vague_gyn_area(explicit_area):
        suggestions = build_gyn_suggestions(
            ChatRequest(
                message=req.message,
                mode=req.mode,
                session_id=session_id,
                city=explicit_area,
                address_hint=explicit_area,
            )
        )
        run = db.create_retrieval_run(conn, query="gyn_suggestions", found_count=0, pmids=[])
        ans = answer_gyn_suggestions_result(
            oai,
            model=settings.openai_chat_model,
            area=explicit_area,
            count=len(suggestions),
            mode=req.mode,
        )
        answer_text = (ans.text or "").strip()
        db.finalize_message_ok(conn, message_id=message_id, answer=answer_text, retrieval_run_id=run.id, cited_pmids=[])
        return ChatResponse(
            answer=answer_text,
            retrieval=RetrievalInfo(query="gyn_suggestions", found=0, pmids=[], cached=0, fetched=0),
            citations=[],
            suggestions=suggestions,
        )

    if explicit_area and _vague_gyn_area(explicit_area):
        run = db.create_retrieval_run(conn, query="gyn_area_request", found_count=0, pmids=[])
        answer_text = "Certo. In quale città o zona vuoi cercare una ginecologa?"
        db.finalize_message_ok(conn, message_id=message_id, answer=answer_text, retrieval_run_id=run.id, cited_pmids=[])
        return ChatResponse(
            answer=answer_text,
            retrieval=RetrievalInfo(query="gyn_area_request", found=0, pmids=[], cached=0, fetched=0),
            citations=[],
            suggestions=[],
        )

    if not _is_doctor_mode(req.mode) and _explicit_gyn_request(req.message):
        run = db.create_retrieval_run(conn, query="gyn_area_request", found_count=0, pmids=[])
        answer_text = "Certo. In quale città o zona vuoi cercare una ginecologa?"
        db.finalize_message_ok(conn, message_id=message_id, answer=answer_text, retrieval_run_id=run.id, cited_pmids=[])
        return ChatResponse(
            answer=answer_text,
            retrieval=RetrievalInfo(query="gyn_area_request", found=0, pmids=[], cached=0, fetched=0),
            citations=[],
            suggestions=[],
        )

    if _needs_clarification(req.message):
        run = db.create_retrieval_run(conn, query="clarification", found_count=0, pmids=[])
        ans = answer_clarification(
            oai,
            model=settings.openai_chat_model,
            question=req.message,
            mode=req.mode,
            reason="generic_question",
            history=history,
        )
        answer_text = (ans.text or "").strip()
        db.finalize_message_ok(conn, message_id=message_id, answer=answer_text, retrieval_run_id=run.id, cited_pmids=[])
        return ChatResponse(
            answer=answer_text,
            retrieval=RetrievalInfo(query="clarification", found=0, pmids=[], cached=0, fetched=0),
            citations=[],
            suggestions=[],
        )

    try:
        dbg(f"/chat mode={req.mode!r} msg_len={len((req.message or '').strip())}")
        if settings.external_rag_db_path:
            p = Path(settings.external_rag_db_path)
            dbg(f"EXTERNAL_RAG_DB_PATH={settings.external_rag_db_path!r} exists={p.exists()} is_dir={p.is_dir()} is_file={p.is_file()}")
        pubmed = PubMedClient(
            api_key=settings.ncbi_api_key,
            tool=settings.ncbi_tool,
            email=settings.ncbi_email,
            timeout_s=settings.pubmed_timeout_s,
        )

        

        # Router (safe-by-default): only allow direct for clearly non-medical/meta queries.
        retrieval_question = contextualize_question(
            oai,
            model=settings.openai_chat_model,
            question=req.message,
            history=history,
        )
        print(f"[CONTEXT] original={req.message!r} retrieval_question={retrieval_question!r}", flush=True)
        
        # Router: GPT decides whether this needs fresh scientific retrieval or a direct conversational answer.
        try:
            decision = decide_route(oai, model=settings.openai_chat_model, question=retrieval_question)
        except Exception as e:
            print(f"[ROUTER] error={type(e).__name__}: {str(e)}", flush=True)
            decision = None
        use_direct = bool(decision and decision.route == "direct")
        print(
            f"[ROUTER] route={decision.route if decision else 'none'} "
            f"term={(decision.term if decision else None)!r} "
            f"use_direct={use_direct}",
            flush=True,
        )
        if use_direct:
            run = db.create_retrieval_run(conn, query="direct", found_count=0, pmids=[])
            ans = answer_direct(oai, model=settings.openai_chat_model, question=req.message, mode=req.mode, history=history)
            answer_text = (ans.text or "").strip()
            cited_pmids: list[str] = []
            citations: list[Citation] = []
            db.finalize_message_ok(conn, message_id=message_id, answer=answer_text, retrieval_run_id=run.id, cited_pmids=cited_pmids)
            suggestions = build_gyn_suggestions(req)

            return ChatResponse(
                answer=answer_text,
                retrieval=RetrievalInfo(query="direct", found=0, pmids=[], cached=0, fetched=0),
                citations=citations,
                suggestions=suggestions,
            )

        doctor_mode = _is_doctor_mode(req.mode)
        pubmed_retmax = max(int(settings.pubmed_retmax), 25) if doctor_mode else int(settings.pubmed_retmax)
        pubmed_top_k = max(int(settings.top_k), 20) if doctor_mode else int(settings.top_k)
        external_candidates = max(int(settings.external_candidates), 20) if doctor_mode else int(settings.external_candidates)
        final_external_k = max(int(settings.final_external_k), 20) if doctor_mode else int(settings.final_external_k)

        pmids: list[str] = []
        query_used: str = ""
        terms = []
        if decision and decision.route == "pubmed" and decision.term:
            terms = [decision.term]
        else:
            terms = list(build_pubmed_term_candidates(retrieval_question))

        for term in terms:
            query_used = term
            pmids = pubmed.esearch(term, retmax=pubmed_retmax)
            if len(pmids) >= pubmed_retmax:
                break

        pmids = pmids[:pubmed_retmax]
        cached = db.get_cached_papers(conn, pmids)
        missing = [p for p in pmids if p not in cached]
        fetched = 0
        if missing:
            fetched_papers = pubmed.efetch(missing)
            fetched = len(fetched_papers)
            db.upsert_papers(conn, fetched_papers)
            cached = db.get_cached_papers(conn, pmids)

        papers = [cached[p] for p in pmids if p in cached]
        print(
            f"[PUBMED] query={query_used!r} pmids={len(pmids)} papers={len(papers)} cached={len(pmids) - fetched} fetched={fetched}",
            flush=True,
        )
        run = db.create_retrieval_run(conn, query=query_used, found_count=len(pmids), pmids=pmids)

        # Optional reranking: keep only top-k most relevant abstracts before answering.
        if papers and pubmed_top_k > 0 and pubmed_top_k < len(papers):
            reranked = select_top_k(
                oai,
                embed_model=settings.openai_embed_model,
                question=retrieval_question,
                papers=papers,
                top_k=pubmed_top_k,
            )
            papers = reranked.papers

        external_docs = []
        print("[EUROPEPMC] external block reached", flush=True)
        if settings.external_rag_db_path and external_candidates > 0 and final_external_k > 0:
            if settings.external_rag_db_path.strip().lower().startswith("http"):
                raise RuntimeError("EXTERNAL_RAG_DB_PATH is a URL. Provide a local path (Drive-synced folder/file) instead.")
            p = Path(settings.external_rag_db_path)
            print(
                f"[EUROPEPMC] path_check exists={p.exists()} is_dir={p.is_dir()} is_file={p.is_file()} path={str(p)!r}",
                flush=True,
            )
            if p.is_dir():
                print(f"[EUROPEPMC] connecting Chroma collection={settings.external_chroma_collection!r}", flush=True)
                chroma_ext = connect_chroma(settings.external_rag_db_path, collection_name=settings.external_chroma_collection)
                try:
                    print("[EUROPEPMC] Chroma retrieval start", flush=True)
                    docs = retrieve_top_n_chroma(
                        chroma_ext,
                        oai=oai,
                        embed_model=settings.openai_embed_model,
                        question=retrieval_question,
                        top_n=external_candidates,
                    )
                    external_docs = docs[: max(0, final_external_k)]
                    print(f"[EUROPEPMC] Chroma retrieval done candidates={len(docs)} final={len(external_docs)}", flush=True)
                    dbg(f"External(Chroma) docs={len(docs)} final={len(external_docs)}")
                except Exception as e:
                    print(f"[EUROPEPMC] Chroma error {type(e).__name__}: {str(e)}", flush=True)
                    dbg(f"External(Chroma) retrieval error: {type(e).__name__}: {str(e).strip()}")
                    external_docs = []
            else:
                print("[EUROPEPMC] using SQLite external retriever", flush=True)
                ext_conn = connect_external(settings.external_rag_db_path)
                try:
                    q_vec = oai.embed(model=settings.openai_embed_model, text=req.message)
                    hits = retrieve_top_n(ext_conn, query_vec=q_vec, top_n=external_candidates)
                    external_docs = [h.doc for h in hits[: max(0, final_external_k)]]
                    dbg(f"External(SQLite) hits={len(hits)} final={len(external_docs)}")
                finally:
                    ext_conn.close()
                    
        print(
            f"[EUROPEPMC] enabled={bool(settings.external_rag_db_path)} path={settings.external_rag_db_path!r} docs={len(external_docs)}",
            flush=True,
        )

        if external_docs:
            ans = answer_with_pubmed_and_external(
                oai,
                model=settings.openai_chat_model,
                question=req.message,
                mode=req.mode,
                history=history,
                disclaimer=settings.disclaimer,
                pubmed_papers=papers,
                external_docs=external_docs,
            )
            answer_text = (ans.text or "").strip()
        else:
            ans = answer_with_pubmed(
                oai,
                model=settings.openai_chat_model,
                question=req.message,
                mode=req.mode,
                history=history,
                disclaimer=settings.disclaimer,
                papers=papers,
            )
            answer_text = (ans.text or "").strip()

        # Best-effort second pass: if citations are too few, ask the model to revise (without inventing).
        if not external_docs and int(settings.min_distinct_citations) > 1:
            revised = revise_to_meet_min_citations(
                oai,
                model=settings.openai_chat_model,
                question=req.message,
                mode=req.mode,
                disclaimer=settings.disclaimer,
                papers=papers,
                draft_answer=answer_text,
                min_distinct_pmids=int(settings.min_distinct_citations),
            )
            answer_text = (revised.text or "").strip()

        cited_pmids = extract_cited_pmids(answer_text)
        cited_doc_ids = extract_cited_doc_ids(answer_text)
        print(f"[CITATIONS] pmids={len(cited_pmids)} docs={len(cited_doc_ids)} doc_ids={cited_doc_ids[:5]}", flush=True)
        
        citations = _build_citations(conn, cited_pmids)
        citations.extend(_build_external_citations(external_docs, cited_doc_ids))

        if not citations and not papers and not external_docs:
            ans = answer_clarification(
                oai,
                model=settings.openai_chat_model,
                question=req.message,
                mode=req.mode,
                reason="no_sources",
                history=history,
            )
            answer_text = (ans.text or "").strip()
            cited_pmids = []
            cited_doc_ids = []

        db.finalize_message_ok(conn, message_id=message_id, answer=answer_text, retrieval_run_id=run.id, cited_pmids=cited_pmids)

        print(
            f"[RETRIEVAL] query={query_used!r} found={len(pmids)} cached={len(pmids) - fetched} fetched={fetched}",
            flush=True,
        )

        suggestions = build_gyn_suggestions(req)

        if not suggestions and _should_offer_gyn(req, history, answer_text, citations):
            ans = answer_with_gyn_area_offer(
                oai,
                model=settings.openai_chat_model,
                current_answer=answer_text,
                mode=req.mode,
            )
            answer_text = (ans.text or "").strip()

        return ChatResponse(
            answer=answer_text,
            retrieval=RetrievalInfo(
                query=query_used,
                found=len(pmids),
                pmids=pmids,
                cached=len(pmids) - fetched,
                fetched=fetched,
            ),
            citations=citations,
            suggestions=suggestions,
        )
    except Exception as e:
        db.finalize_message_error(conn, message_id=message_id, error=str(e))
        raise
    finally:
        conn.close()

def build_gyn_suggestions(req: ChatRequest) -> list[GynSuggestionOut]:
    has_location = bool(
        req.city
        or req.address_hint
        or req.latitude is not None
        or req.longitude is not None
    )

    if _is_doctor_mode(req.mode) or not has_location:
        return []

    raw_suggestions = suggest_top3(
        city=req.city,
        address_hint=req.address_hint,
        latitude=req.latitude,
        longitude=req.longitude,
    )
    return [
        GynSuggestionOut(
            name=s.name,
            address=s.address,
            phone=s.phone,
            website=s.website,
            emails=s.emails,
            rating=s.rating,
            reviews=s.reviews,
            distance_km=s.distance_km,
        )
        for s in raw_suggestions
    ]

_GENERIC_SYMPTOMS = {
    "mal di pancia",
    "male alla pancia",
    "dolore pancia",
    "dolore addome",
    "dolore basso ventre",
    "bruciore",
    "perdite",
    "prurito",
    "sanguinamento",
    "ritardo",
    "nausea",
}

def _needs_clarification(text: str) -> bool:
    q = (text or "").strip().lower()
    words = [w for w in q.replace("?", " ").split() if w]
    if len(words) <= 5 and any(symptom in q for symptom in _GENERIC_SYMPTOMS):
        return True
    return False

def _is_doctor_mode(mode: str) -> bool:
    return (mode or "").strip().lower() in {"doctor", "medico", "ginecologo", "ginecologa"}

def _asked_for_gyn_area(history: list[dict[str, str]]) -> bool:
    prompts = (
        "area di interesse",
        "città o zona",
        "zona",
        "area",
        "in quell'area",
        "in quella zona",
        "indicami",
        "mi indichi",
        "puoi indicarmi",
        "posso provare a suggerire alcune ginecologhe",
        "posso suggerirti alcune ginecologhe",
    )
    return any(
        any(prompt in (h.get("answer") or "").lower() for prompt in prompts)
        for h in history[-3:]
    )

def _already_offered_gyn(history: list[dict[str, str]]) -> bool:
    return any("posso provare a suggerirti" in (h.get("answer") or "").lower() for h in history)

def _explicit_gyn_request(text: str) -> bool:
    q = (text or "").lower()
    return any(
        x in q
        for x in [
            "consigli una ginecologa",
            "consigliami una ginecologa",
            "trova una ginecologa",
            "trovami una ginecologa",
            "trovami medici",
            "trova medici",
            "cerca medici",
            "ginecologa vicino",
            "specialista vicino",
            "da chi posso andare",
        ]
    )

def _vague_gyn_area(text: str) -> bool:
    q = (text or "").lower()
    return any(x in q for x in [
        "dove vivo",
        "vicino a me",
        "vicino casa",
        "mia zona",
        "nella mia zona",
        "nella tua zona",
        "nella tua area",
        "qui vicino",
    ])

def _extract_area_from_gyn_request(text: str) -> str | None:
    q = (text or "").strip().lower()
    if not q:
        return None
    request_terms = ("ginecolog", "medic", "dottor", "specialist", "trovami", "trova", "cerca", "consigli")
    if not any(term in q for term in request_terms):
        return None
    for marker in (" vicino a ", " vicino ", " in zona ", " zona ", " a ", " in ", " su "):
        if marker in q:
            area = q.split(marker, 1)[1].strip(" .?!,;:")
            return area or None
    return None

def _clinical_turns(history: list[dict[str, str]]) -> int:
    return sum(1 for h in history if (h.get("question") or "").strip())

def _should_offer_gyn(req: ChatRequest, history: list[dict[str, str]], answer_text: str, citations: list[Citation]) -> bool:
    if _is_doctor_mode(req.mode) or _already_offered_gyn(history):
        return False
    if _clinical_turns(history) < 2:
        return False

    if _explicit_gyn_request(req.message):
        return True
    if "rivolg" in answer_text.lower() and "ginecolog" in answer_text.lower():
        return True
    if _clinical_turns(history) >= 3:
        return True
    if not citations:
        return True
    return False

def _build_citations(conn: Any, pmids: list[str]) -> list[Citation]:
    out: list[Citation] = []
    for pmid in pmids:
        p = db.get_paper(conn, pmid)
        if not p:
            continue
        out.append(
            Citation(
                source="pubmed",
                pmid=p.pmid,
                url=p.pubmed_url,
                title=p.title,
                year=p.year,
                journal=p.journal,
                doi=p.doi,
            )
        )
    return out

def _build_external_citations(external_docs: list[Any], doc_ids: list[str]) -> list[Citation]:
    by_id = {str(doc.doc_id): doc for doc in external_docs or []}
    out: list[Citation] = []

    for doc_id in doc_ids:
        doc = by_id.get(str(doc_id))
        if not doc:
            continue

        out.append(
            Citation(
                source="external_rag",
                doc_id=str(doc.doc_id),
                title=doc.title or str(doc.doc_id),
                url=doc.url,
            )
        )

    return out
