from __future__ import annotations
import json
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

class RetrievalOnlyRequest(BaseModel):
    message: str = Field(..., min_length=1)
    mode: str = Field(default="patient", description="patient|doctor|menopause")
    session_id: Optional[str] = None
    context: Optional[str] = None
    pubmed_k: int = Field(default=5, ge=0, le=5)
    external_k: int = Field(default=5, ge=0, le=5)

class RetrievalSourceOut(BaseModel):
    source: str
    title: str
    url: Optional[str] = None
    full_text: str
    score: Optional[float] = None
    paper: dict[str, Any]

class RetrievalOnlyResponse(BaseModel):
    message: str
    mode: str
    retrieval_query: str
    pubmed_count: int
    external_count: int
    total_count: int
    sources: list[RetrievalSourceOut]


RETRIEVAL_RECOMMENDED_TEXT_CHARS = 15_000
RETRIEVAL_MAX_TEXT_CHARS = 30_000
RETRIEVAL_MAX_JSON_BYTES = 100 * 1024
TRUNCATION_SUFFIX = " ... [truncated]"


class SupportfastRetrievalRequest(BaseModel):
    message: str = Field(..., min_length=1)
    mode: Optional[str] = Field(default="patient", description="patient|doctor|menopause")
    context: str = ""

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

@app.post("/retrieval", response_model=RetrievalOnlyResponse)
def retrieval_only(req: RetrievalOnlyRequest) -> RetrievalOnlyResponse:
    settings = load_settings()
    if not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")

    conn = db.connect()
    db.init_db(conn)
    try:
        return _run_retrieval_only(conn=conn, req=req, settings=settings)
    finally:
        conn.close()

@app.post("/supportfast/retrieval", response_model=RetrievalOnlyResponse)
def supportfast_retrieval(req: SupportfastRetrievalRequest) -> RetrievalOnlyResponse:
    settings = load_settings()
    if not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")

    internal_req = RetrievalOnlyRequest(
        message=req.message,
        mode=req.mode or "patient",
        context=req.context or "",
        pubmed_k=5,
        external_k=5,
    )

    conn = db.connect()
    db.init_db(conn)
    try:
        return _run_retrieval_only(conn=conn, req=internal_req, settings=settings)
    finally:
        conn.close()

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

    if not _is_doctor_mode(req.mode):
        gyn_action = _gyn_orchestrator(
            oai,
            model=settings.openai_chat_model,
            message=req.message,
            history=history,
            phase="before_answer",
        )

        if gyn_action["action"] == "ask_area":
            answer_text = str(gyn_action["reply"] or "").strip()

            run = db.create_retrieval_run(
                conn,
                query="gyn_area_request",
                found_count=0,
                pmids=[],
            )

            db.finalize_message_ok(
                conn,
                message_id=message_id,
                answer=answer_text,
                retrieval_run_id=run.id,
                cited_pmids=[],
            )
            conn.close()

            return ChatResponse(
                answer=answer_text,
                retrieval=RetrievalInfo(
                    query="gyn_area_request",
                    found=0,
                    pmids=[],
                    cached=0,
                    fetched=0,
                ),
                citations=[],
                suggestions=[],
            )

        if gyn_action["action"] == "search":
            area = str(gyn_action["area"] or "").strip()

            suggestions = build_gyn_suggestions(
                ChatRequest(
                    message=req.message,
                    mode=req.mode,
                    session_id=session_id,
                    city=area,
                    address_hint=area,
                )
            )

            run = db.create_retrieval_run(
                conn,
                query="gyn_suggestions",
                found_count=len(suggestions),
                pmids=[],
            )

            ans = answer_gyn_suggestions_result(
                oai,
                model=settings.openai_chat_model,
                area=area,
                count=len(suggestions),
                mode=req.mode,
            )
            answer_text = (ans.text or "").strip()

            db.finalize_message_ok(
                conn,
                message_id=message_id,
                answer=answer_text,
                retrieval_run_id=run.id,
                cited_pmids=[],
            )
            conn.close()

            return ChatResponse(
                answer=answer_text,
                retrieval=RetrievalInfo(
                    query="gyn_suggestions",
                    found=len(suggestions),
                    pmids=[],
                    cached=0,
                    fetched=0,
                ),
                citations=[],
                suggestions=suggestions,
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
            suggestions: list[GynSuggestionOut] = []

            return ChatResponse(
                answer=answer_text,
                retrieval=RetrievalInfo(query="direct", found=0, pmids=[], cached=0, fetched=0),
                citations=citations,
                suggestions=suggestions,
            )

        pubmed_retmax = int(settings.pubmed_retmax)
        pubmed_top_k = int(settings.top_k)
        external_candidates = int(settings.external_candidates)
        final_external_k = int(settings.final_external_k)

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

        print(
            f"[RETRIEVAL] query={query_used!r} found={len(pmids)} cached={len(pmids) - fetched} fetched={fetched}",
            flush=True,
        )

        suggestions: list[GynSuggestionOut] = []

        if not _is_doctor_mode(req.mode) and len(history) >= 2:
            gyn_offer = _gyn_orchestrator(
                oai,
                model=settings.openai_chat_model,
                message=req.message,
                history=history,
                phase="after_answer",
                current_answer=answer_text,
            )

            if gyn_offer["action"] == "offer":
                offer_text = str(gyn_offer["reply"] or "").strip()
                if offer_text:
                    answer_text = answer_text.rstrip() + "\n\n" + offer_text

        db.finalize_message_ok(
            conn,
            message_id=message_id,
            answer=answer_text,
            retrieval_run_id=run.id,
            cited_pmids=cited_pmids,
        )
        
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

def _run_retrieval_only(*, conn: Any, req: RetrievalOnlyRequest, settings: Any) -> RetrievalOnlyResponse:
    pubmed_k = max(0, min(int(req.pubmed_k), 5))
    external_k = max(0, min(int(req.external_k), 5))
    retrieval_query = "\n\n".join(
        part.strip()
        for part in ((req.context or ""), req.message)
        if part and part.strip()
    )

    oai = OpenAIClient(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    pubmed = PubMedClient(
        api_key=settings.ncbi_api_key,
        tool=settings.ncbi_tool,
        email=settings.ncbi_email,
        timeout_s=settings.pubmed_timeout_s,
    )

    source_query = retrieval_query

    try:
        generated_query = oai.chat(
            model=settings.openai_chat_model,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You convert user questions into effective PubMed search queries. "
                        "Always return only one PubMed query in English. "
                        "Do not answer the question. "
                        "Do not return JSON. "
                        "Include biomedical synonyms and MeSH terms when useful. "
                        "Never return a generic gynecology or obstetrics query if the user asks about a specific topic."
                    ),
                },
                {
                    "role": "user",
                    "content": retrieval_query,
                },
            ],
        )
        generated_query = (generated_query or "").strip()
        if generated_query:
            source_query = generated_query
    except Exception as e:
        print(f"[SUPPORTFAST_QUERY_LLM_ERROR] {type(e).__name__}: {str(e)}", flush=True)
        source_query = retrieval_query

    pmids: list[str] = []
    query_used = ""
    
    if pubmed_k > 0:
        terms: list[str] = []
    
        if source_query != retrieval_query:
            terms.append(source_query)
    
        terms.extend(build_pubmed_term_candidates(retrieval_query))
    
        seen_terms: set[str] = set()
        unique_terms: list[str] = []
        for term in terms:
            term = (term or "").strip()
            if term and term not in seen_terms:
                seen_terms.add(term)
                unique_terms.append(term)
    
        for term in unique_terms:
            query_used = term
            pmids = pubmed.esearch(term, retmax=pubmed_k)
            if len(pmids) >= pubmed_k:
                break

    pmids = pmids[:pubmed_k]
    cached = db.get_cached_papers(conn, pmids)
    missing = [p for p in pmids if p not in cached]
    if missing:
        fetched_papers = pubmed.efetch(missing)
        db.upsert_papers(conn, fetched_papers)
        cached = db.get_cached_papers(conn, pmids)

    papers = [cached[p] for p in pmids if p in cached]
    pubmed_scores: dict[str, float] = {}
    if papers and pubmed_k > 0 and pubmed_k < len(papers):
        reranked = select_top_k(
            oai,
            embed_model=settings.openai_embed_model,
            question=retrieval_query,
            papers=papers,
            top_k=pubmed_k,
        )
        papers = reranked.papers
        pubmed_scores = {
            paper.pmid: float(score)
            for paper, score in zip(reranked.papers, reranked.scores)
        }

    sources: list[RetrievalSourceOut] = []
    for p in papers[:pubmed_k]:
        abstract = p.abstract or ""
        sources.append(
            RetrievalSourceOut(
                source="pubmed",
                title=p.title or "",
                url=p.pubmed_url,
                full_text=abstract,
                score=pubmed_scores.get(p.pmid),
                paper={
                    "pmid": p.pmid,
                    "title": p.title,
                    "abstract": p.abstract,
                    "year": p.year,
                    "journal": p.journal,
                    "doi": p.doi,
                    "url": p.pubmed_url,
                },
            )
        )

    external_sources: list[RetrievalSourceOut] = []
    if settings.external_rag_db_path and external_k > 0:
        if settings.external_rag_db_path.strip().lower().startswith("http"):
            raise RuntimeError("EXTERNAL_RAG_DB_PATH is a URL. Provide a local path instead.")
        p = Path(settings.external_rag_db_path)
        if p.is_dir():
            chroma_ext = connect_chroma(
                settings.external_rag_db_path,
                collection_name=settings.external_chroma_collection,
            )
            docs = retrieve_top_n_chroma(
                chroma_ext,
                oai=oai,
                embed_model=settings.openai_embed_model,
                question=source_query,
                top_n=external_k,
            )
            
            seen_external_doc_ids: set[str] = set()

            for doc in docs:
                base_doc_id = str(doc.doc_id or "").split("_")[0]
                text = (doc.text or "").strip()
            
                if not text:
                    continue
            
                if base_doc_id in seen_external_doc_ids:
                    continue
            
                seen_external_doc_ids.add(base_doc_id)
            
                external_sources.append(
                    RetrievalSourceOut(
                        source="external_rag",
                        title=doc.title or str(doc.doc_id),
                        url=doc.url,
                        full_text=text,
                        score=None,
                        paper={
                            "doc_id": doc.doc_id,
                            "title": doc.title,
                            "text": text,
                            "url": doc.url,
                        },
                    )
                )
        else:
            ext_conn = connect_external(settings.external_rag_db_path)
            try:
                q_vec = oai.embed(model=settings.openai_embed_model, text=source_query)
                hits = retrieve_top_n(ext_conn, query_vec=q_vec, top_n=external_k)
                seen_external_doc_ids: set[str] = set()

                for hit in hits:
                    doc = hit.doc
                    base_doc_id = str(doc.doc_id or "").split("_")[0]
                    text = (doc.text or "").strip()
                
                    if not text:
                        continue
                
                    if base_doc_id in seen_external_doc_ids:
                        continue
                
                    seen_external_doc_ids.add(base_doc_id)
                
                    external_sources.append(
                        RetrievalSourceOut(
                            source="external_rag",
                            title=doc.title or str(doc.doc_id),
                            url=doc.url,
                            full_text=text,
                            score=float(hit.score),
                            paper={
                                "doc_id": doc.doc_id,
                                "title": doc.title,
                                "text": text,
                                "url": doc.url,
                            },
                        )
                    )
                
                    if len(external_sources) >= external_k:
                        break
            finally:
                ext_conn.close()

    sources.extend(external_sources)
    response = RetrievalOnlyResponse(
        message=req.message,
        mode=req.mode,
        retrieval_query=query_used or source_query,
        pubmed_count=len(papers[:pubmed_k]),
        external_count=len(external_sources),
        total_count=len(sources),
        sources=sources,
    )
    return _limit_retrieval_response(response)


def _limit_retrieval_response(response: RetrievalOnlyResponse) -> RetrievalOnlyResponse:
    """
    Keep Supportfast retrieval payloads within agreed limits.
    Sources are already ordered with PubMed first, then external RAG, so dropping
    from the end preserves PubMed priority.
    """
    _trim_response_text(response, RETRIEVAL_MAX_TEXT_CHARS)
    if _response_json_size(response) <= RETRIEVAL_MAX_JSON_BYTES:
        return _refresh_retrieval_counts(response)

    for max_chars in (RETRIEVAL_RECOMMENDED_TEXT_CHARS, 10_000, 6_000, 3_000):
        _trim_response_text(response, max_chars)
        if _response_json_size(response) <= RETRIEVAL_MAX_JSON_BYTES:
            return _refresh_retrieval_counts(response)

    while response.sources and _response_json_size(response) > RETRIEVAL_MAX_JSON_BYTES:
        response.sources.pop()

    return _refresh_retrieval_counts(response)


def _trim_response_text(response: RetrievalOnlyResponse, max_chars: int) -> None:
    remaining = max(0, int(max_chars))
    for source in response.sources:
        text = source.full_text or ""
        if remaining <= 0:
            _set_source_text(source, "")
            continue

        if len(text) > remaining:
            text = _truncate_text(text, remaining)

        _set_source_text(source, text)
        remaining -= len(text)


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= len(TRUNCATION_SUFFIX):
        return text[:max_chars]
    return text[: max_chars - len(TRUNCATION_SUFFIX)].rstrip() + TRUNCATION_SUFFIX


def _set_source_text(source: RetrievalSourceOut, text: str) -> None:
    source.full_text = text
    if source.source == "pubmed":
        source.paper["abstract"] = text
    elif source.source == "external_rag":
        source.paper["text"] = text


def _response_json_size(response: RetrievalOnlyResponse) -> int:
    return len(json.dumps(_response_to_plain_dict(response), ensure_ascii=False).encode("utf-8"))


def _response_to_plain_dict(response: RetrievalOnlyResponse) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return response.dict()


def _refresh_retrieval_counts(response: RetrievalOnlyResponse) -> RetrievalOnlyResponse:
    response.pubmed_count = sum(1 for s in response.sources if s.source == "pubmed")
    response.external_count = sum(1 for s in response.sources if s.source == "external_rag")
    response.total_count = len(response.sources)
    return response


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

def _detect_gyn_request(
    oai: OpenAIClient,
    *,
    model: str,
    message: str,
    history: list[dict[str, str]],
) -> tuple[bool, str | None]:
    recent_history = "\n".join(
        f"Utente: {item.get('question', '')}\n"
        f"Assistente: {item.get('answer', '')}"
        for item in history[-5:]
    )

    prompt = f"""
Analizza il messaggio e stabilisci se l'utente vuole trovare, scegliere
o ricevere il contatto di una ginecologa o professionista ginecologica.

Comprendi qualsiasi formulazione naturale, inclusi:
- dottoressa, ginecologa, specialista, professionista;
- "da chi posso andare";
- "chi mi consigli";
- "conosci qualcuna";
- "vicino a me";
- richieste espresse nel contesto dei messaggi precedenti.

Estrai città o zona solo se indicata esplicitamente nel messaggio o nella
conversazione. Non inventare località.

Conversazione:
{recent_history or "<nessuna>"}

Messaggio attuale:
{message}

Rispondi esclusivamente con JSON:
{{
  "wants_gynecologist": true,
  "area": "Roma"
}}

Se manca la località, usa null.
""".strip()

    result = oai.chat(
        model=model,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": "Sei un classificatore di richieste. Produci solo JSON valido.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    try:
        data = json.loads(result.strip())
    except Exception:
        return False, None

    wants = bool(data.get("wants_gynecologist"))
    area = str(data.get("area") or "").strip() or None
    return wants, area

def _needs_clarification(text: str) -> bool:
    q = (text or "").strip().lower()
    words = [w for w in q.replace("?", " ").split() if w]
    if len(words) <= 5 and any(symptom in q for symptom in _GENERIC_SYMPTOMS):
        return True
    return False

def _is_doctor_mode(mode: str) -> bool:
    return (mode or "").strip().lower() in {"doctor", "medico", "ginecologo", "ginecologa"}

def _gyn_orchestrator(
    oai: OpenAIClient,
    *,
    model: str,
    message: str,
    history: list[dict[str, str]],
    phase: str,
    current_answer: str | None = None,
) -> dict[str, str | None]:
    conversation = "\n\n".join(
        f"Utente: {(item.get('question') or '').strip()}\n"
        f"Assistente: {(item.get('answer') or '').strip()}"
        for item in history[-10:]
    )

    system_prompt = """
Sei l'orchestratore della funzione che cerca ginecologhe in un database locale.

Comprendi semanticamente messaggio e conversazione, anche con linguaggio
colloquiale, richieste indirette o errori grammaticali.

Azioni:
- "none": non occorre usare il database delle ginecologhe.
- "ask_area": l'utente cerca una ginecologa o professionista, ma manca una località precisa.
- "search": l'utente cerca una professionista e la località è disponibile.
- "offer": dopo una conversazione clinica è opportuno proporre la ricerca.

Regole:
- Cerca la località anche nella cronologia.
- Non inventare mai una località.
- "Vicino a me" o "nella mia zona" non sono località sufficienti.
- Se l'assistente aveva chiesto la zona e l'utente risponde con una città,
  usa "search".
- Con "ask_area", genera in "reply" una domanda naturale e contestuale.
- Con "offer", genera in "reply" una proposta breve e non insistente.
- Con "search", inserisci in "area" solo città o zona da passare al database.
- Non suggerire ricerche online: il sistema possiede un database locale.
- Restituisci esclusivamente JSON valido.
""".strip()

    user_prompt = f"""
Fase: {phase}

Conversazione precedente:
{conversation or "<nessuna>"}

Messaggio attuale:
{message}

Risposta clinica corrente:
{current_answer or "<nessuna>"}

Nella fase "before_answer" usa solo:
"none", "ask_area", "search".

Nella fase "after_answer" usa solo:
"none", "offer".

Formato:
{{
  "action": "none",
  "area": null,
  "reply": null
}}
""".strip()

    try:
        raw = oai.chat(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        data = json.loads((raw or "").strip())
    except Exception:
        return {"action": "none", "area": None, "reply": None}

    action = str(data.get("action") or "none").strip().lower()
    area = str(data.get("area") or "").strip() or None
    reply = str(data.get("reply") or "").strip() or None

    allowed = (
        {"none", "ask_area", "search"}
        if phase == "before_answer"
        else {"none", "offer"}
    )

    if action not in allowed:
        return {"action": "none", "area": None, "reply": None}

    if action == "search" and not area:
        return {"action": "none", "area": None, "reply": None}

    if action in {"ask_area", "offer"} and not reply:
        return {"action": "none", "area": None, "reply": None}

    return {
        "action": action,
        "area": area,
        "reply": reply,
    }

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
    by_title = {(doc.title or "").strip().lower(): doc for doc in external_docs or []}
    out: list[Citation] = []

    for doc_title in doc_ids:
        doc = by_title.get(str(doc_title).strip().lower())
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
