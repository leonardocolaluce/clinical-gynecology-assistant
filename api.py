from __future__ import annotations

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
    answer_direct,
    answer_with_pubmed,
    answer_with_pubmed_and_external,
    extract_cited_pmids,
    revise_to_meet_min_citations,
)
from .flow._4_router import allow_direct_without_sources, decide_route
from .flow._9_external_rag import connect_external, retrieve_top_n
from .flow._13_external_chroma import connect_chroma, retrieve_top_n_chroma
from .flow._14_debug import dbg


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    mode: str = Field(default="patient", description="patient|doctor|menopause")
    session_id: Optional[str] = None
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

class Citation(BaseModel):
    pmid: str
    url: str
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

class PromptConfig(BaseModel):
    patient: str
    menopause: str
    doctor: str

app = FastAPI(title="Pipeline M1 - Chatbot Gin", version="0.1")


@app.on_event("startup")
def _startup() -> None:
    conn = db.connect()
    db.init_db(conn)
    conn.close()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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

    try:
        dbg(f"/chat mode={req.mode!r} msg_len={len((req.message or '').strip())}")
        if settings.external_rag_db_path:
            p = Path(settings.external_rag_db_path)
            dbg(f"EXTERNAL_RAG_DB_PATH={settings.external_rag_db_path!r} exists={p.exists()} is_dir={p.is_dir()} is_file={p.is_file()}")
        oai = OpenAIClient(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        pubmed = PubMedClient(
            api_key=settings.ncbi_api_key,
            tool=settings.ncbi_tool,
            email=settings.ncbi_email,
            timeout_s=settings.pubmed_timeout_s,
        )

        # Router (safe-by-default): only allow direct for clearly non-medical/meta queries.
        try:
            decision = decide_route(oai, model=settings.openai_chat_model, question=req.message)
        except Exception:
            decision = None
        use_direct = bool(decision and decision.route == "direct" and allow_direct_without_sources(req.message))
        if use_direct:
            run = db.create_retrieval_run(conn, query="direct", found_count=0, pmids=[])
            ans = answer_direct(oai, model=settings.openai_chat_model, question=req.message, mode=req.mode, history=history)
            answer_text = (ans.text or "").strip()
            cited_pmids: list[str] = []
            citations: list[Citation] = []
            db.finalize_message_ok(conn, message_id=message_id, answer=answer_text, retrieval_run_id=run.id, cited_pmids=cited_pmids)
            return ChatResponse(
                answer=answer_text,
                retrieval=RetrievalInfo(query="direct", found=0, pmids=[], cached=0, fetched=0),
                citations=citations,
            )

        pmids: list[str] = []
        query_used: str = ""
        terms = []
        if decision and decision.route == "pubmed" and decision.term:
            terms = [decision.term]
        else:
            terms = list(build_pubmed_term_candidates(req.message))

        for term in terms:
            query_used = term
            pmids = pubmed.esearch(term, retmax=settings.pubmed_retmax)
            if len(pmids) >= settings.pubmed_retmax:
                break

        pmids = pmids[: settings.pubmed_retmax]
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
        if papers and int(settings.top_k) > 0 and int(settings.top_k) < len(papers):
            reranked = select_top_k(
                oai,
                embed_model=settings.openai_embed_model,
                question=req.message,
                papers=papers,
                top_k=int(settings.top_k),
            )
            papers = reranked.papers

        external_docs = []
        print("[EUROPEPMC] external block reached", flush=True)
        if settings.external_rag_db_path and int(settings.external_candidates) > 0 and int(settings.final_external_k) > 0:
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
                        question=req.message,
                        top_n=int(settings.external_candidates),
                    )
                    external_docs = docs[: max(0, int(settings.final_external_k))]
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
                    hits = retrieve_top_n(ext_conn, query_vec=q_vec, top_n=int(settings.external_candidates))
                    external_docs = [h.doc for h in hits[: max(0, int(settings.final_external_k))]]
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
        if int(settings.min_distinct_citations) > 1:
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
        citations = _build_citations(conn, cited_pmids)

        db.finalize_message_ok(conn, message_id=message_id, answer=answer_text, retrieval_run_id=run.id, cited_pmids=cited_pmids)

        print(
            f"[RETRIEVAL] query={query_used!r} found={len(pmids)} cached={len(pmids) - fetched} fetched={fetched}",
            flush=True,
        )

        suggestions: list[GynSuggestionOut] = []
        has_location = bool(req.city or req.address_hint or req.latitude is not None or req.longitude is not None)

        if req.mode != "doctor" and has_location and req.city:
            raw_suggestions = suggest_top3(city=req.city, address_hint=req.address_hint)
            suggestions = [
                GynSuggestionOut(
                    name=s.name,
                    address=s.address,
                    phone=s.phone,
                    website=s.website,
                    emails=s.emails,
                    rating=s.rating,
                    reviews=s.reviews,
                )
                for s in raw_suggestions
            ]

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


def _build_citations(conn: Any, pmids: list[str]) -> list[Citation]:
    out: list[Citation] = []
    for pmid in pmids:
        p = db.get_paper(conn, pmid)
        if not p:
            continue
        out.append(
            Citation(
                pmid=p.pmid,
                url=p.pubmed_url,
                title=p.title,
                year=p.year,
                journal=p.journal,
                doi=p.doi,
            )
        )
    return out
