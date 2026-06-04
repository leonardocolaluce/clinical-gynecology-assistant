from __future__ import annotations

import sys
from pathlib import Path

try:
    from pipeline_m2 import db
    from pipeline_m2.flow._1_config import load_settings
    from pipeline_m2.flow._3_openai_client import OpenAIClient
    from pipeline_m2.flow._5_pubmed_client import PubMedClient
    from pipeline_m2.flow._6_query_builder import build_pubmed_term_candidates
    from pipeline_m2.flow._7_retrieval import select_top_k
    from pipeline_m2.flow._8_answering import (
        answer_direct,
        answer_with_pubmed,
        answer_with_pubmed_and_external,
        extract_cited_pmids,
        revise_to_meet_min_citations,
    )
    from pipeline_m2.flow._9_external_rag import connect_external, retrieve_top_n
    from pipeline_m2.flow._4_router import allow_direct_without_sources, decide_route
    from pipeline_m2.flow._11_mode_select import prompt_user_mode
    from pipeline_m2.flow._12_gyn_suggest import suggest_top3
    from pipeline_m2.flow._13_external_chroma import connect_chroma, retrieve_top_n_chroma
    from pipeline_m2.flow._14_debug import dbg
except ModuleNotFoundError:
    # When executed as a script (python path/to/main.py), ensure the package root is importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../2_Milestone
    from pipeline_m2 import db
    from pipeline_m2.flow._1_config import load_settings
    from pipeline_m2.flow._3_openai_client import OpenAIClient
    from pipeline_m2.flow._5_pubmed_client import PubMedClient
    from pipeline_m2.flow._6_query_builder import build_pubmed_term_candidates
    from pipeline_m2.flow._7_retrieval import select_top_k
    from pipeline_m2.flow._8_answering import (
        answer_direct,
        answer_with_pubmed,
        answer_with_pubmed_and_external,
        extract_cited_pmids,
        revise_to_meet_min_citations,
    )
    from pipeline_m2.flow._9_external_rag import connect_external, retrieve_top_n
    from pipeline_m2.flow._4_router import allow_direct_without_sources, decide_route
    from pipeline_m2.flow._11_mode_select import prompt_user_mode
    from pipeline_m2.flow._12_gyn_suggest import suggest_top3
    from pipeline_m2.flow._13_external_chroma import connect_chroma, retrieve_top_n_chroma
    from pipeline_m2.flow._14_debug import dbg


def main() -> int:
    settings = load_settings()
    if not settings.openai_api_key:
        print(
            "Manca OPENAI_API_KEY. Impostala come variabile d'ambiente oppure mettila in `pipeline_m2/config` "
            '(es. `OPENAI_API_KEY="sk-..."` o `api_key="sk-..."`).',
            file=sys.stderr,
        )
        return 2

    oai = OpenAIClient(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    pubmed = PubMedClient(
        api_key=settings.ncbi_api_key,
        tool=settings.ncbi_tool,
        email=settings.ncbi_email,
        timeout_s=settings.pubmed_timeout_s,
    )

    conn = db.connect()
    db.init_db(conn)

    ext_conn = None
    chroma_ext = None
    if settings.external_rag_db_path:
        try:
            p = Path(settings.external_rag_db_path)
            dbg(f"EXTERNAL_RAG_DB_PATH={settings.external_rag_db_path!r} resolved={str(p.resolve())!r} exists={p.exists()} is_dir={p.is_dir()} is_file={p.is_file()}")
            dbg(f"EXTERNAL_CHROMA_COLLECTION={settings.external_chroma_collection!r}")
            if p.is_dir():
                chroma_ext = connect_chroma(settings.external_rag_db_path, collection_name=settings.external_chroma_collection)
                print(f"[ExternalRAG] Chroma collegato: {settings.external_rag_db_path} (collection={settings.external_chroma_collection})")
            else:
                ext_conn = connect_external(settings.external_rag_db_path)
                print(f"[ExternalRAG] DB collegato: {settings.external_rag_db_path}")
        except Exception as e:
            ext_conn = None
            chroma_ext = None
            dbg(f"ExternalRAG init error: {type(e).__name__}: {str(e).strip()}")
            print(f"[ExternalRAG] DB non disponibile: {str(e).strip()}")
    else:
        print("[ExternalRAG] Disattivato (imposta EXTERNAL_RAG_DB_PATH).")

    print("Assistente virtuale. Ctrl+C per uscire.\n")
    print("Fai una domanda e ti rispondo con fonti citate da PubMed e Europe PMC.\n")

    mode = prompt_user_mode()
    print(f"\nModalità selezionata: {mode}\n")

    city = ""
    address_hint: str | None = None
    if mode != "doctor":
        while True:
            city = input("Inserisci la tua città (obbligatoria): ").strip()
            if city:
                break
            print("La città è obbligatoria.")
        address_hint = input("Inserisci via/indirizzo (opzionale, per suggerimenti più vicini): ").strip() or None
        print("")

    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            if ext_conn is not None:
                ext_conn.close()
            conn.close()
            return 0

        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            if ext_conn is not None:
                ext_conn.close()
            conn.close()
            return 0

        message_id = db.create_message(conn, mode=mode, question=q)
        try:
            # Router (safe-by-default): allow "direct" only for clearly non-medical/meta questions.
            try:
                decision = decide_route(oai, model=settings.openai_chat_model, question=q)
            except Exception:
                decision = None

            use_direct = bool(decision and decision.route == "direct" and allow_direct_without_sources(q))

            if use_direct:
                run = db.create_retrieval_run(conn, query="direct", found_count=0, pmids=[])
                ans = answer_direct(oai, model=settings.openai_chat_model, question=q, mode=mode)
                answer_text = (ans.text or "").strip()
                cited_pmids: list[str] = []
            else:
                pmids: list[str] = []
                term_used: str | None = None

                # Use router term if provided, otherwise build candidates as before.
                if decision and decision.route == "pubmed" and decision.term:
                    terms = [decision.term]
                else:
                    terms = list(build_pubmed_term_candidates(q))

                for term in terms:
                    term_used = term
                    pmids = pubmed.esearch(term_used, retmax=settings.pubmed_retmax)
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
                print(f"\n[PubMed] Trovati {len(papers)} file. (cache={len(pmids) - fetched}, fetched={fetched})")

                run = db.create_retrieval_run(conn, query=term_used or "", found_count=len(pmids), pmids=pmids)

                pubmed_k = int(settings.final_pubmed_k) if int(settings.final_pubmed_k) > 0 else int(settings.top_k)
                if papers and pubmed_k > 0 and pubmed_k < len(papers):
                    reranked = select_top_k(
                        oai,
                        embed_model=settings.openai_embed_model,
                        question=q,
                        papers=papers,
                        top_k=pubmed_k,
                    )
                    papers = reranked.papers
                    print(f"[Retrieval] PubMed: selezionati top_k={len(papers)} abstract più rilevanti.")

                external_docs = []
                if int(settings.external_candidates) > 0 and int(settings.final_external_k) > 0:
                    if chroma_ext is not None:
                        try:
                            docs = retrieve_top_n_chroma(
                                chroma_ext,
                                oai=oai,
                                embed_model=settings.openai_embed_model,
                                question=q,
                                top_n=int(settings.external_candidates),
                            )
                            external_docs = docs[: max(0, int(settings.final_external_k))]
                            print(f"[Retrieval] External(Chroma): candidati={len(docs)}, final_k={len(external_docs)}")
                            if external_docs:
                                dbg(f"External(Chroma) sample id={external_docs[0].doc_id!r} title={(external_docs[0].title or '')[:80]!r}")
                        except Exception as e:
                            dbg(f"External(Chroma) retrieval error: {type(e).__name__}: {str(e).strip()}")
                            print(f"[ExternalRAG] Warning: {str(e).strip()}")
                    elif ext_conn is not None:
                        q_vec = oai.embed(model=settings.openai_embed_model, text=q)
                        hits = retrieve_top_n(
                            ext_conn,
                            query_vec=q_vec,
                            top_n=int(settings.external_candidates),
                        )
                        external_docs = [h.doc for h in hits[: max(0, int(settings.final_external_k))]]
                        print(f"[Retrieval] External(SQLite): candidati={len(hits)}, final_k={len(external_docs)}")
                        if external_docs:
                            dbg(f"External(SQLite) sample id={external_docs[0].doc_id!r} title={(external_docs[0].title or '')[:80]!r}")

                if external_docs:
                    ans = answer_with_pubmed_and_external(
                        oai,
                        model=settings.openai_chat_model,
                        question=q,
                        mode=mode,
                        disclaimer=settings.disclaimer,
                        pubmed_papers=papers,
                        external_docs=external_docs,
                    )
                    answer_text = (ans.text or "").strip()
                else:
                    ans = answer_with_pubmed(
                        oai,
                        model=settings.openai_chat_model,
                        question=q,
                        mode=mode,
                        disclaimer=settings.disclaimer,
                        papers=papers,
                    )
                    answer_text = (ans.text or "").strip()

                    if int(settings.min_distinct_citations) > 1:
                        revised = revise_to_meet_min_citations(
                            oai,
                            model=settings.openai_chat_model,
                            question=q,
                            mode=mode,
                            disclaimer=settings.disclaimer,
                            papers=papers,
                            draft_answer=answer_text,
                            min_distinct_pmids=int(settings.min_distinct_citations),
                        )
                        answer_text = (revised.text or "").strip()

                cited_pmids = extract_cited_pmids(answer_text)

            db.finalize_message_ok(conn, message_id=message_id, answer=answer_text, retrieval_run_id=run.id, cited_pmids=cited_pmids)
            print("\n" + answer_text + "\n")

            if mode != "doctor" and city:
                suggestions = suggest_top3(city=city, address_hint=address_hint)
                if suggestions:
                    print("Suggerimento ginecologa per la tua località (Top 3):")
                    for idx, s in enumerate(suggestions, start=1):
                        meta = []
                        if s.phone:
                            meta.append(f"Tel: {s.phone}")
                        if s.website:
                            meta.append(f"Sito: {s.website}")
                        if s.emails:
                            meta.append(f"Email: {s.emails}")
                        meta_str = " | ".join(meta)
                        print(f"{idx}) {s.name} — {s.address}")
                        if meta_str:
                            print(f"   {meta_str}")
                    print("")
        except Exception as e:
            # Don't crash the chat loop on transient API errors/rate limits.
            db.finalize_message_error(conn, message_id=message_id, error=str(e))
            print("\n" + str(e).strip() + "\n")
            continue


if __name__ == "__main__":
    raise SystemExit(main())
