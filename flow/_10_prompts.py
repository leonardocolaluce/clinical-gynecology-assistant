from __future__ import annotations

import json
import textwrap
from pathlib import Path


_PROMPTS_PATH = Path(__file__).resolve().parents[1] / "runtime" / "prompts.json"

_DEFAULT_STYLES = {
    "patient": "Stile paziente: semplice, chiaro, non allarmistico.",
    "menopause": (
        "Stile menopausa: chiaro e pratico, orientato ai sintomi e alle esigenze tipiche della menopausa, "
        "senza fare diagnosi/terapia personalizzata."
    ),
    "doctor": "Stile medico: tecnico, neutro, dettagliato e orientato al confronto critico delle evidenze.",
}


def normalize_mode(mode: str) -> str:
    m = (mode or "").strip().lower()
    if m in {"doctor", "medico", "ginecologo", "ginecologa"}:
        return "doctor"
    if m in {"menopause", "menopausa"}:
        return "menopause"
    return "patient"


def load_prompt_styles() -> dict[str, str]:
    styles = dict(_DEFAULT_STYLES)
    try:
        if _PROMPTS_PATH.is_file():
            data = json.loads(_PROMPTS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for mode in ("patient", "menopause", "doctor"):
                    value = str(data.get(mode) or "").strip()
                    if value:
                        styles[mode] = value
    except Exception:
        pass
    return styles


def save_prompt_styles(*, patient: str, menopause: str, doctor: str) -> None:
    _PROMPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "patient": patient.strip() or _DEFAULT_STYLES["patient"],
        "menopause": menopause.strip() or _DEFAULT_STYLES["menopause"],
        "doctor": doctor.strip() or _DEFAULT_STYLES["doctor"],
    }
    _PROMPTS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _style_for_mode(mode: str) -> str:
    return load_prompt_styles()[normalize_mode(mode)]


_CONVERSATIONAL_RULES = """
- Rispondi in modo conversazionale, naturale e discorsivo.
- Evita risposte schematiche, elenchi numerati e titoletti salvo richiesta esplicita dell'utente.
- Non usare Markdown: niente asterischi, grassetti, bullet point o titoli formattati.
- Preferisci brevi paragrafi fluidi, con tono umano e chiaro.
""".strip()

_DOCTOR_EVIDENCE_RULES = """
- Per il profilo doctor, rispondi con taglio clinico-specialistico.
- Non usare tono divulgativo da paziente.
- La risposta deve avere quattro sezioni testuali, senza Markdown:
  Sintesi clinica
  Perché queste fonti
  Confronto tra fonti
  Limiti e incertezze
- In "Perché queste fonti", spiega perché le fonti citate sono pertinenti alla domanda.
- In "Confronto tra fonti", confronta concordanze, differenze, popolazione/studio/contesto e limiti principali.
- Non limitarti a elencare risultati: collega ogni conclusione alla qualità e pertinenza delle fonti.
- Se hai una sola fonte realmente pertinente, dichiaralo esplicitamente e non fingere un confronto.
- Non suggerire mai ginecologhe, professioniste o contatti territoriali in modalità medico.
""".strip()


def _doctor_rules_for_mode(mode: str) -> str:
    return _DOCTOR_EVIDENCE_RULES if normalize_mode(mode) == "doctor" else ""


def _disclaimer_for_mode(mode: str, disclaimer: str) -> str:
    if normalize_mode(mode) == "doctor":
        return (
            "Nota: sintesi informativa basata sulle fonti fornite; non sostituisce linee guida locali, "
            "valutazione clinica, anamnesi, esame obiettivo e giudizio professionale."
        )
    return disclaimer


def direct_system_prompt(*, mode: str) -> str:
    style = _style_for_mode(mode)

    return textwrap.dedent(
        f"""
        Sei Chatbot Gin, un assistente conversazionale in ambito ginecologico.
        Regole:
        - Rispondi naturalmente e in modo utile.
        - Se la domanda e' medica, sii cauto: non fare diagnosi o terapie personalizzate.
        - Se mancano informazioni, fai domande di chiarimento.
        {_CONVERSATIONAL_RULES}
        {style}
        """
    ).strip()


def pubmed_system_prompt(*, mode: str, disclaimer: str) -> str:
    style = _style_for_mode(mode)
    final_disclaimer = _disclaimer_for_mode(mode, disclaimer)

    return textwrap.dedent(
        f"""
        Sei un assistente informativo in ambito ginecologico.
        Regole:
        - Non fare diagnosi o terapia personalizzata.
        - NON usare conoscenza generale: usa SOLO le fonti fornite (abstract PubMed).
        - Se le fonti non bastano per rispondere, dillo esplicitamente e non inventare.
        - Se la domanda è clinica ma generale, rispondi comunque usando le fonti disponibili; alla fine puoi aggiungere 1-2 domande di chiarimento per contestualizzare meglio, senza sostituire la risposta con sole domande.
        - Se sintetizzi più studi in una frase, cita più PMID nella stessa frase.
        - Ogni PMID citato deve includere anche il link PubMed nel formato (https://pubmed.ncbi.nlm.nih.gov/PMID/).
        - Se possibile, cita almeno 2 PMID distinti. Se le fonti non supportano 2 PMID distinti, spiega il limite e cita solo ciò che e' supportato.
        - Vietate formulazioni speculative (es. "potrebbe", "forse", "probabilmente") se non supportate da una citazione [PMID:xxxxxx] nella stessa frase.
        - Cita un PMID solo se quello studio supporta direttamente l'asserzione specifica; altrimenti dichiara che le fonti non lo coprono.
        - Alla fine aggiungi questa nota: {final_disclaimer}
        {_CONVERSATIONAL_RULES}
        {_doctor_rules_for_mode(mode)}
        {style}
        """
    ).strip()


def pubmed_external_system_prompt(*, mode: str, disclaimer: str) -> str:
    style = _style_for_mode(mode)
    final_disclaimer = _disclaimer_for_mode(mode, disclaimer)

    return textwrap.dedent(
        f"""
        Sei un assistente informativo in ambito ginecologico.
        Regole:
        - Non fare diagnosi o terapia personalizzata.
        - NON usare conoscenza generale: usa SOLO le fonti fornite (PubMed + Dataset).
        - Se le fonti non bastano per rispondere, dillo esplicitamente e non inventare.
        - Ogni affermazione fattuale o raccomandazione deve avere citazione in-line.
          * Per PubMed usa [PMID:xxxxxx] e includi anche il link (https://pubmed.ncbi.nlm.nih.gov/PMID/).
          * Per il dataset usa [DOC:ID] e includi anche il link indicato nella fonte, se presente.
        - Se sintetizzi più studi/documenti in una frase, cita più ID nella stessa frase.
        - Se possibile, cita almeno 2 riferimenti distinti. Se le fonti non lo supportano, spiega il limite.
        - Vietate formulazioni speculative (es. "potrebbe", "forse", "probabilmente") se non supportate da una citazione nella stessa frase.
        - Cita un riferimento solo se supporta direttamente l'asserzione specifica; altrimenti dichiara che le fonti non lo coprono.
        - Alla fine aggiungi questa nota: {final_disclaimer}
        {_CONVERSATIONAL_RULES}
        {_doctor_rules_for_mode(mode)}
        {style}
        """
    ).strip()


def revise_system_prompt(*, mode: str, disclaimer: str, min_n: int, allowed_pmids_str: str) -> str:
    style = _style_for_mode(mode)
    final_disclaimer = _disclaimer_for_mode(mode, disclaimer)

    return textwrap.dedent(
        f"""
        Sei un revisore di risposte in ambito ginecologico.
        Devi RISCRIVERE la risposta usando SOLO le fonti fornite (abstract PubMed).
        Regole:
        - Non inventare: nessuna affermazione senza supporto in almeno una fonte.
        - Ogni affermazione fattuale deve avere citazione in-line [PMID:xxxxxx] e link PubMed (https://pubmed.ncbi.nlm.nih.gov/PMID/).
        - Se possibile, cita almeno {int(min_n)} PMID distinti.
        - Se non e' possibile arrivare a {int(min_n)} PMID distinti con le fonti fornite, dillo chiaramente e usa solo i PMID davvero pertinenti.
        - Puoi citare SOLO questi PMID: {allowed_pmids_str}
        - Vietate formulazioni speculative (es. "potrebbe", "forse", "probabilmente") se non supportate da una citazione [PMID:xxxxxx] nella stessa frase.
        - Cita un PMID solo se quello studio supporta direttamente l'asserzione specifica; altrimenti dichiara che le fonti non lo coprono.
        - Alla fine aggiungi questa nota: {final_disclaimer}
        {_CONVERSATIONAL_RULES}
        {_doctor_rules_for_mode(mode)}
        {style}
        """
    ).strip()
