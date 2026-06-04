from __future__ import annotations


def prompt_user_mode() -> str:
    """
    CLI-only helper.
    Asks once at startup which user category applies, then returns a normalized mode:
      - "patient"
      - "menopause"
      - "doctor"
    """
    print("Seleziona la tua categoria (una sola volta, vale per tutta la sessione):")
    print("  A) Donna NON in menopausa")
    print("  B) Donna IN menopausa")
    print("  C) Medico ginecologo/a")

    while True:
        try:
            raw = input("Scelta [A/B/C]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("")
            return "patient"

        if raw in {"a", "1", "utente", "patient"}:
            return "patient"
        if raw in {"b", "2", "menopausa", "menopause"}:
            return "menopause"
        if raw in {"c", "3", "doctor", "medico"}:
            return "doctor"

        print("Scelta non valida. Inserisci A, B o C.")
