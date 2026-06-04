from __future__ import annotations

import os


def debug_enabled() -> bool:
    v = (os.getenv("DEBUG_RAG") or os.getenv("PIPELINE_DEBUG") or "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def dbg(msg: str) -> None:
    if debug_enabled():
        print(f"[DEBUG] {msg}")

