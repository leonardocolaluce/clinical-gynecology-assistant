from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Paper:
    pmid: str
    title: str
    abstract: str
    year: Optional[str] = None
    journal: Optional[str] = None
    doi: Optional[str] = None

    @property
    def pubmed_url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"


@dataclass(frozen=True)
class ExternalDoc:
    doc_id: str
    title: str
    text: str
    url: Optional[str] = None
