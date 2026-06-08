from __future__ import annotations
import ssl
import certifi
import time
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable, Optional

from ._2_models import Paper


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


@dataclass(frozen=True)
class PubMedClient:
    api_key: str
    tool: str
    email: str
    timeout_s: int = 30
    min_delay_s: float = 0.12

    def _get(self, path: str, params: dict[str, str]) -> bytes:
        clean_params = {k: v for k, v in (params or {}).items() if v is not None and str(v).strip() != ""}
        url = EUTILS_BASE + path + "?" + urllib.parse.urlencode(clean_params)
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": f"{self.tool}/0.1 ({self.email})",
                "Accept": "application/xml,text/xml,*/*;q=0.8",
            },
            method="GET",
        )
        time.sleep(self.min_delay_s)
        try:
            ssl_context = ssl.create_default_context(cafile=certifi.where())
        
            with urllib.request.urlopen(req, timeout=self.timeout_s, context=ssl_context) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            raise RuntimeError(f"PubMed HTTP error {e.code}. {detail}".strip()) from e

    def esearch(self, term: str, *, retmax: int) -> list[str]:
        data = self._get(
            "esearch.fcgi",
            {
                "db": "pubmed",
                "term": term,
                "retmode": "xml",
                "retmax": str(retmax),
                "api_key": (self.api_key or "").strip(),
                "tool": self.tool,
                "email": self.email,
                "sort": "relevance",
            },
        )
        root = ET.fromstring(data)
        return [e.text.strip() for e in root.findall(".//IdList/Id") if e.text and e.text.strip()]

    def efetch(self, pmids: Iterable[str]) -> list[Paper]:
        pmid_list = [p.strip() for p in pmids if p and p.strip()]
        if not pmid_list:
            return []
        data = self._get(
            "efetch.fcgi",
            {
                "db": "pubmed",
                "id": ",".join(pmid_list),
                "retmode": "xml",
                "api_key": (self.api_key or "").strip(),
                "tool": self.tool,
                "email": self.email,
            },
        )
        root = ET.fromstring(data)
        out: list[Paper] = []
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//MedlineCitation/PMID")
            pmid = (pmid_el.text or "").strip() if pmid_el is not None else ""
            if not pmid:
                continue
            title_el = article.find(".//Article/ArticleTitle")
            title = "".join(title_el.itertext()).strip() if title_el is not None else ""
            abstract_els = article.findall(".//Article/Abstract/AbstractText")
            parts: list[str] = []
            for ael in abstract_els:
                part = "".join(ael.itertext()).strip()
                if not part:
                    continue
                label = (ael.attrib.get("Label") or "").strip()
                parts.append(f"{label}: {part}" if label else part)
            abstract = "\n".join(parts).strip()
            out.append(
                Paper(
                    pmid=pmid,
                    title=title,
                    abstract=abstract,
                    year=_extract_year(article),
                    journal=_extract_journal(article),
                    doi=_extract_doi(article),
                )
            )
        return out


def _extract_year(article: ET.Element) -> Optional[str]:
    year_el = article.find(".//Article/Journal/JournalIssue/PubDate/Year")
    if year_el is not None and year_el.text and year_el.text.strip():
        return year_el.text.strip()
    medline_date_el = article.find(".//Article/Journal/JournalIssue/PubDate/MedlineDate")
    if medline_date_el is not None and medline_date_el.text and medline_date_el.text.strip():
        tok = medline_date_el.text.strip().split()[0]
        if tok.isdigit():
            return tok
    return None


def _extract_journal(article: ET.Element) -> Optional[str]:
    j_el = article.find(".//Article/Journal/Title")
    if j_el is not None and j_el.text and j_el.text.strip():
        return j_el.text.strip()
    return None


def _extract_doi(article: ET.Element) -> Optional[str]:
    for aid in article.findall(".//ArticleIdList/ArticleId"):
        if (aid.attrib.get("IdType") or "").lower() == "doi" and aid.text and aid.text.strip():
            return aid.text.strip()
    return None
