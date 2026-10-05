"""PubMed ingestion via NCBI E-utilities (esearch -> efetch XML)."""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import date

from ..models import Document
from .http import PoliteSession

log = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


class PubMedClient:
    def __init__(self, session: PoliteSession | None = None, api_key: str | None = None,
                 email: str | None = None, tool: str = "ssve") -> None:
        self.session = session or PoliteSession(min_interval=0.11 if api_key else 0.34)
        self.base_params = {"tool": tool}
        if api_key:
            self.base_params["api_key"] = api_key
        if email:
            self.base_params["email"] = email

    def search(self, query: str, days: int = 7, max_results: int = 100) -> list[str]:
        """Return PMIDs matching `query` published in the last `days` days."""
        params = {**self.base_params, "db": "pubmed", "term": query, "retmode": "json",
                  "retmax": max_results, "datetype": "pdat", "reldate": days, "sort": "pub_date"}
        data = self.session.get(f"{EUTILS}/esearch.fcgi", params=params).json()
        return data.get("esearchresult", {}).get("idlist", [])

    def fetch(self, pmids: list[str], batch_size: int = 200) -> list[Document]:
        docs: list[Document] = []
        for start in range(0, len(pmids), batch_size):
            batch = pmids[start:start + batch_size]
            params = {**self.base_params, "db": "pubmed", "id": ",".join(batch), "retmode": "xml"}
            xml = self.session.get(f"{EUTILS}/efetch.fcgi", params=params).text
            docs.extend(parse_pubmed_xml(xml))
        return docs

    def ingest(self, query: str, days: int = 7, max_results: int = 100) -> list[Document]:
        pmids = self.search(query, days=days, max_results=max_results)
        log.info("PubMed query %r returned %d PMIDs", query, len(pmids))
        return self.fetch(pmids) if pmids else []


def _text(el: ET.Element | None) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def _pub_date(article: ET.Element) -> date | None:
    node = article.find(".//Article/ArticleDate")
    if node is None:
        node = article.find(".//JournalIssue/PubDate")
    if node is None:
        return None
    try:
        year = int(_text(node.find("Year")))
    except ValueError:
        return None
    month_raw = _text(node.find("Month")) or "1"
    month = int(month_raw) if month_raw.isdigit() else _MONTHS.get(month_raw[:3].lower(), 1)
    day_raw = _text(node.find("Day"))
    day = int(day_raw) if day_raw.isdigit() else 1
    return date(year, month, day)


def parse_pubmed_xml(xml: str) -> list[Document]:
    root = ET.fromstring(xml)
    docs = []
    for article in root.iter("PubmedArticle"):
        pmid = _text(article.find(".//MedlineCitation/PMID"))
        title = _text(article.find(".//ArticleTitle"))
        abstract_parts = []
        for part in article.findall(".//Abstract/AbstractText"):
            label = part.get("Label")
            body = _text(part)
            abstract_parts.append(f"{label}: {body}" if label else body)
        doi = None
        for aid in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi":
                doi = _text(aid)
                break
        if doi is None:
            eloc = article.find(".//ELocationID[@EIdType='doi']")
            doi = _text(eloc) or None
        journal = _text(article.find(".//Journal/Title"))
        authors = [
            f"{_text(a.find('LastName'))} {_text(a.find('Initials'))}".strip()
            for a in article.findall(".//AuthorList/Author") if a.find("LastName") is not None
        ]
        docs.append(Document(
            source="pubmed",
            title=title,
            text="\n".join(abstract_parts) or title,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
            doi=doi,
            published=_pub_date(article),
            kind="literature",
            metadata={"pmid": pmid, "journal": journal, "authors": authors},
        ))
    return docs
