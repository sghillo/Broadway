"""RSS/Atom ingestion for regulatory feeds (e.g. WADA, governing bodies) and journal TOCs."""
from __future__ import annotations

import logging
from datetime import date

import feedparser

from ..models import Document, normalise_doi
from .http import PoliteSession

log = logging.getLogger(__name__)


class RSSIngestor:
    def __init__(self, session: PoliteSession | None = None) -> None:
        self.session = session or PoliteSession(min_interval=1.0)

    def ingest(self, name: str, url: str, kind: str = "regulatory",
               since: date | None = None) -> list[Document]:
        raw = self.session.get(url).content
        return parse_feed(raw, name=name, kind=kind, since=since)


def parse_feed(raw: bytes | str, name: str, kind: str = "regulatory",
               since: date | None = None) -> list[Document]:
    feed = feedparser.parse(raw)
    if feed.bozo and not feed.entries:
        log.warning("Feed %s could not be parsed: %s", name, feed.bozo_exception)
        return []
    docs = []
    for entry in feed.entries:
        published = None
        parsed = entry.get("published_parsed") or entry.get("updated_parsed")
        if parsed:
            published = date(parsed.tm_year, parsed.tm_mon, parsed.tm_mday)
        if since and published and published < since:
            continue
        summary = entry.get("summary", "") or entry.get("description", "")
        doi = (entry.get("prism_doi") or entry.get("dc_identifier")
               or normalise_doi(entry.get("link", "")) or normalise_doi(summary))
        docs.append(Document(
            source=f"rss:{name}",
            title=entry.get("title", "").strip(),
            text=summary.strip() or entry.get("title", ""),
            url=entry.get("link"),
            doi=doi,
            published=published,
            kind=kind,
        ))
    return docs
