"""Shared data models passed between agents."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

DOI_PATTERN = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>\]]+)", re.IGNORECASE)


def normalise_doi(raw: str | None) -> str | None:
    """Return a canonical lower-case DOI (no URL prefix, no trailing punctuation)."""
    if not raw:
        return None
    match = DOI_PATTERN.search(raw.strip())
    if not match:
        return None
    doi = match.group(1).rstrip(".,;:")
    # Parentheses are legal inside DOIs (e.g. 10.1016/S0140-6736(20)30183-5), so only trim
    # a trailing ")" when it is unbalanced, i.e. it closes surrounding prose.
    while doi.endswith(")") and doi.count(")") > doi.count("("):
        doi = doi[:-1].rstrip(".,;:")
    return doi.lower()


@dataclass
class Document:
    """A unit of ingested content: a paper, a regulatory notice or a competition record."""

    source: str  # "pubmed", "rss:<feed>", "competition:<provider>"
    title: str
    text: str
    url: str | None = None
    doi: str | None = None
    published: date | None = None
    kind: str = "literature"  # literature | regulatory | competition
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.doi = normalise_doi(self.doi)

    @property
    def id(self) -> str:
        """Stable ID: DOI when available, else a hash of source + title + url."""
        key = self.doi or f"{self.source}|{self.title}|{self.url or ''}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()

    def to_payload(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "title": self.title,
            "text": self.text,
            "url": self.url,
            "doi": self.doi,
            "published": self.published.isoformat() if self.published else None,
            "kind": self.kind,
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Document":
        published = payload.get("published")
        return cls(
            source=payload["source"],
            title=payload["title"],
            text=payload["text"],
            url=payload.get("url"),
            doi=payload.get("doi"),
            published=date.fromisoformat(published) if published else None,
            kind=payload.get("kind", "literature"),
            metadata=payload.get("metadata") or {},
        )


@dataclass
class RetrievedChunk:
    document: Document
    score: float


@dataclass
class Answer:
    question: str
    text: str
    citations: list[str]  # DOIs actually cited in text and verified against retrieval
    rejected_citations: list[str] = field(default_factory=list)
    grounded: bool = True
    generated_at: datetime = field(default_factory=datetime.utcnow)
