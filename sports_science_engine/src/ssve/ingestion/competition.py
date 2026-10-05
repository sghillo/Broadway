"""Competition data ingestion from JSON or CSV endpoints/files.

Providers differ wildly, so each source is described by a field mapping rather than
bespoke code. Always check the provider's terms of use before scraping.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from ..models import Document
from .http import PoliteSession


@dataclass
class CompetitionSource:
    name: str
    location: str  # http(s) URL or local path
    fmt: str = "json"  # json | csv
    records_path: list[str] = field(default_factory=list)  # keys to the record list in JSON
    title_fields: list[str] = field(default_factory=lambda: ["event", "athlete"])
    date_field: str | None = "date"


class CompetitionIngestor:
    def __init__(self, session: PoliteSession | None = None) -> None:
        self.session = session or PoliteSession(min_interval=1.0)

    def _load(self, source: CompetitionSource) -> str:
        if source.location.startswith(("http://", "https://")):
            return self.session.get(source.location).text
        return Path(source.location).read_text(encoding="utf-8")

    def ingest(self, source: CompetitionSource) -> list[Document]:
        return parse_records(self._load(source), source)


def parse_records(raw: str, source: CompetitionSource) -> list[Document]:
    if source.fmt == "csv":
        records: list[dict[str, Any]] = list(csv.DictReader(io.StringIO(raw)))
    else:
        data: Any = json.loads(raw)
        for key in source.records_path:
            data = data[key]
        records = list(data)
    docs = []
    for rec in records:
        title = " - ".join(str(rec[f]) for f in source.title_fields if rec.get(f))
        published = None
        if source.date_field and rec.get(source.date_field):
            try:
                published = date.fromisoformat(str(rec[source.date_field])[:10])
            except ValueError:
                published = None
        text = "; ".join(f"{k}: {v}" for k, v in rec.items())
        docs.append(Document(source=f"competition:{source.name}", title=title or source.name,
                             text=text, published=published, kind="competition",
                             metadata={"record": rec}))
    return docs
