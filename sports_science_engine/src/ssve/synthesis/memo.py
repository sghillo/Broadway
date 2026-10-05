"""Competitor intelligence memos."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from ..models import Document
from ..rag.pipeline import RAGPipeline


@dataclass
class Competitor:
    name: str
    aliases: list[str] = field(default_factory=list)
    focus_questions: list[str] = field(default_factory=list)

    def pattern(self) -> re.Pattern[str]:
        names = [re.escape(n) for n in [self.name, *self.aliases]]
        return re.compile(r"\b(" + "|".join(names) + r")\b", re.IGNORECASE)


def mentions(docs: list[Document], competitor: Competitor) -> list[Document]:
    pat = competitor.pattern()
    return [d for d in docs if pat.search(d.title) or pat.search(d.text)]


def build_competitor_memo(competitor: Competitor, docs: list[Document],
                          pipeline: RAGPipeline | None = None, as_of: date | None = None) -> str:
    as_of = as_of or date.today()
    hits = mentions(docs, competitor)
    out = [f"# Competitor Memo — {competitor.name}", f"_As of {as_of:%d %b %Y}_", "",
           "## Snapshot",
           f"{len(hits)} ingested item(s) mention {competitor.name} this period.", ""]

    out += ["## Signals", ""]
    if hits:
        for d in sorted(hits, key=lambda d: d.published or as_of, reverse=True)[:15]:
            ref = f"https://doi.org/{d.doi}" if d.doi else (d.url or d.source)
            out.append(f"- **{d.title}** ({d.kind}, {d.published or 'undated'}) — {ref}")
    else:
        out.append("_No new signals._")
    out.append("")

    if pipeline and competitor.focus_questions:
        out += ["## Evidence Check", ""]
        for q in competitor.focus_questions:
            ans = pipeline.answer(q)
            out += [f"**{q}**", "", ans.text, ""]

    out += ["## So What", "",
            "- _Analyst to complete: implications for our roadmap and positioning._", ""]
    return "\n".join(out)
