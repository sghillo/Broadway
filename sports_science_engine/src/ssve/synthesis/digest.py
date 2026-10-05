"""Weekly executive digest generation (Markdown)."""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import date, timedelta

from ..models import Answer, Document
from ..rag.pipeline import RAGPipeline


def _doi_link(doi: str) -> str:
    return f"[{doi}](https://doi.org/{doi})"


def _cite_links(text: str) -> str:
    """Turn ``[doi:10.x/y]`` markers into clickable doi.org links for readers."""
    return re.sub(r"\[doi:\s*(10\.[^\]\s]+)\s*\]", r"([doi](https://doi.org/\1))", text)


def _in_window(doc: Document, start: date, end: date) -> bool:
    return doc.published is None or start <= doc.published <= end


def build_weekly_digest(docs: list[Document], pipeline: RAGPipeline | None = None,
                        themes: list[str] | None = None, week_ending: date | None = None,
                        max_items: int = 10) -> str:
    end = week_ending or date.today()
    start = end - timedelta(days=6)
    recent = [d for d in docs if _in_window(d, start, end)]
    by_kind: dict[str, list[Document]] = defaultdict(list)
    for d in recent:
        by_kind[d.kind].append(d)

    lit = sorted(by_kind["literature"], key=lambda d: d.published or end, reverse=True)
    out = [
        f"# Sports Science Weekly Digest — w/e {end:%d %b %Y}",
        "",
        "## Snapshot",
        f"- **{len(lit)}** new papers, **{len(by_kind['regulatory'])}** regulatory updates, "
        f"**{len(by_kind['competition'])}** competition records ingested ({start:%d %b}–{end:%d %b}).",
        "",
    ]

    if pipeline and themes:
        out += ["## Evidence Briefs", ""]
        for theme in themes:
            ans: Answer = pipeline.answer(theme)
            out += [f"### {theme}", _cite_links(ans.text), ""]
            if ans.rejected_citations:
                out += [f"> Guardrail removed {len(ans.rejected_citations)} unverifiable citation(s).", ""]

    out += ["## New Literature", ""]
    if lit:
        out += ["| Published | Title | Journal | DOI |", "|---|---|---|---|"]
        for d in lit[:max_items]:
            journal = d.metadata.get("journal", "")
            doi = _doi_link(d.doi) if d.doi else "—"
            title = d.title.replace("|", "\\|")
            out.append(f"| {d.published or '—'} | {title} | {journal} | {doi} |")
    else:
        out.append("_No new literature this week._")
    out.append("")

    out += ["## Regulatory Watch", ""]
    reg = by_kind["regulatory"]
    out += [f"- **{d.title}** ({d.source.removeprefix('rss:')}, {d.published or 'undated'})"
            + (f" — {d.url}" if d.url else "") for d in reg[:max_items]] or ["_No regulatory updates._"]
    out.append("")

    out += ["## Competition Highlights", ""]
    comp = by_kind["competition"]
    out += [f"- {d.title} ({d.published or 'undated'})" for d in comp[:max_items]] \
        or ["_No competition data this week._"]
    out.append("")
    return "\n".join(out)
