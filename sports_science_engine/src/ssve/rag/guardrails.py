"""Strict DOI citation guardrails.

Rules enforced on every generated answer:
1. Every factual sentence must carry at least one ``[doi:...]`` citation.
2. Every cited DOI must belong to a document that was actually retrieved for this
   question (no hallucinated or "remembered" references).
3. Uncited or unverifiable sentences are removed; if nothing survives, the answer is
   replaced with an explicit "insufficient evidence" message.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models import normalise_doi

CITATION = re.compile(r"\[doi:\s*(10\.[^\]\s]+)\s*\]", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[\"'(*-])")
INSUFFICIENT = "Insufficient peer-reviewed evidence in the indexed corpus to answer this question."

# Lines that are structure, not claims (headings, separators, empty lines).
_NON_CLAIM = re.compile(r"^\s*(#{1,6}\s|[-=*_]{3,}\s*$|\s*$)")


@dataclass
class GuardrailResult:
    text: str
    cited: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    dropped_sentences: list[str] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        return bool(self.cited) and self.text != INSUFFICIENT


def extract_citations(text: str) -> list[str]:
    return [d for d in (normalise_doi(m) for m in CITATION.findall(text)) if d]


def enforce(text: str, allowed_dois: set[str]) -> GuardrailResult:
    allowed = {d for d in (normalise_doi(x) for x in allowed_dois) if d}
    result = GuardrailResult(text="")
    kept_lines: list[str] = []
    for line in text.splitlines():
        if _NON_CLAIM.match(line):
            kept_lines.append(line)
            continue
        kept_sentences = []
        for sentence in _SENTENCE_SPLIT.split(line):
            dois = extract_citations(sentence)
            bad = [d for d in dois if d not in allowed]
            result.rejected.extend(b for b in bad if b not in result.rejected)
            if dois and not bad:
                kept_sentences.append(sentence)
                result.cited.extend(d for d in dois if d not in result.cited)
            else:
                result.dropped_sentences.append(sentence)
        if kept_sentences:
            kept_lines.append(" ".join(kept_sentences))
    body = "\n".join(kept_lines).strip()
    # Only headings survived -> nothing evidential left.
    has_claims = any(not _NON_CLAIM.match(line) for line in body.splitlines())
    result.text = body if has_claims else INSUFFICIENT
    if not has_claims:
        result.cited = []
    return result
