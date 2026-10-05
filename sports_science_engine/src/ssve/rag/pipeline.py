"""Retrieval-augmented answering with enforced DOI citations."""
from __future__ import annotations

from ..models import Answer, Document, RetrievedChunk
from .guardrails import enforce
from .llm import LLM
from .store import VectorStore

SYSTEM_PROMPT = """You are a sports-science research analyst.
Answer ONLY from the provided context blocks. Each block starts with [doi:<DOI>].
Rules:
- Every sentence that states a fact MUST end with one or more citations in the exact form [doi:<DOI>].
- Only cite DOIs that appear in the context. Never invent or recall references.
- If the context does not support an answer, reply exactly: INSUFFICIENT EVIDENCE.
- Be concise and quantitative where the context allows (effect sizes, n, population)."""


def format_context(chunks: list[RetrievedChunk], max_chars: int = 1500) -> str:
    blocks = []
    for c in chunks:
        d = c.document
        blocks.append(f"[doi:{d.doi}] {d.title}\n{d.text[:max_chars]}")
    return "\n\n".join(blocks)


class RAGPipeline:
    def __init__(self, store: VectorStore, llm: LLM, top_k: int = 6) -> None:
        self.store = store
        self.llm = llm
        self.top_k = top_k

    def index(self, docs: list[Document]) -> int:
        return self.store.upsert(docs)

    def retrieve(self, question: str, kind: str | None = "literature") -> list[RetrievedChunk]:
        # Only DOI-bearing documents are citable, so only they enter the context window.
        return self.store.search(question, limit=self.top_k, kind=kind, require_doi=True)

    def answer(self, question: str, kind: str | None = "literature") -> Answer:
        chunks = self.retrieve(question, kind=kind)
        if not chunks:
            return Answer(question, enforce("", set()).text, [], grounded=False)
        context = format_context(chunks)
        raw = self.llm.complete(SYSTEM_PROMPT, f"{context}\n\nQuestion: {question}")
        checked = enforce(raw, {c.document.doi for c in chunks if c.document.doi})
        return Answer(question=question, text=checked.text, citations=checked.cited,
                      rejected_citations=checked.rejected, grounded=checked.grounded)
