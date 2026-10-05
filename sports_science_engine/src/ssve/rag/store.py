"""Vector stores: Qdrant for production, an in-memory store with the same interface for tests."""
from __future__ import annotations

import uuid
from typing import Protocol

import numpy as np

from ..models import Document, RetrievedChunk
from .embeddings import Embedder


class VectorStore(Protocol):
    def upsert(self, docs: list[Document]) -> int: ...

    def search(self, query: str, limit: int = 5, kind: str | None = None,
               require_doi: bool = False) -> list[RetrievedChunk]: ...


def _point_id(doc: Document) -> str:
    # Qdrant accepts UUIDs; derive one deterministically so re-ingestion is idempotent.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, doc.id))


def _embed_text(doc: Document) -> str:
    return f"{doc.title}\n{doc.text}"


class InMemoryStore:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._docs: dict[str, Document] = {}
        self._vecs: dict[str, np.ndarray] = {}

    def upsert(self, docs: list[Document]) -> int:
        if not docs:
            return 0
        vectors = self.embedder.embed([_embed_text(d) for d in docs])
        for doc, vec in zip(docs, vectors):
            pid = _point_id(doc)
            self._docs[pid] = doc
            self._vecs[pid] = vec
        return len(docs)

    def search(self, query: str, limit: int = 5, kind: str | None = None,
               require_doi: bool = False) -> list[RetrievedChunk]:
        candidates = [pid for pid, d in self._docs.items()
                      if (kind is None or d.kind == kind) and (not require_doi or d.doi)]
        if not candidates:
            return []
        q = self.embedder.embed([query])[0]
        scores = np.stack([self._vecs[p] for p in candidates]) @ q
        order = np.argsort(-scores)[:limit]
        return [RetrievedChunk(self._docs[candidates[i]], float(scores[i])) for i in order]

    def __len__(self) -> int:
        return len(self._docs)


class QdrantStore:
    def __init__(self, embedder: Embedder, collection: str = "sports_science",
                 url: str | None = "http://localhost:6333", api_key: str | None = None,
                 client=None) -> None:
        from qdrant_client import QdrantClient, models  # optional dependency

        self.models = models
        self.embedder = embedder
        self.collection = collection
        self.client = client or QdrantClient(url=url, api_key=api_key)
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=embedder.dim,
                                                   distance=models.Distance.COSINE),
            )
            for key in ("kind", "doi"):
                self.client.create_payload_index(collection, key,
                                                 field_schema=models.PayloadSchemaType.KEYWORD)

    def upsert(self, docs: list[Document]) -> int:
        if not docs:
            return 0
        vectors = self.embedder.embed([_embed_text(d) for d in docs])
        points = [self.models.PointStruct(id=_point_id(d), vector=v.tolist(), payload=d.to_payload())
                  for d, v in zip(docs, vectors)]
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def search(self, query: str, limit: int = 5, kind: str | None = None,
               require_doi: bool = False) -> list[RetrievedChunk]:
        m = self.models
        must = []
        if kind:
            must.append(m.FieldCondition(key="kind", match=m.MatchValue(value=kind)))
        must_not = [m.IsNullCondition(is_null=m.PayloadField(key="doi"))] if require_doi else []
        flt = m.Filter(must=must, must_not=must_not) if (must or must_not) else None
        vector = self.embedder.embed([query])[0].tolist()
        hits = self.client.query_points(collection_name=self.collection, query=vector,
                                        query_filter=flt, limit=limit, with_payload=True).points
        return [RetrievedChunk(Document.from_payload(h.payload), h.score) for h in hits]
