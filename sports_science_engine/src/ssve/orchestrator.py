"""Autonomous execution pipeline: wires the four manifest agents into one weekly run."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml

from .ingestion import CompetitionIngestor, CompetitionSource, PubMedClient, RSSIngestor
from .models import Document
from .rag import (ExtractiveLLM, HashingEmbedder, InMemoryStore, QdrantStore, RAGPipeline,
                  SentenceTransformerEmbedder, VLLMClient)
from .synthesis import Competitor, EmailSender, build_competitor_memo, build_weekly_digest

log = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class RunReport:
    ingested: int = 0
    indexed: int = 0
    errors: list[str] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)


class Orchestrator:
    def __init__(self, config: dict[str, Any], pipeline: RAGPipeline | None = None) -> None:
        self.config = config
        self.pipeline = pipeline or self._build_pipeline(config.get("rag", {}))

    # -- Agent 2 -------------------------------------------------------------
    @staticmethod
    def _build_pipeline(cfg: dict[str, Any]) -> RAGPipeline:
        try:
            embedder = SentenceTransformerEmbedder(cfg.get("embedding_model", "BAAI/bge-small-en-v1.5"))
        except ImportError:
            log.warning("sentence-transformers not installed; using HashingEmbedder")
            embedder = HashingEmbedder()
        store: Any = None
        if cfg.get("backend", "memory") == "qdrant":
            try:
                store = QdrantStore(embedder, collection=cfg.get("collection", "sports_science"),
                                    url=cfg.get("qdrant_url"))
            except ImportError:
                log.warning("qdrant-client not installed; falling back to InMemoryStore")
        store = store or InMemoryStore(embedder)
        llm = VLLMClient() if cfg.get("llm") == "vllm" else ExtractiveLLM()
        return RAGPipeline(store, llm, top_k=cfg.get("top_k", 6))

    # -- Agent 1 -------------------------------------------------------------
    def ingest(self, report: RunReport) -> list[Document]:
        cfg = self.config.get("ingestion", {})
        days = cfg.get("lookback_days", 7)
        since = date.today() - timedelta(days=days)
        docs: list[Document] = []

        pm = cfg.get("pubmed") or {}
        if pm.get("queries"):
            client = PubMedClient()
            for q in pm["queries"]:
                try:
                    docs += client.ingest(q, days=days, max_results=pm.get("max_results", 100))
                except Exception as exc:  # one bad source must not sink the run
                    report.errors.append(f"pubmed {q!r}: {exc}")

        rss = RSSIngestor()
        for feed in cfg.get("rss") or []:
            try:
                docs += rss.ingest(feed["name"], feed["url"], kind=feed.get("kind", "regulatory"),
                                   since=since)
            except Exception as exc:
                report.errors.append(f"rss {feed['name']}: {exc}")

        comp = CompetitionIngestor()
        for src in cfg.get("competition") or []:
            try:
                docs += comp.ingest(CompetitionSource(**src))
            except Exception as exc:
                report.errors.append(f"competition {src.get('name')}: {exc}")

        # De-duplicate (same DOI from PubMed and a journal RSS feed, etc.).
        unique = {d.id: d for d in docs}
        report.ingested = len(unique)
        return list(unique.values())

    # -- Agent 3 -------------------------------------------------------------
    def synthesise(self, docs: list[Document], report: RunReport, week_ending: date | None = None) -> None:
        cfg = self.config.get("synthesis", {})
        out_dir = Path(cfg.get("output_dir", "reports"))
        out_dir.mkdir(parents=True, exist_ok=True)
        end = week_ending or date.today()

        digest = build_weekly_digest(docs, self.pipeline, cfg.get("themes"), week_ending=end)
        digest_path = out_dir / f"digest_{end.isoformat()}.md"
        digest_path.write_text(digest, encoding="utf-8")
        report.outputs.append(digest_path)

        for c in cfg.get("competitors") or []:
            comp = Competitor(c["name"], c.get("aliases", []), c.get("focus_questions", []))
            memo = build_competitor_memo(comp, docs, self.pipeline, as_of=end)
            slug = "".join(ch if ch.isalnum() else "_" for ch in comp.name.lower())
            path = out_dir / f"memo_{slug}_{end.isoformat()}.md"
            path.write_text(memo, encoding="utf-8")
            report.outputs.append(path)

        email = cfg.get("email") or {}
        if email.get("to"):
            sender = EmailSender(dry_run=email.get("dry_run", True), outbox=out_dir / "outbox")
            sent = sender.send(email["to"], f"Sports Science Weekly Digest — {end:%d %b %Y}", digest)
            if sent:
                report.outputs.append(sent)

    def run_weekly(self, week_ending: date | None = None) -> RunReport:
        report = RunReport()
        docs = self.ingest(report)
        report.indexed = self.pipeline.index(docs)
        self.synthesise(docs, report, week_ending)
        return report
