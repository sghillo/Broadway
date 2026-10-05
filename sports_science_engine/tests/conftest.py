from datetime import date

import pytest

from ssve.models import Document
from ssve.rag import ExtractiveLLM, HashingEmbedder, InMemoryStore, RAGPipeline


@pytest.fixture
def corpus() -> list[Document]:
    return [
        Document(source="pubmed", title="Acute:chronic workload ratio and injury in elite football",
                 text="Spikes in acute:chronic workload ratio above 1.5 were associated with higher "
                      "non-contact injury risk in elite football players. Gradual load progression "
                      "was protective.",
                 doi="10.1136/bjsports-2024-000001", published=date(2026, 10, 1),
                 metadata={"journal": "Br J Sports Med"}),
        Document(source="pubmed", title="Markerless pose estimation validity for sprint kinematics",
                 text="Markerless pose estimation showed knee angle errors below 5 degrees compared "
                      "with marker-based motion capture during sprinting.",
                 doi="https://doi.org/10.1080/02640414.2026.0002", published=date(2026, 10, 2),
                 metadata={"journal": "J Sports Sci"}),
        Document(source="rss:wada", title="2027 Prohibited List consultation opens",
                 text="WADA opened consultation on the 2027 Prohibited List.",
                 published=date(2026, 10, 3), kind="regulatory"),
        Document(source="competition:demo", title="100m - A. Athlete", text="event: 100m; result: 9.98",
                 published=date(2026, 10, 4), kind="competition"),
    ]


@pytest.fixture
def pipeline(corpus) -> RAGPipeline:
    p = RAGPipeline(InMemoryStore(HashingEmbedder()), ExtractiveLLM(), top_k=3)
    p.index(corpus)
    return p
