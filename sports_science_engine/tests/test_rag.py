from ssve.models import Document, normalise_doi
from ssve.rag import INSUFFICIENT, enforce, extract_citations


def test_normalise_doi_strips_url_and_case():
    assert normalise_doi("https://doi.org/10.1080/ABC.12.") == "10.1080/abc.12"
    assert normalise_doi("no doi here") is None


def test_normalise_doi_keeps_balanced_parentheses():
    assert normalise_doi("10.1016/S0140-6736(20)30183-5") == "10.1016/s0140-6736(20)30183-5"
    assert normalise_doi("(see 10.1000/xyz).") == "10.1000/xyz"


def test_document_id_is_stable_on_doi():
    a = Document(source="x", title="A", text="", doi="10.1000/abc")
    b = Document(source="y", title="B", text="", doi="https://doi.org/10.1000/ABC")
    assert a.id == b.id


def test_guardrail_keeps_only_verified_citations():
    text = ("Load spikes raise injury risk [doi:10.1000/a]. "
            "Creatine cures everything [doi:10.9999/fake]. "
            "Uncited claim here.")
    r = enforce(text, {"10.1000/a"})
    assert r.text == "Load spikes raise injury risk [doi:10.1000/a]."
    assert r.cited == ["10.1000/a"]
    assert r.rejected == ["10.9999/fake"]
    assert len(r.dropped_sentences) == 2
    assert r.grounded


def test_guardrail_returns_insufficient_when_nothing_survives():
    r = enforce("## Heading\nAll uncited.", {"10.1000/a"})
    assert r.text == INSUFFICIENT and not r.grounded


def test_extract_citations_handles_whitespace_and_case():
    assert extract_citations("x [DOI: 10.1000/AbC ] y") == ["10.1000/abc"]


def test_pipeline_answers_with_retrieved_dois_only(pipeline):
    ans = pipeline.answer("workload ratio injury risk football")
    assert ans.grounded
    assert ans.citations and all(d.startswith("10.") for d in ans.citations)
    assert "10.1136/bjsports-2024-000001" in ans.citations


def test_pipeline_never_retrieves_doi_less_docs(pipeline):
    chunks = pipeline.retrieve("WADA prohibited list", kind=None)
    assert all(c.document.doi for c in chunks)


def test_hallucinating_llm_is_blocked(corpus):
    from ssve.rag import HashingEmbedder, InMemoryStore, RAGPipeline

    class Liar:
        def complete(self, system, user, **_):
            return "Ice baths double VO2max [doi:10.5555/not-real]."

    p = RAGPipeline(InMemoryStore(HashingEmbedder()), Liar())
    p.index(corpus)
    ans = p.answer("recovery")
    assert ans.text == INSUFFICIENT
    assert ans.rejected_citations == ["10.5555/not-real"]
    assert not ans.grounded
