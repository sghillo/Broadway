from datetime import date
from email import message_from_bytes

from ssve.models import Document
from ssve.synthesis import Competitor, EmailSender, build_competitor_memo, build_weekly_digest


def test_weekly_digest_sections_and_links(corpus, pipeline):
    md = build_weekly_digest(corpus, pipeline, themes=["workload and injury risk"],
                             week_ending=date(2026, 10, 5))
    assert md.startswith("# Sports Science Weekly Digest — w/e 05 Oct 2026")
    assert "**2** new papers, **1** regulatory updates, **1** competition records" in md
    for heading in ("## Evidence Briefs", "## New Literature", "## Regulatory Watch",
                    "## Competition Highlights"):
        assert heading in md
    assert "https://doi.org/10.1136/bjsports-2024-000001" in md
    assert "[doi:" not in md  # raw markers converted to links


def test_digest_excludes_items_outside_window(corpus):
    old = Document(source="pubmed", title="Ancient paper", text="x", doi="10.1/old",
                   published=date(2020, 1, 1))
    md = build_weekly_digest(corpus + [old], week_ending=date(2026, 10, 5))
    assert "Ancient paper" not in md


def test_competitor_memo_matches_aliases():
    docs = [Document(source="rss:x", title="Hawkeye launches skeletal tracking", text="",
                     kind="regulatory", url="https://example.org/h")]
    md = build_competitor_memo(Competitor("Hawk-Eye", aliases=["Hawkeye"]), docs,
                               as_of=date(2026, 10, 5))
    assert "1 ingested item(s) mention Hawk-Eye" in md
    assert "Hawkeye launches skeletal tracking" in md


def test_email_dry_run_writes_eml(tmp_path):
    sender = EmailSender(dry_run=True, outbox=tmp_path)
    path = sender.send(["team@example.org"], "Weekly Digest", "# Hello\n\n- item")
    msg = message_from_bytes(path.read_bytes())
    assert msg["To"] == "team@example.org"
    assert msg.is_multipart()
