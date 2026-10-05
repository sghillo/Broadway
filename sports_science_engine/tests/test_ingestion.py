from datetime import date

from ssve.ingestion import CompetitionSource, parse_feed, parse_pubmed_xml, parse_records

PUBMED_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>40000001</PMID>
      <Article>
        <Journal><Title>Journal of Sports Sciences</Title>
          <JournalIssue><PubDate><Year>2026</Year><Month>Sep</Month></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>Heat acclimation in <i>elite</i> rowers</ArticleTitle>
        <Abstract>
          <AbstractText Label="METHODS">Ten days of heat training.</AbstractText>
          <AbstractText Label="RESULTS">Plasma volume rose 6%.</AbstractText>
        </Abstract>
        <AuthorList><Author><LastName>Smith</LastName><Initials>J</Initials></Author></AuthorList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">40000001</ArticleId>
        <ArticleId IdType="doi">10.1080/02640414.2026.123</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>"""

RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>Reg</title>
  <item><title>New rule</title><link>https://example.org/a</link>
    <description>Rule change effective 2027.</description>
    <pubDate>Fri, 02 Oct 2026 10:00:00 GMT</pubDate></item>
  <item><title>Old rule</title><link>https://example.org/b</link>
    <pubDate>Mon, 01 Jan 2024 10:00:00 GMT</pubDate></item>
</channel></rss>"""


def test_parse_pubmed_extracts_doi_abstract_and_date():
    [doc] = parse_pubmed_xml(PUBMED_XML)
    assert doc.doi == "10.1080/02640414.2026.123"
    assert doc.title == "Heat acclimation in elite rowers"
    assert "RESULTS: Plasma volume rose 6%." in doc.text
    assert doc.published == date(2026, 9, 1)
    assert doc.metadata["pmid"] == "40000001"
    assert doc.metadata["authors"] == ["Smith J"]


def test_parse_feed_filters_by_since():
    docs = parse_feed(RSS, name="reg", since=date(2026, 9, 28))
    assert [d.title for d in docs] == ["New rule"]
    assert docs[0].source == "rss:reg" and docs[0].kind == "regulatory"


def test_parse_competition_csv():
    src = CompetitionSource(name="demo", location="-", fmt="csv",
                            title_fields=["event", "athlete"], date_field="date")
    docs = parse_records("event,athlete,result,date\n100m,A,9.98,2026-10-04\n", src)
    assert docs[0].title == "100m - A"
    assert docs[0].published == date(2026, 10, 4)
    assert docs[0].kind == "competition"


def test_parse_competition_json_nested():
    src = CompetitionSource(name="demo", location="-", records_path=["data", "results"],
                            title_fields=["event"], date_field=None)
    docs = parse_records('{"data": {"results": [{"event": "Long jump"}]}}', src)
    assert docs[0].title == "Long jump"
