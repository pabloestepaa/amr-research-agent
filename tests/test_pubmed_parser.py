from amr_agent.ingest.pubmed import parse_pubmed_xml

XML = """<?xml version="1.0" ?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">111</PMID>
      <Article>
        <Journal><Title>Nature Microbiology</Title>
          <JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue></Journal>
        <ArticleTitle>Resistance to <i>beta</i>-lactams</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">ESBLs are   widespread.</AbstractText>
          <AbstractText Label="RESULTS">CTX-M-15 dominated.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">222</PMID>
      <Article>
        <Journal><Title>J X</Title>
          <JournalIssue><PubDate><MedlineDate>2019 Nov-Dec</MedlineDate></PubDate>
          </JournalIssue></Journal>
        <ArticleTitle>Unstructured</ArticleTitle>
        <Abstract><AbstractText>Plain abstract.</AbstractText></Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">333</PMID>
      <Article><ArticleTitle>No abstract, should be skipped</ArticleTitle></Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


def test_parses_structured_abstract_and_inline_tags():
    first = parse_pubmed_xml(XML)[0]
    assert first.pmid == "111"
    assert first.title == "Resistance to beta-lactams"
    assert first.abstract == "Background: ESBLs are widespread.\nResults: CTX-M-15 dominated."
    assert first.journal == "Nature Microbiology"
    assert first.year == 2021


def test_year_from_medline_date():
    assert parse_pubmed_xml(XML)[1].year == 2019


def test_skips_articles_without_abstract():
    assert [a.pmid for a in parse_pubmed_xml(XML)] == ["111", "222"]
