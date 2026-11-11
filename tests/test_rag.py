from amr_agent.rag import NO_EVIDENCE, RagService, extract_citations, normalize_citations
from amr_agent.retrieval import rrf_fuse
from tests.fakes import FakeLLM, FakeRetriever, make_hit


def test_rrf_rewards_documents_ranked_well_in_both_lists():
    a, b, c = make_hit("a"), make_hit("b"), make_hit("c")
    fused = rrf_fuse([[a, b, c], [b, c, a]], top_n=3)
    # b: 1/62 + 1/61 > a: 1/61 + 1/63
    assert [h.pmid for h in fused] == ["b", "a", "c"]


def test_rrf_respects_top_n_and_handles_disjoint_lists():
    fused = rrf_fuse([[make_hit("a")], [make_hit("b")]], top_n=1)
    assert len(fused) == 1


def test_extract_citations_dedupes_in_order():
    text = "X [PMID:2] and Y [PMID: 1]. Again [PMID:2]."
    assert extract_citations(text) == ["2", "1"]


async def test_fullwidth_brackets_are_parsed_and_normalized():
    # Caso real: gpt-oss-120b citó con 【PMID:…】 en vez de [PMID:…].
    llm = FakeLLM("mcr-1 is dominant【PMID:1001】 and ［PMID：1002］.")
    rag = RagService(FakeRetriever([make_hit("1001"), make_hit("1002")]), llm)
    result = await rag.ask("colistin?")
    assert [c.pmid for c in result.citations] == ["1001", "1002"]
    assert result.answer == "mcr-1 is dominant[PMID:1001] and [PMID:1002]."


def test_browsing_style_and_parenthesised_citations_are_normalized():
    # Casos reales de gpt-oss en las evals del agente (multi-02 y cat-06).
    text = ("Strong synergy【3†PMID:42852341】, confirmed【4†PMID:41719126†L10-L12】. "
            "Seen in K. pneumoniae (≈ 47 %)【2】 and E. coli【3†L1-L5】; "
            "resistant (PMID 41847190).")
    assert normalize_citations(text) == (
        "Strong synergy[PMID:42852341], confirmed[PMID:41719126]. "
        "Seen in K. pneumoniae (≈ 47 %) and E. coli; resistant [PMID:41847190].")
    assert extract_citations(text) == ["42852341", "41719126", "41847190"]


def test_several_pmids_in_one_bracket_become_separate_citations():
    # cat-05: "【PMID:11470367, PMID:37255469】"
    text = "KPC-2 was described in 2001【PMID:11470367, PMID:37255469】 and later [PMID:5; 6]."
    assert normalize_citations(text) == (
        "KPC-2 was described in 2001[PMID:11470367][PMID:37255469] and later [PMID:5][PMID:6].")
    assert extract_citations(text) == ["11470367", "37255469", "5", "6"]


def test_browsing_positions_in_json_are_removed():
    # /agent en Render, 2026-11-11: la posición del resultado en vez de un PMID.
    text = ('Carbapenems【{"cursor":0,"loc":0}】 and '
            'K. pneumoniae【{"cursor": 2, "loc": 14}】 [PMID:1].')
    assert normalize_citations(text) == "Carbapenems and K. pneumoniae [PMID:1]."


def test_normalize_leaves_ordinary_brackets_and_parentheses_alone():
    text = "Resistance rose [2021 data] (n = 12) to 47 % [PMID:1] (PMID-free note)."
    assert normalize_citations(text) == text


async def test_answer_keeps_valid_citations_and_flags_hallucinated_ones():
    llm = FakeLLM("mcr-1 confers colistin resistance [PMID:1001]. Also [PMID:9999].")
    rag = RagService(FakeRetriever([make_hit("1001"), make_hit("1002")]), llm)

    result = await rag.ask("colistin?")

    assert [c.pmid for c in result.citations] == ["1001"]
    assert result.invalid_citations == ["9999"]
    assert len(result.sources) == 2
    assert result.citations[0].url == "https://pubmed.ncbi.nlm.nih.gov/1001/"
    assert "[PMID:1002]" in llm.calls[0][1]  # las fuentes llegan al prompt


async def test_no_hits_skips_the_llm():
    llm = FakeLLM("should not be called")
    result = await RagService(FakeRetriever([]), llm).ask("anything?")
    assert result.answer == NO_EVIDENCE
    assert llm.calls == []


async def test_k_and_mode_are_forwarded_to_the_retriever():
    retriever = FakeRetriever([make_hit("1")])
    await RagService(retriever, FakeLLM("ok [PMID:1]")).ask("q?", k=3, mode="keyword")
    assert retriever.calls == [("q?", 3, "keyword")]


def test_rrf_weights_let_one_ranking_dominate():
    a, b = make_hit("a"), make_hit("b")
    # a gana en la primera lista y b en la segunda; con peso 0.5 en la segunda, gana a.
    fused = rrf_fuse([[a, b], [b, a]], top_n=2, weights=[1.0, 0.5])
    assert [h.pmid for h in fused] == ["a", "b"]
