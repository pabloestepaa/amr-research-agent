"""Tests de integración contra Postgres + pgvector reales (en CI, un contenedor de servicio)."""

import pytest

pytestmark = pytest.mark.db


async def test_keyword_search_matches_exact_gene_names(retriever):
    hits = await retriever.search("mcr-1 plasmid", k=3, mode="keyword")
    assert hits[0].pmid == "1001"


async def test_vector_search_returns_k_results_ordered_by_similarity(retriever):
    hits = await retriever.search("carbapenems imipenem meropenem", k=3, mode="vector")
    assert hits[0].pmid == "1002"
    assert len(hits) == 3
    assert hits[0].score >= hits[1].score >= hits[2].score


async def test_hybrid_search(retriever):
    hits = await retriever.search("gyrA mutations ciprofloxacin", k=2, mode="hybrid")
    assert hits[0].pmid == "1003"
    assert len(hits) == 2


async def test_keyword_search_with_only_stopwords_returns_nothing(retriever):
    assert await retriever.search("the of and", k=3, mode="keyword") == []
