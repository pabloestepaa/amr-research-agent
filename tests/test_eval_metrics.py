import math

import pytest

from amr_agent.evals.dataset import GoldenItem
from amr_agent.evals.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from amr_agent.evals.retrieval_eval import check_thresholds, evaluate
from tests.fakes import FakeRetriever, make_hit


def test_recall_at_k_counts_relevant_in_top_k():
    assert recall_at_k(["a", "b", "c"], {"b", "z"}, k=2) == 0.5
    assert recall_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0


def test_recall_without_relevant_documents_is_an_error():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), k=1)


def test_reciprocal_rank_uses_first_relevant_position():
    assert reciprocal_rank(["a", "b", "c"], {"c", "b"}) == 0.5
    assert reciprocal_rank(["a"], {"z"}) == 0.0


def test_ndcg_is_one_for_perfect_ranking_and_lower_otherwise():
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3) == 1.0
    # un único relevante en la posición 2: 1/log2(3)
    assert math.isclose(ndcg_at_k(["x", "a"], {"a"}, k=10), 1 / math.log2(3))


async def test_evaluate_aggregates_by_mode_and_type_and_skips_out_of_scope():
    items = [
        GoldenItem(id="q1", question="mcr-1?", type="gene_specific", relevant_pmids=["1"]),
        GoldenItem(id="q2", question="ndm?", type="factual", relevant_pmids=["3"]),
        GoldenItem(id="q3", question="malaria?", type="out_of_scope"),
    ]
    backend = FakeRetriever([make_hit("1"), make_hit("2"), make_hit("3")])
    report = await evaluate(backend, items, modes=("vector",))

    overall = report["modes"]["vector"]["overall"]
    assert report["n_questions"] == 2
    assert overall["recall@1"] == 0.5  # q1 acierta en la 1.ª posición, q2 no
    assert overall["mrr"] == round((1 + 1 / 3) / 2, 4)
    assert report["modes"]["vector"]["by_type"]["factual"]["n"] == 1


def test_check_thresholds_reports_regressions():
    report = {"modes": {"hybrid": {"overall": {"recall@5": 0.7, "mrr": 0.9}}}}
    failures = check_thresholds(report, {"hybrid": {"recall@5": 0.8, "mrr": 0.5}})
    assert failures == ["hybrid recall@5 = 0.7 < 0.8"]
