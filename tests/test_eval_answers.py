import pytest

from amr_agent.evals.answers_eval import (
    RecordingRetriever,
    evaluate_item,
    parse_judgement,
    summarize,
)
from amr_agent.evals.dataset import GoldenItem
from amr_agent.rag import NO_EVIDENCE, RagService
from tests.fakes import FakeLLM, FakeRetriever, make_hit

JUDGE_REPLY = (
    '{"claims": [{"claim": "mcr-1 is on IncI2", "supported": true},'
    ' {"claim": "it came from Mars", "supported": false}], "relevance": 5}'
)


def test_parse_judgement_tolerates_text_around_the_json():
    j = parse_judgement(f"Sure! {JUDGE_REPLY} Hope it helps.")
    assert (j.supported, j.total, j.relevance) == (1, 2, 5)
    assert j.faithfulness == 0.5
    assert j.unsupported_claims == ["it came from Mars"]


def test_parse_judgement_rejects_out_of_range_relevance():
    with pytest.raises(ValueError):
        parse_judgement('{"claims": [], "relevance": 9}')


def _setup(answer: str):
    recorder = RecordingRetriever(FakeRetriever([make_hit("1001", abstract="mcr-1 on IncI2")]))
    return RagService(recorder, FakeLLM(answer)), recorder


async def test_in_scope_answer_is_judged_with_the_cited_abstracts():
    rag, recorder = _setup("mcr-1 sits on IncI2 [PMID:1001] and [PMID:999].")
    judge = FakeLLM(JUDGE_REPLY)
    item = GoldenItem(id="g", question="mcr-1?", type="gene_specific", relevant_pmids=["1001"])

    row = await evaluate_item(item, rag, recorder, judge, k=5, mode="hybrid")

    assert row["invalid_citations"] == ["999"]
    assert row["context_hit"] is True
    assert row["faithfulness"] == 0.5 and row["relevance"] == 1.0
    assert "mcr-1 on IncI2" in judge.calls[0][1]  # el juez ve el texto del abstract citado


async def test_abstention_is_not_sent_to_the_judge():
    rag, recorder = _setup(NO_EVIDENCE)
    judge = FakeLLM(JUDGE_REPLY)
    item = GoldenItem(id="o", question="malaria?", type="out_of_scope")

    row = await evaluate_item(item, rag, recorder, judge, k=5, mode="hybrid")

    assert row["abstained"] is True and row["faithfulness"] is None
    assert judge.calls == []


def test_summarize_separates_in_scope_and_out_of_scope():
    base = {"invalid_citations": [], "latency_ms": 10.0, "faithfulness": None,
            "relevance": None, "citations": [], "context_hit": None}
    rows = [
        base | {"type": "factual", "abstained": False, "citations": ["1"], "context_hit": True,
                "faithfulness": 1.0, "relevance": 0.75},
        base | {"type": "factual", "abstained": True, "context_hit": False},
        base | {"type": "out_of_scope", "abstained": True},
        base | {"type": "out_of_scope", "abstained": False, "invalid_citations": ["9"]},
    ]
    s = summarize(rows)
    assert s["faithfulness"] == 1.0 and s["relevance"] == 0.75
    assert s["false_abstention_rate"] == 0.5
    assert s["abstention_accuracy"] == 0.5
    assert s["citation_validity"] == 0.75
    assert s["context_hit_rate"] == 0.5


def test_only_the_daily_token_limit_stops_the_run():
    from amr_agent.evals.answers_eval import is_daily_limit

    assert is_daily_limit(Exception("Rate limit reached ... on tokens per day (TPD)"))
    assert not is_daily_limit(Exception("Rate limit reached ... on tokens per minute (TPM)"))
