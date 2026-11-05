"""Comprueba que los ficheros versionados de evals son coherentes entre sí."""

from amr_agent.evals.dataset import CORPUS_PATH, GOLDEN_PATH, load_corpus, load_golden


def test_golden_set_is_valid_and_its_relevant_pmids_are_in_the_snapshot():
    golden = load_golden(GOLDEN_PATH)
    corpus = {a.pmid for a in load_corpus(CORPUS_PATH)}
    assert len(golden) >= 40
    assert {i.type for i in golden} == {"factual", "gene_specific", "paraphrase", "out_of_scope"}
    missing = {p for i in golden for p in i.relevant_pmids} - corpus
    assert not missing, f"relevant PMIDs missing from {CORPUS_PATH}: {missing}"
