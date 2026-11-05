"""Formato de los ficheros de evals (JSON Lines: un objeto JSON por línea, fácil de revisar
y de comparar en un diff de git)."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from amr_agent.ingest.pubmed import Article

EVALS_DIR = Path("evals")
GOLDEN_PATH = EVALS_DIR / "golden.jsonl"
CORPUS_PATH = EVALS_DIR / "corpus.jsonl"
RESULTS_DIR = EVALS_DIR / "results"

QuestionType = Literal["factual", "gene_specific", "paraphrase", "out_of_scope"]


class GoldenItem(BaseModel):
    id: str
    question: str
    type: QuestionType
    # PMIDs que responden a la pregunta. Vacío solo en `out_of_scope`, donde lo correcto es
    # que el sistema se abstenga.
    relevant_pmids: list[str] = Field(default_factory=list)
    source_pmid: str | None = None  # abstract del que se generó la pregunta
    reviewed: bool = False  # True cuando una persona la ha revisado

    @property
    def in_scope(self) -> bool:
        return self.type != "out_of_scope"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenItem]:
    items = [GoldenItem(**row) for row in read_jsonl(path)]
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicated ids in {path}")
    for item in items:
        if item.in_scope and not item.relevant_pmids:
            raise ValueError(f"{item.id}: in-scope questions need relevant_pmids")
    return items


def load_corpus(path: Path = CORPUS_PATH) -> list[Article]:
    return [Article(**row) for row in read_jsonl(path)]
