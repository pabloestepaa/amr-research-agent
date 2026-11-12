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


# --- Evals del agente (trayectoria) ----------------------------------------------------------

AGENT_GOLDEN_PATH = EVALS_DIR / "golden_agent.jsonl"
CASSETTE_PATH = EVALS_DIR / "agent_cassette.jsonl"
# Modo literatura (decisión 36): preguntas sobre datos que solo están en el texto completo.
LITERATURE_GOLDEN_PATH = EVALS_DIR / "golden_literature.jsonl"
# Fuera de git: guarda pasajes de artículos y no todos tienen licencia que permita redistribuirlos.
LITERATURE_CASSETTE_PATH = EVALS_DIR / "literature_cassette.jsonl"

TaskCategory = Literal[
    "catalog", "hosts", "references", "literature", "multi_tool", "out_of_scope", "full_text",
    "general"]  # general: modo Papers, la pregunta no nombra un estudio (decisión 40)


class AgentTask(BaseModel):
    """Una tarea para el agente y lo que se espera de su trayectoria.

    Alternativas con "|": "get_amr_gene|search_amr_genes" se cumple si llama a cualquiera de las
    dos, y "Klebsiella pneumoniae|K. pneumoniae" si la respuesta contiene cualquiera de los dos.
    Todas las comparaciones de texto ignoran mayúsculas.
    """

    id: str
    question: str
    category: TaskCategory
    # Herramientas que debe llamar (al menos una vez cada grupo).
    expect_tools: list[str] = Field(default_factory=list)
    # Grupo de herramientas -> texto que debe aparecer en los argumentos de alguna de sus
    # llamadas. Ej.: {"get_protein_hosts": "WP_004199234"} comprueba que usó la accesión correcta.
    expect_args: dict[str, str] = Field(default_factory=dict)
    # Textos que debe contener la respuesta (hechos del catálogo y de NCBI, verificables).
    answer_contains: list[str] = Field(default_factory=list)
    # Artículos que responden a la pregunta: se mira si alguna búsqueda los encontró (y, en el
    # modo literatura, si read_paper los leyó).
    relevant_pmids: list[str] = Field(default_factory=list)
    should_abstain: bool = False
    reviewed: bool = False


def load_agent_golden(path: Path = AGENT_GOLDEN_PATH) -> list[AgentTask]:
    tasks = [AgentTask(**row) for row in read_jsonl(path)]
    ids = [t.id for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicated ids in {path}")
    for t in tasks:
        if t.should_abstain and (t.expect_tools or t.answer_contains):
            raise ValueError(f"{t.id}: an abstention task cannot expect tools or answer text")
        if not t.should_abstain and not t.expect_tools:
            raise ValueError(f"{t.id}: expect at least one tool")
        unknown = set(t.expect_args) - set(t.expect_tools)
        if unknown:
            raise ValueError(f"{t.id}: expect_args for tools not in expect_tools: {unknown}")
    return tasks
