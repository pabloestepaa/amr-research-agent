"""RAG: recuperar abstracts relevantes y generar una respuesta citada a partir de ellos."""

import re
import time

from pydantic import BaseModel

from amr_agent.llm import LLMClient
from amr_agent.retrieval import Hit, SearchBackend, SearchMode

NO_EVIDENCE = "I could not find evidence for this in the indexed literature."

SYSTEM_PROMPT = f"""You are a research assistant specialised in antimicrobial resistance (AMR).
Answer the user's question using ONLY the sources provided below the question.

Rules:
- Support every claim with a citation in the exact form [PMID:12345678].
- Only cite PMIDs that appear in the sources.
- If the sources do not answer the question, reply exactly: "{NO_EVIDENCE}"
  Do not use outside knowledge.
- Be concise: at most two short paragraphs."""

# Los modelos no siempre respetan el formato pedido. Variantes vistas con gpt-oss:
#   【PMID:123】 o ［PMID:123］   corchetes de ancho completo
#   【3†PMID:123】                el formato de citas de su herramienta de navegación
#   (PMID 123)                   entre paréntesis
# Las aceptamos y las normalizamos a [PMID:123]: así la respuesta es consistente y, sobre todo,
# esas citas pasan por la validación (una cita que no se reconoce tampoco se comprueba).
CITATION_RE = re.compile(
    r"[\[【［]\s*(?:\d+\s*†\s*)?PMID\s*[:：]\s*(\d+)\s*(?:†[^\]】］]*)?[\]】］]"
    r"|\(\s*PMID\s*[:：]?\s*(\d+)\s*\)"
)
# Varios PMIDs en un mismo corchete (cat-05): 【PMID:11470367, PMID:37255469】 o [PMID:1; 2].
CITATION_LIST_RE = re.compile(
    r"[\[【［]\s*(PMID\s*[:：]?\s*\d+(?:\s*[,;]\s*(?:PMID\s*[:：]?\s*)?\d+)+)\s*[\]】］]"
)
# Referencias de navegación sin PMID, como 【2】 o 【3†L1-L5】: no le dicen nada al lector.
BARE_REF_RE = re.compile(r"[ \t]*【\s*\d+\s*(?:†[^】]*)?】")


def normalize_citations(text: str) -> str:
    text = CITATION_LIST_RE.sub(
        lambda m: "".join(f"[PMID:{p}]" for p in re.findall(r"\d+", m.group(1))), text)
    text = CITATION_RE.sub(lambda m: f"[PMID:{m.group(1) or m.group(2)}]", text)
    return BARE_REF_RE.sub("", text)


class Source(BaseModel):
    pmid: str
    title: str
    journal: str | None
    year: int | None
    score: float
    url: str

    @classmethod
    def from_hit(cls, hit: Hit) -> "Source":
        return cls(
            pmid=hit.pmid,
            title=hit.title,
            journal=hit.journal,
            year=hit.year,
            score=round(hit.score, 4),
            url=f"https://pubmed.ncbi.nlm.nih.gov/{hit.pmid}/",
        )


class Usage(BaseModel):
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None


class RagAnswer(BaseModel):
    answer: str
    citations: list[Source]  # fuentes recuperadas que la respuesta cita de verdad
    sources: list[Source]  # todo lo recuperado (útil para depurar y para las evals)
    invalid_citations: list[str]  # PMIDs citados que NO estaban en el contexto
    usage: Usage | None
    latency_ms: dict[str, float]


def format_sources(hits: list[Hit]) -> str:
    blocks = []
    for hit in hits:
        meta = ", ".join(str(x) for x in (hit.journal, hit.year) if x)
        blocks.append(f"[PMID:{hit.pmid}] {hit.title} ({meta})\n{hit.abstract}")
    return "\n\n---\n\n".join(blocks)


def extract_citations(text: str) -> list[str]:
    text = normalize_citations(text)  # para que cuenten también las listas de PMIDs
    pmids = (m.group(1) or m.group(2) for m in CITATION_RE.finditer(text))
    return list(dict.fromkeys(pmids))  # únicos y en orden de aparición


class RagService:
    def __init__(self, retriever: SearchBackend, llm: LLMClient, top_k: int = 5) -> None:
        self._retriever = retriever
        self._llm = llm
        self.top_k = top_k

    async def ask(self, question: str, k: int | None = None, mode: SearchMode = "hybrid"):
        t0 = time.perf_counter()
        hits = await self._retriever.search(question, k or self.top_k, mode)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        sources = [Source.from_hit(h) for h in hits]

        if not hits:  # sin contexto no llamamos al LLM: ahorra cuota y evita inventar
            return RagAnswer(answer=NO_EVIDENCE, citations=[], sources=[], invalid_citations=[],
                             usage=None, latency_ms={"retrieval": round(retrieval_ms, 1)})

        user = f"Question: {question}\n\nSources:\n\n{format_sources(hits)}"
        completion = await self._llm.complete(SYSTEM_PROMPT, user)

        # Guardarraíl: comprobamos que cada cita apunta a un documento recuperado.
        by_pmid = {s.pmid: s for s in sources}
        answer = normalize_citations(completion.text)
        cited = extract_citations(answer)
        return RagAnswer(
            answer=answer,
            citations=[by_pmid[p] for p in cited if p in by_pmid],
            sources=sources,
            invalid_citations=[p for p in cited if p not in by_pmid],
            usage=Usage(model=completion.model, prompt_tokens=completion.prompt_tokens,
                        completion_tokens=completion.completion_tokens),
            latency_ms={"retrieval": round(retrieval_ms, 1),
                        "generation": round(completion.latency_ms, 1)},
        )
