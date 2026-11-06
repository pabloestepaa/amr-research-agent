"""Agente con tool calling: el LLM decide qué herramientas usar, el bucle las ejecuta.

    pregunta ─► LLM ─┬─ pide herramientas ─► ejecutarlas ─► añadir resultados ─┐
                     │                                                         │
                     └─ responde ◄──────────────── (como mucho max_steps) ◄────┘

No usamos un framework de agentes (LangGraph, PydanticAI...): el bucle son unas decenas de
líneas y así se ve exactamente qué se envía al modelo en cada paso.
"""

import asyncio
import json
import re
import time
from typing import Any, Literal

from pydantic import BaseModel

from amr_agent.llm import ChatLLM, ChatTurn, ToolCall
from amr_agent.rag import NO_EVIDENCE, extract_citations, normalize_citations
from amr_agent.tools import ToolBox, ToolResult

ROLE = "You are a research assistant specialised in antimicrobial resistance (AMR)."

RULES = f"""Rules:
- Cite claims from articles as [PMID:12345678], only with PMIDs returned by the tools.
- Attribute database facts in the text, e.g. "according to the NCBI Reference Gene Catalog".
- If the tools return nothing relevant, reply exactly: "{NO_EVIDENCE}"
- Answer in the language of the question. Be concise: at most two short paragraphs."""

SYSTEM_PROMPT = f"""{ROLE}
Answer the user's question with the tools below. Do not answer from memory.

Tools:
- search_literature: findings from recent PubMed abstracts (mechanisms, epidemiology, clinical
  evidence). Query it in English, whatever the language of the question.
- search_amr_genes / get_amr_gene: what a gene is, from the NCBI Reference Gene Catalog (drug
  class, gene family, alleles, RefSeq protein accession).
- get_protein_hosts: species in which a protein has been sequenced. It needs the RefSeq protein
  accession (WP_...) from the catalog. Use it for "which species carry gene X": it counts all
  NCBI sequence records, while the literature only covers a few studies.
- get_pubmed_references: title and journal of PMIDs, e.g. those in a catalog entry.

{RULES}"""

# Vuelta final (al llegar al límite de pasos o de tokens). No reenviamos la conversación con
# tool_choice="none": gpt-oss, viendo las herramientas y su historial, sigue intentando llamarlas
# (Groq responde 400 o la llamada se cuela en el texto; ver docs/decisions.md). En su lugar, una
# conversación nueva, sin herramientas, con los resultados ya obtenidos como texto.
FINAL_SYSTEM_PROMPT = f"""{ROLE}
The research is finished: answer the user's question using only the tool results provided.
No tools are available now.

{RULES}"""

FINAL_NUDGE = "Answer the question now, using only these results and following the rules."

# Restos de una llamada a herramienta escrita como texto (formato "harmony" de gpt-oss), p. ej.
# 【to=functions.search_amr_genes <|constrain|>json {"query": "x"}】, o el nombre de una
# herramienta usado como cita: 【get_amr_gene】. Las citas válidas, 【PMID:x】, no encajan.
TOOL_MARKUP_RE = re.compile(r"【\s*(?:to=functions[^】]*|[a-z_]+)\s*】|<\|[a-z_]+\|>")


def strip_tool_markup(text: str) -> str:
    cleaned = TOOL_MARKUP_RE.sub("", text)
    if cleaned == text:
        return text  # el caso normal: no tocamos ni un espacio
    # Quitar la llamada deja huecos: "carbapenemase  [PMID:1]" o "resistance ."
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return re.sub(r"[ \t]+([.,;:)])", r"\1", cleaned)


StopReason = Literal["answer", "max_steps", "token_budget"]


class ToolStep(BaseModel):
    """Una llamada a una herramienta: la trayectoria es la lista de estos pasos."""

    step: int  # vuelta del bucle (varias llamadas en paralelo comparten número)
    tool: str
    arguments: dict[str, Any] | None
    is_error: bool
    truncated: bool
    result_chars: int
    pmids: list[str]
    latency_ms: float
    preview: str  # comienzo del resultado, para la demo y para depurar


class Citation(BaseModel):
    pmid: str
    url: str


class AgentUsage(BaseModel):
    model: str
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int


class AgentAnswer(BaseModel):
    answer: str
    citations: list[Citation]  # PMIDs citados que salieron de alguna herramienta
    invalid_citations: list[str]  # PMIDs citados que ninguna herramienta devolvió
    steps: list[ToolStep]
    stop_reason: StopReason
    usage: AgentUsage
    latency_ms: dict[str, float]


class Agent:
    def __init__(self, llm: ChatLLM, tools: ToolBox, *, max_steps: int = 5,
                 token_budget: int = 12_000, max_calls_per_step: int = 4) -> None:
        # Dos frenos para no agotar la cuota de Groq (8.000 tokens/min) con una sola pregunta:
        # un máximo de vueltas y un presupuesto de tokens. Al llegar a cualquiera de los dos,
        # se pide la respuesta final sin herramientas en vez de cortar sin responder.
        self._llm = llm
        self._tools = tools
        self.max_steps = max_steps
        self.token_budget = token_budget
        self.max_calls_per_step = max_calls_per_step

    async def run(self, question: str) -> AgentAnswer:
        t0 = time.perf_counter()
        schemas = [s.to_openai() for s in self._tools.specs()]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        steps: list[ToolStep] = []
        evidence: list[tuple[ToolCall, ToolResult]] = []  # para la vuelta final
        seen_pmids: set[str] = set()
        turns: list[ChatTurn] = []
        llm_ms = tools_ms = 0.0
        stop: StopReason = "answer"

        for step in range(1, self.max_steps + 1):
            used = sum((t.prompt_tokens or 0) + (t.completion_tokens or 0) for t in turns)
            if step == self.max_steps or used >= self.token_budget:
                stop = "max_steps" if step == self.max_steps else "token_budget"
                turn = await self._llm.chat(self._final_messages(question, evidence), [],
                                            tool_choice="none")
            else:
                turn = await self._llm.chat(messages, schemas)
            turns.append(turn)
            llm_ms += turn.latency_ms

            if not turn.tool_calls or stop != "answer":
                break  # respuesta final (si aun así pide herramientas, no seguimos)

            messages.append(turn.as_message())
            t_tools = time.perf_counter()
            # Si pide varias herramientas a la vez, son independientes: en paralelo.
            results = await asyncio.gather(*(self._call(c, i) for i, c in
                                             enumerate(turn.tool_calls)))
            tools_ms += (time.perf_counter() - t_tools) * 1000
            for call, result in zip(turn.tool_calls, results, strict=True):
                # La API exige un mensaje `tool` por cada llamada, también las fallidas.
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": result.content})
                evidence.append((call, result))
                seen_pmids |= result.pmids
                steps.append(ToolStep(
                    step=step, tool=call.name, arguments=call.arguments,
                    is_error=result.is_error, truncated=result.truncated,
                    result_chars=len(result.content), pmids=sorted(result.pmids),
                    latency_ms=result.latency_ms, preview=result.content[:300],
                ))

        # Guardarraíl, igual que en /ask: una cita solo vale si una herramienta devolvió ese PMID.
        answer = normalize_citations(strip_tool_markup(turn.text)).strip() or NO_EVIDENCE
        cited = extract_citations(answer)
        return AgentAnswer(
            answer=answer,
            citations=[Citation(pmid=p, url=f"https://pubmed.ncbi.nlm.nih.gov/{p}/")
                       for p in cited if p in seen_pmids],
            invalid_citations=[p for p in cited if p not in seen_pmids],
            steps=steps,
            stop_reason=stop,
            usage=AgentUsage(
                model=turns[-1].model,
                llm_calls=len(turns),
                prompt_tokens=sum(t.prompt_tokens or 0 for t in turns),
                completion_tokens=sum(t.completion_tokens or 0 for t in turns),
            ),
            latency_ms={"llm": round(llm_ms, 1), "tools": round(tools_ms, 1),
                        "total": round((time.perf_counter() - t0) * 1000, 1)},
        )

    @staticmethod
    def _final_messages(question: str,
                        evidence: list[tuple[ToolCall, ToolResult]]) -> list[dict[str, Any]]:
        """La pregunta y los resultados como texto plano: sin herramientas que intentar llamar."""
        blocks = [f"[{i}] {call.name} {json.dumps(call.arguments, ensure_ascii=False)}\n"
                  f"{result.content}" for i, (call, result) in enumerate(evidence, start=1)]
        results = "\n\n".join(blocks) or "(no tool results)"
        return [
            {"role": "system", "content": FINAL_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nTool results:\n\n{results}"
                                        f"\n\n{FINAL_NUDGE}"},
        ]

    async def _call(self, call: ToolCall, index: int) -> ToolResult:
        if index >= self.max_calls_per_step:
            return ToolResult(f"ERROR: at most {self.max_calls_per_step} tool calls per step",
                              is_error=True)
        if call.arguments is None:
            return ToolResult("ERROR: arguments must be a JSON object", is_error=True)
        return await self._tools.call(call.name, call.arguments)
