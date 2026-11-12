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
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

from amr_agent import telemetry
from amr_agent.llm import ChatLLM, ChatTurn, ToolCall
from amr_agent.quotes import RawQuote, locate, split_quotes
from amr_agent.rag import NO_EVIDENCE, NO_EVIDENCE_PAPERS, extract_citations, normalize_citations
from amr_agent.tools import (
    READ_PAPER,
    SEARCH_LITERATURE,
    SEARCH_PAPERS,
    Excerpt,
    Paper,
    ToolBox,
    ToolResult,
    drop_seen_abstracts,
)

LANGUAGE = "- Answer in the language of the question. Be concise: at most two short paragraphs."

# Modo Papers (decisión 41). gpt-oss contesta en español pero piensa en inglés, y se nota:
# "estéril de esteroides" (steroid-free), "mycophenolate mofetil" sin traducir, "indución",
# "AML" por LMA. Los ejemplos salen de esas respuestas; para medirlo hubo además dos preguntas de
# otros temas (docs/decisions.md).
NATIVE_LANGUAGE = """\
- Answer in the language of the question. If it is not English, write it as a native scientific
  writer of that language would, not as a translation of the English sources:
  - Every word in that language. Translate the names of drugs, procedures, diseases and
    outcomes to their standard form (in Spanish: "micofenolato de mofetilo", "globulina
    antitimocítica", "sin corticoides" for steroid-free, "supervivencia global").
  - Acronyms: for diseases and outcomes, the one used in that language, spelled out the first
    time (in Spanish: "leucemia mieloide aguda (LMA)", not AML). Keep as they are the names of
    genes, proteins, drugs, trials and products (IL-6, CD19, ZUMA-1, CAR-T).
  - Reread the answer before giving it: fix spelling and any English word or word order left.
- Be concise: at most two short paragraphs."""


def _rules(scope: str, extra: str = "", no_evidence: str = NO_EVIDENCE,
           language: str = LANGUAGE) -> str:
    """Reglas comunes a los dos modos; cambia el ámbito, el mensaje de abstención, la regla de
    idioma y AMR añade cómo citar el catálogo."""
    return f"""Rules:
- {scope} For anything else (writing
  code, general knowledge, news, weather...) do not call any tool and reply exactly:
  "{no_evidence}"
- Cite claims from articles as [PMID:12345678], only with PMIDs returned by the tools.
{extra}- Results often use other words than the question: a broader or narrower term, a synonym, an
  acronym. If a result answers the question in other words, answer with it and say how its terms
  match the question. You may use textbook knowledge for that link, never for the findings
  themselves: those come from the results. If the results answer only part of it, answer that
  part and say what is missing.
- Only if no tool result is about the question, reply exactly: "{no_evidence}"
{language}"""


def _final_prompt(role: str, rules: str) -> str:
    # Vuelta final (al llegar al límite de pasos o de tokens). No reenviamos la conversación con
    # tool_choice="none": gpt-oss, viendo las herramientas y su historial, sigue intentando
    # llamarlas (Groq responde 400 o la llamada se cuela en el texto; ver docs/decisions.md).
    # En su lugar, una conversación nueva, sin herramientas, con los resultados como texto.
    return f"""{role}
The research is finished: answer the user's question using only the tool results provided.
No tools are available now.

{rules}"""


# --- Modo AMR: catálogo de genes (amr-mcp) + nuestro índice de abstracts ---------------------

ROLE = "You are a research assistant specialised in antimicrobial resistance (AMR)."

RULES = _rules(
    "You only answer questions about antimicrobial resistance research.",
    '- Attribute database facts in the text, e.g. "according to the NCBI Reference Gene '
    'Catalog".\n')

SYSTEM_PROMPT = f"""{ROLE}
Answer the user's question with the tools below. Do not answer from memory.

Which tool:
- What a gene is, its drug class or the antibiotics it confers resistance to: the gene catalog
  (get_amr_gene, or search_amr_genes), not the literature.
- Which species carry a gene: get_amr_gene, then get_protein_hosts.
- Findings of studies (mechanisms, prevalence, treatments, experiments): search_literature.

Tools:
- search_literature: findings from recent PubMed abstracts (mechanisms, epidemiology, clinical
  evidence). Query it in English, whatever the language of the question. At most
  {{max_searches}} searches per question, and one is usually enough: search again only if the
  results do not answer the question, with different terms. It only lists abstracts you have
  not seen yet.
- search_amr_genes / get_amr_gene: what a gene is, from the NCBI Reference Gene Catalog (drug
  class, gene family, alleles, RefSeq protein accession).
- get_protein_hosts: species in which a protein has been sequenced. It needs the RefSeq protein
  accession (WP_...) from the catalog. Use it for "which species carry gene X": it counts all
  NCBI sequence records, while the literature only covers a few studies.
- get_pubmed_references: title and journal of PMIDs, e.g. those in a catalog entry.

{RULES}"""

FINAL_SYSTEM_PROMPT = _final_prompt(ROLE, RULES)

# --- Modo literatura: cualquier tema biomédico, búsqueda en vivo + texto completo -------------

LITERATURE_ROLE = "You are a research assistant for the biomedical literature."

# Modo Papers (decisión 42): la frase que respalda cada cita, para que se vea de dónde sale el
# dato sin abrir el artículo. El código la busca en lo que devolvieron las herramientas (quotes.py).
# gpt-oss, con el bloque, a veces sacaba las citas del texto (notas [^1] o una lista de PMIDs).
QUOTES = """\
- Put each citation in the text, right after its claim: no footnotes, no list of references.
- After the answer, write a line "Quotes:" and, for each article you cited, a line with the
  sentence from its abstract or passages that supports what you say, copied word for word in the
  language of the article. Only the quotes keep that language: the answer above them follows the
  language rule below, names of drugs included.
  [PMID:12345678] "the sentence, exactly as in the tool results"
  Do not translate or reword it; to shorten it, cut it with "…". The quotes do not replace the
  citations in the text. Each quote is checked against the tool results. If you reply that you
  found no evidence, write no quotes.
"""

LITERATURE_RULES = _rules(
    "You only answer questions about biomedical and life-science research.",
    QUOTES, no_evidence=NO_EVIDENCE_PAPERS, language=NATIVE_LANGUAGE)

LITERATURE_SYSTEM_PROMPT = f"""{LITERATURE_ROLE}
Answer the user's question with the tools below. Do not answer from memory.

How to work:
1. search_papers finds articles by title and abstract. Search for the study, not for the detail:
   the trial or therapy name, the disease, the population (e.g. "ELARA tisagenlecleucel
   follicular lymphoma"), because details such as doses or rates are usually only in the full
   text. If the question names no study, search for the topic. Query in English, as keywords,
   whatever the language of the question. At most {{max_searches}} searches per question, and
   one is usually enough: search again only if no result is about the question, with different
   terms. It only lists articles you have not seen yet.
2. If the question names no study (e.g. "what share of patients..." about a treatment in
   general), there is no single right article: answer from the abstracts. Give the figures of
   two or three studies with their setting (design, number of patients, follow-up), cite each
   one and say that results vary between studies. Read an article only if no abstract has a
   figure.
3. If the question asks for a detail of a study (a dose, a rate, a number of patients, a
   method) and the abstracts do not give it, call read_paper on the most relevant article with
   full_text true, with the question in English. Each article can be read once: if its passages
   do not have the answer, read another article or answer with what you have.
4. Answer with the specific fact, say which study it comes from (therapy, disease, design) and
   cite it. If the passages answer only part of the question or articles disagree, say so.

{LITERATURE_RULES}"""

LITERATURE_FINAL_SYSTEM_PROMPT = _final_prompt(LITERATURE_ROLE, LITERATURE_RULES)


@dataclass(frozen=True)
class AgentMode:
    """Lo que cambia entre modos: los prompts y qué herramienta de búsqueda se limita."""

    name: str
    system_prompt: str  # plantilla con {max_searches}
    final_system_prompt: str
    # Herramienta de búsqueda cuyas llamadas se cuentan (máx. max_searches) y cuyos resultados
    # ya vistos se quitan (decisiones 30 y 31).
    search_tool: str
    max_steps: int = 5
    # Herramienta que lee un artículo entero: cada artículo una sola vez por pregunta (decisión
    # 40). Releerlo con otra pregunta o más pasajes casi nunca trae nada nuevo y cuesta miles
    # de tokens que se reenvían en cada vuelta.
    read_tool: str | None = None
    no_evidence: str = NO_EVIDENCE
    quotes: bool = False  # la respuesta trae un bloque "Quotes:" que se comprueba (decisión 42)


AMR_MODE = AgentMode("amr", SYSTEM_PROMPT, FINAL_SYSTEM_PROMPT, SEARCH_LITERATURE.name)
# Un paso más que AMR: buscar, leer, a veces leer otro artículo y responder (con 5, una
# búsqueda de más dejaba sin paso para leer el segundo artículo).
LITERATURE_MODE = AgentMode("literature", LITERATURE_SYSTEM_PROMPT,
                            LITERATURE_FINAL_SYSTEM_PROMPT, SEARCH_PAPERS.name, max_steps=6,
                            read_tool=READ_PAPER.name, no_evidence=NO_EVIDENCE_PAPERS,
                            quotes=True)

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


def call_key(call: ToolCall) -> str | None:
    # sort_keys: {"a":1,"b":2} y {"b":2,"a":1} son la misma llamada.
    if call.arguments is None:
        return None
    return f"{call.name} {json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)}"


async def _search_limit(max_searches: int, tool: str) -> ToolResult:
    return ToolResult(f"ERROR: at most {max_searches} {tool} calls per question. "
                      "Answer with the abstracts above, or use another tool.", is_error=True)


def paper_key(call: ToolCall) -> str | None:
    """El artículo que pide read_paper, normalizado: "PMID: 123" y "123" son el mismo."""
    paper = str((call.arguments or {}).get("paper_id") or "").strip().upper()
    return re.sub(r"^PMID:?\s*", "", paper) or None


async def _already_read(paper: str, step: int) -> ToolResult:
    return ToolResult(f"ERROR: you already read {paper} in step {step}; its passages are above. "
                      "Each article can be read once: read another article or answer with what "
                      "you have.", is_error=True)


async def _repeated(call: ToolCall, step: int) -> ToolResult:
    # No se vuelve a ejecutar: costaría tiempo (NCBI) y tokens para obtener lo mismo.
    return ToolResult(f"ERROR: you already called {call.name} with these arguments in step "
                      f"{step}; its result is above. Use it, or try different arguments or "
                      "another tool.", is_error=True)


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


class Quote(BaseModel):
    """Frase del artículo que el modelo da como respaldo de una cita (modo Papers)."""

    text: str
    verified: bool  # aparece en el abstract o en los pasajes que devolvieron las herramientas
    section: str | None = None  # dónde: "Abstract", "Results"...; None si no aparece


class Citation(BaseModel):
    pmid: str
    url: str
    # Solo si alguna herramienta trajo el artículo (search_literature, get_pubmed_references);
    # un PMID que solo sale del catálogo de genes no tiene título.
    title: str | None = None
    journal: str | None = None
    year: int | None = None
    quotes: list[Quote] = []


def _citation(pmid: str, paper: Paper | None, quotes: list[Quote]) -> Citation:
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    if paper is None:
        return Citation(pmid=pmid, url=url, quotes=quotes)
    return Citation(pmid=pmid, url=url, title=paper.title, journal=paper.journal, year=paper.year,
                    quotes=quotes)


def _check_quotes(raw: list[RawQuote], excerpts: list[Excerpt]) -> dict[str, list[Quote]]:
    by_pmid: dict[str, list[Quote]] = {}
    for quote in raw:
        found = locate(quote, excerpts)
        by_pmid.setdefault(quote.pmid, []).append(Quote(
            text=quote.text, verified=found is not None, section=found.section if found else None))
    return by_pmid


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
    def __init__(self, llm: ChatLLM, tools: ToolBox, *, mode: AgentMode = AMR_MODE,
                 max_steps: int | None = None, token_budget: int = 12_000,
                 max_calls_per_step: int = 4,
                 max_literature_searches: int = 2) -> None:
        # Dos frenos para no agotar la cuota de Groq (8.000 tokens/min) con una sola pregunta:
        # un máximo de vueltas y un presupuesto de tokens. Al llegar a cualquiera de los dos,
        # se pide la respuesta final sin herramientas en vez de cortar sin responder.
        self._llm = llm
        self._tools = tools
        self.mode = mode
        self.max_steps = max_steps or mode.max_steps
        self.token_budget = token_budget
        self.max_calls_per_step = max_calls_per_step
        # En dos ejecuciones de las evals, el abstract relevante salió en la 1.ª o 2.ª búsqueda
        # en 11 de 12 casos; la 3.ª y la 4.ª casi nunca aportaban y llevaban al límite de pasos
        # (decisión 31). gpt-oss no deja de buscar aunque se lo pida el prompt: lo cortamos aquí.
        self.max_literature_searches = max_literature_searches

    async def run(self, question: str) -> AgentAnswer:
        """Responde a la pregunta y deja la trayectoria en una traza (telemetry.py)."""
        with telemetry.agent_span(
                question, mode=self.mode.name,
                provider=getattr(self._llm, "provider", "unknown"),
                model=getattr(self._llm, "model", "unknown")) as span:
            result = await self._run(question)
            telemetry.record_agent(
                span, answer=result.answer, input_tokens=result.usage.prompt_tokens,
                output_tokens=result.usage.completion_tokens, llm_calls=result.usage.llm_calls,
                tool_calls=len(result.steps), stop_reason=result.stop_reason,
                invalid_citations=len(result.invalid_citations),
                quotes=[q for c in result.citations for q in c.quotes] if self.mode.quotes
                else None)
        return result

    async def _run(self, question: str) -> AgentAnswer:
        t0 = time.perf_counter()
        schemas = [s.to_openai() for s in self._tools.specs()]
        messages: list[dict[str, Any]] = [
            {"role": "system",
             "content": self.mode.system_prompt.format(
                 max_searches=self.max_literature_searches)},
            {"role": "user", "content": question},
        ]
        steps: list[ToolStep] = []
        evidence: list[tuple[ToolCall, ToolResult]] = []  # para la vuelta final
        called: dict[str, int] = {}  # llamada (herramienta + argumentos) -> paso en que se hizo
        shown_abstracts: set[str] = set()  # PMIDs que la búsqueda ya mostró
        read: dict[str, int] = {}  # artículo leído con éxito -> paso (decisión 40)
        # Búsquedas de literatura que devolvieron algo. No cuentan las repetidas, las rechazadas
        # (argumento inventado, fallo de red) ni las vacías: no han enseñado nada al modelo y,
        # si contaran, se quedaría sin búsquedas antes de ver un solo artículo (decisión 37).
        searches = 0
        seen_pmids: set[str] = set()
        papers: dict[str, Paper] = {}  # PMID -> título, revista y año, para las referencias
        excerpts: list[Excerpt] = []  # abstracts y pasajes: las frases citadas deben estar aquí
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
            pending: list[Awaitable[ToolResult]] = []
            counted: set[int] = set()  # índices de las búsquedas contadas en este paso
            for i, call in enumerate(turn.tool_calls):
                key = call_key(call)
                paper = paper_key(call) if call.name == self.mode.read_tool else None
                if key is not None and key in called:  # misma herramienta, mismos argumentos
                    pending.append(_repeated(call, called[key]))
                elif paper is not None and paper in read:
                    pending.append(_already_read(paper, read[paper]))
                elif (call.name == self.mode.search_tool
                      and searches >= self.max_literature_searches):
                    pending.append(_search_limit(self.max_literature_searches,
                                                 self.mode.search_tool))
                else:
                    if call.name == self.mode.search_tool:
                        searches += 1  # provisional: se descuenta abajo si no devolvió nada
                        counted.add(i)
                    if key is not None:
                        called[key] = step
                    pending.append(self._call(call, i))
            # Si pide varias herramientas a la vez, son independientes: en paralelo.
            results = await asyncio.gather(*(self._traced(call, p) for call, p
                                             in zip(turn.tool_calls, pending, strict=True)))
            tools_ms += (time.perf_counter() - t_tools) * 1000
            searches -= sum(1 for i in counted if results[i].is_error or not results[i].pmids)
            for call, result in zip(turn.tool_calls, results, strict=True):
                if call.name == self.mode.search_tool:
                    pmids = result.pmids
                    result = drop_seen_abstracts(result, shown_abstracts)
                    shown_abstracts.update(pmids)
                elif call.name == self.mode.read_tool and not result.is_error:
                    # Solo las lecturas que trajeron pasajes: si falló (sin texto completo,
                    # argumento mal escrito), volver a pedirlo bien no es releer.
                    paper = paper_key(call)
                    if paper is not None:
                        read.setdefault(paper, step)
                # La API exige un mensaje `tool` por cada llamada, también las fallidas.
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": result.content})
                evidence.append((call, result))
                seen_pmids |= result.pmids
                papers.update((p.pmid, p) for p in result.papers)
                excerpts.extend(result.excerpts)
                steps.append(ToolStep(
                    step=step, tool=call.name, arguments=call.arguments,
                    is_error=result.is_error, truncated=result.truncated,
                    result_chars=len(result.content), pmids=sorted(result.pmids),
                    latency_ms=result.latency_ms, preview=result.content[:300],
                ))

        # Guardarraíl, igual que en /ask: una cita solo vale si una herramienta devolvió ese PMID.
        answer = normalize_citations(strip_tool_markup(turn.text)).strip()
        # Y una frase solo vale si está en lo que devolvieron las herramientas para ese PMID.
        raw_quotes: list[RawQuote] = []
        if self.mode.quotes:
            answer, raw_quotes = split_quotes(answer)
        answer = answer or self.mode.no_evidence
        # Un artículo del que da una frase también está citado, aunque falte en el texto: a veces
        # gpt-oss pasaba todas las citas al bloque y dejaba notas [^1] (que split_quotes quita).
        cited = list(dict.fromkeys([*extract_citations(answer), *(q.pmid for q in raw_quotes)]))
        quotes = _check_quotes(raw_quotes, excerpts)
        return AgentAnswer(
            answer=answer,
            citations=[_citation(p, papers.get(p), quotes.get(p, [])) for p in cited
                       if p in seen_pmids],
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

    def _final_messages(self, question: str,
                        evidence: list[tuple[ToolCall, ToolResult]]) -> list[dict[str, Any]]:
        """La pregunta y los resultados como texto plano: sin herramientas que intentar llamar."""
        blocks = [f"[{i}] {call.name} {json.dumps(call.arguments, ensure_ascii=False)}\n"
                  f"{result.content}" for i, (call, result) in enumerate(evidence, start=1)]
        results = "\n\n".join(blocks) or "(no tool results)"
        return [
            {"role": "system", "content": self.mode.final_system_prompt},
            {"role": "user", "content": f"Question: {question}\n\nTool results:\n\n{results}"
                                        f"\n\n{FINAL_NUDGE}"},
        ]

    async def _traced(self, call: ToolCall, pending: Awaitable[ToolResult]) -> ToolResult:
        """Un span `execute_tool` por llamada, también las repetidas o rechazadas: son decisiones
        del modelo y en la traza deben verse. Se espera dentro del span para que lo que cree la
        herramienta (el span de MCP) quede colgando de él."""
        with telemetry.tool_span(call.name, call.id, call.arguments) as span:
            result = await pending
            telemetry.record_tool(span, is_error=result.is_error, error=result.content,
                                  pmids=len(result.pmids), truncated=result.truncated,
                                  result_chars=len(result.content))
        return result

    async def _call(self, call: ToolCall, index: int) -> ToolResult:
        if index >= self.max_calls_per_step:
            return ToolResult(f"ERROR: at most {self.max_calls_per_step} tool calls per step",
                              is_error=True)
        if call.arguments is None:
            return ToolResult("ERROR: arguments must be a JSON object", is_error=True)
        return await self._tools.call(call.name, call.arguments)
