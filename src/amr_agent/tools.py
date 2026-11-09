"""Herramientas del agente: la búsqueda en nuestra literatura, la lectura del texto completo de
un artículo y las del servidor MCP `amr-mcp`.

El agente solo ve un `ToolBox`: una lista de esquemas (`specs`) y un método `call`. Le da igual
si la herramienta es una función local o vive en otro proceso detrás del protocolo MCP.
"""

import copy
import json
import os
import re
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import httpx
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from amr_agent.fulltext import Article, best_passages, parse_jats
from amr_agent.retrieval import SearchBackend

# Herramientas de amr-mcp que exponemos. Cuantas menos, mejor elige el modelo y menos tokens
# cuesta cada llamada (los esquemas se reenvían en cada vuelta del bucle). Quedan fuera:
# run_amrfinder (lee rutas de ficheros del servidor), fetch_sequence (un FASTA son miles de
# tokens que no ayudan a responder), y UniProt y catalog_info (no las piden las preguntas típicas).
MCP_TOOLS = ("search_amr_genes", "get_amr_gene", "get_protein_hosts", "get_pubmed_references")

# Campos que el modelo no necesita y solo gastan tokens. La accesión RefSeq de proteína (WP_...)
# sí se conserva: es la que pide get_protein_hosts.
DROP_KEYS = frozenset({
    "genbank_protein_accession", "genbank_nucleotide_accession", "refseq_nucleotide_accession",
    "scope", "subtype",
})

# Límite de caracteres de cada resultado (~1.500 tokens). Groq Free da 8.000 tokens por minuto
# y el historial se reenvía en cada paso, así que sin recorte una sola pregunta lo agota.
MAX_RESULT_CHARS = 6000
MAX_ABSTRACT_CHARS = 1500

PMID_KEYS = ("pmid", "pubmed_reference")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema de los argumentos

    def to_openai(self) -> dict[str, Any]:
        """Formato `tools` de la API de OpenAI (y de Groq, que la imita)."""
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": bounds_to_description(self.parameters)}}


BOUNDS = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")


def bounds_to_description(schema: dict[str, Any]) -> dict[str, Any]:
    """Pasa los límites numéricos de cada argumento del esquema a su descripción.

    Groq valida los argumentos contra el esquema y, si el modelo se sale (pidió k=10 con un
    máximo de 6 en la eval lit-02), devuelve un 400 que tumba la petición entera. Sin los
    límites en el esquema, el valor llega a nuestro lado: `search_literature` lo recorta y
    amr-mcp devuelve un error que el modelo lee y puede corregir. El modelo sigue viendo el
    rango, ahora en la descripción.
    """
    props = schema.get("properties")
    if not props:
        return schema
    relaxed = {}
    for name, prop in props.items():
        limits = {k: prop[k] for k in BOUNDS if k in prop}
        if limits:
            prop = {k: v for k, v in prop.items() if k not in BOUNDS}
            rng = ", ".join(f"{k} {v}" for k, v in limits.items())
            prop["description"] = f"{prop.get('description', '')} ({rng})".strip()
        relaxed[name] = prop
    return {**schema, "properties": relaxed}


@dataclass(frozen=True)
class Paper:
    """Lo que hace falta para escribir una referencia: la demo numera las citas como un paper."""

    pmid: str
    title: str
    journal: str | None = None
    year: int | None = None


@dataclass(frozen=True)
class ToolResult:
    content: str  # lo que lee el modelo
    is_error: bool = False
    pmids: frozenset[str] = frozenset()  # PMIDs vistos: las citas válidas salen de aquí
    truncated: bool = False
    latency_ms: float = 0.0
    papers: tuple[Paper, ...] = ()  # título, revista y año de los PMIDs que los traen


class ToolBox(Protocol):
    def specs(self) -> list[ToolSpec]: ...

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult: ...


# --- Utilidades para que los resultados quepan en el contexto -------------------------------

def _dumps(obj: Any) -> str:
    # JSON compacto: sin sangría ni espacios ahorra ~30 % frente al texto que devuelve el MCP.
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def drop_keys(obj: Any, keys: frozenset[str] = DROP_KEYS) -> Any:
    if isinstance(obj, dict):
        return {k: drop_keys(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [drop_keys(v, keys) for v in obj]
    return obj


def _longest_list(obj: Any) -> list | None:
    """La lista (a cualquier profundidad) que más ocupa al serializarla."""
    best, best_len = None, 0
    stack = [obj]
    while stack:
        node = stack.pop()
        children = node.values() if isinstance(node, dict) else node if isinstance(
            node, list) else ()
        if isinstance(node, list) and len(node) > 1 and (n := len(_dumps(node))) > best_len:
            best, best_len = node, n
        stack.extend(c for c in children if isinstance(c, dict | list))
    return best


def shrink(obj: Any, max_chars: int = MAX_RESULT_CHARS) -> tuple[str, bool]:
    """Serializa `obj` en <= max_chars recortando listas por la mitad.

    Los resultados vienen ordenados (los más relevantes primero), así que quedarse con la primera
    mitad pierde lo menos útil, y el JSON sigue siendo válido. Cortar el texto a lo bruto solo se
    usa como último recurso.
    """
    text = _dumps(obj)
    if len(text) <= max_chars:
        return text, False
    obj = copy.deepcopy(obj)
    while len(text) > max_chars and (lst := _longest_list(obj)) is not None:
        del lst[(len(lst) + 1) // 2:]
        text = _dumps(obj)
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return text + " [truncated: lists were shortened to fit the context]", True


def find_pmids(obj: Any) -> frozenset[str]:
    """PMIDs en los campos `pmid` / `pubmed_reference` (este puede ser "123,456")."""
    found: set[str] = set()
    stack = [obj]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for k, v in node.items():
                if k in PMID_KEYS and isinstance(v, str | int):
                    found.update(re.findall(r"\d{5,9}", str(v)))
                elif isinstance(v, dict | list):
                    stack.append(v)
        elif isinstance(node, list):
            stack.extend(node)
    return frozenset(found)


def find_papers(obj: Any) -> tuple[Paper, ...]:
    """Artículos con título en cualquier nivel del resultado (`search_literature` y
    `get_pubmed_references` los traen; el catálogo solo trae el PMID). Se lee antes de recortar
    el resultado, así que no depende de lo que quepa en el contexto."""
    papers: list[Paper] = []
    stack = [obj]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            pmid, title = node.get("pmid"), node.get("title")
            if isinstance(pmid, str | int) and isinstance(title, str) and title.strip():
                year = node.get("year")
                if not isinstance(year, int):  # amr-mcp da "pubdate": "2016 Feb"
                    match = re.match(r"\d{4}", str(node.get("pubdate") or ""))
                    year = int(match.group()) if match else None
                journal = node.get("journal")
                papers.append(Paper(str(pmid), title.strip(),
                                    journal if isinstance(journal, str) else None, year))
            stack.extend(v for v in node.values() if isinstance(v, dict | list))
        elif isinstance(node, list):
            stack.extend(node)
    return tuple(papers)


def _error(message: str) -> ToolResult:
    # Los errores se devuelven al modelo en vez de lanzarse: así puede corregir los argumentos
    # o probar otra herramienta (amr-mcp, por ejemplo, sugiere "try search_amr_genes").
    return ToolResult(content=f"ERROR: {message}", is_error=True)


# --- Herramienta local: nuestra búsqueda híbrida --------------------------------------------

SEARCH_LITERATURE = ToolSpec(
    name="search_literature",
    description=(
        "Search an index of ~3,000 recent PubMed abstracts (2022-2026) on antimicrobial "
        "resistance. Returns titles and abstracts with their PMIDs; cite them as [PMID:x]. "
        "Use it for findings, mechanisms, epidemiology and clinical evidence. Write the query in "
        "English."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query in English"},
            "k": {"type": "integer", "minimum": 1, "maximum": 6, "default": 4,
                  "description": "Number of abstracts to return"},
        },
        "required": ["query"],
    },
)


class LiteratureTools:
    def __init__(self, retriever: SearchBackend) -> None:
        self._retriever = retriever

    def specs(self) -> list[ToolSpec]:
        return [SEARCH_LITERATURE]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name != SEARCH_LITERATURE.name:
            return _error(f"unknown tool {name!r}")
        query = str(arguments.get("query") or "").strip()
        if len(query) < 3:
            return _error("'query' must have at least 3 characters")
        k = max(1, min(int(arguments.get("k") or 4), 6))
        hits = await self._retriever.search(query, k, "hybrid")
        if not hits:
            return ToolResult(content=_dumps({"results": [], "note": "no matching abstracts"}))
        results = [{
            "pmid": h.pmid, "title": h.title, "journal": h.journal, "year": h.year,
            "abstract": h.abstract if len(h.abstract) <= MAX_ABSTRACT_CHARS
            else h.abstract[:MAX_ABSTRACT_CHARS] + "…",
        } for h in hits]
        content, truncated = shrink({"results": results})
        return ToolResult(content=content, pmids=frozenset(h.pmid for h in hits),
                          truncated=truncated, papers=find_papers(results))


# --- Herramienta local: leer el texto completo de un artículo -------------------------------

READ_PAPER = ToolSpec(
    name="read_paper",
    description=(
        "Read the full text of an article in PubMed Central and return the passages most "
        "relevant to a question, each with its section. Use it when the answer is a specific "
        "detail (a dose, a response rate, a sample size, a method) that the abstract does not "
        "give. Accepts a PMID or a PMCID; only articles with full_text true in search_papers "
        "can be read. Cite the passages with the article's PMID as [PMID:x]."
    ),
    parameters={
        "type": "object",
        "properties": {
            "paper_id": {"type": "string", "description": "PMID (e.g. 38578105) or PMCID "
                                                          "(e.g. PMC11237425)"},
            "question": {"type": "string", "description": "What to look for, in English"},
            "k": {"type": "integer", "minimum": 1, "maximum": 8, "default": 5,
                  "description": "Number of passages to return; 5 is usually enough"},
        },
        "required": ["paper_id", "question"],
    },
)

EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
IDCONV = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
MAX_XML_BYTES = 20_000_000  # el artículo más grande de LitQA2 ocupaba ~1,5 MB
PASSAGE_CHARS = 1300  # presupuesto por pasaje (los trozos son de ~1.200 caracteres)


class ReadPaperTools:
    """`read_paper`: descarga el JATS de PMC (NCBI) y devuelve los pasajes que mejor responden.

    Europe PMC tiene el mismo XML (`fullTextXML`), pero al probarlo daba 500 tras 12-28 s;
    `efetch db=pmc` respondía en menos de 1 s. Los últimos artículos leídos se guardan en memoria:
    el agente suele preguntar dos cosas del mismo artículo.
    """

    def __init__(self, http: httpx.AsyncClient, api_key: str | None = None,
                 email: str | None = None, cache_size: int = 8) -> None:
        self._http = http  # lo crea y lo cierra quien construye la herramienta
        self._params = {"tool": "amr-research-agent", **({"email": email} if email else {}),
                        **({"api_key": api_key} if api_key else {})}
        self._cache: OrderedDict[str, Article] = OrderedDict()
        self._cache_size = cache_size

    def specs(self) -> list[ToolSpec]:
        return [READ_PAPER]

    async def _pmcid(self, paper_id: str) -> str | None:
        if re.fullmatch(r"PMC\d+", paper_id):
            return paper_id
        resp = await self._http.get(IDCONV, params={"format": "json", "ids": paper_id,
                                                    **self._params})
        resp.raise_for_status()
        records = resp.json().get("records") or [{}]
        return records[0].get("pmcid")

    async def _article(self, pmcid: str) -> Article | None:
        if pmcid in self._cache:
            self._cache.move_to_end(pmcid)
            return self._cache[pmcid]
        resp = await self._http.get(EFETCH, params={
            "db": "pmc", "retmode": "xml", "id": pmcid.removeprefix("PMC"), **self._params})
        resp.raise_for_status()
        # Sin <body> no hay texto completo: PMC solo da los metadatos si no es de acceso abierto.
        if len(resp.content) > MAX_XML_BYTES or b"<body" not in resp.content:
            return None
        article = parse_jats(resp.content)
        self._cache[pmcid] = article
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return article

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name != READ_PAPER.name:
            return _error(f"unknown tool {name!r}")
        paper_id = re.sub(r"^(PMID:?\s*)", "", str(arguments.get("paper_id") or "").strip(),
                          flags=re.I).upper()
        question = str(arguments.get("question") or "").strip()
        if not re.fullmatch(r"PMC\d+|\d{5,9}", paper_id):
            return _error("'paper_id' must be a PMID (digits) or a PMCID (PMC followed by digits)")
        if len(question) < 3:
            return _error("'question' must have at least 3 characters")
        # Máx. 8 (antes 10): el modelo pedía siempre el máximo y cada pasaje (~1.200 caracteres)
        # se reenvía en todas las llamadas siguientes. En LitQA2, 8 encuentran el pasaje igual
        # que 10 (0,92 frente a 0,93) con un 20 % menos de texto (decisión 35).
        k = max(1, min(int(arguments.get("k") or 5), 8))

        pmcid = await self._pmcid(paper_id)
        article = await self._article(pmcid) if pmcid else None
        if article is None:
            return _error(f"no full text in PubMed Central for {paper_id}; "
                          "answer from the abstract instead")
        pmid = article.pmid or (paper_id if paper_id.isdigit() else None)
        passages = best_passages(article, question, k)
        data = {"pmid": pmid, "pmcid": pmcid, "title": article.title,
                "journal": article.journal, "year": article.year,
                "passages": [{"section": p.section, "text": p.text} for p in passages]}
        content, truncated = shrink(data, max(MAX_RESULT_CHARS, k * PASSAGE_CHARS))
        return ToolResult(content=content, pmids=frozenset({pmid}) if pmid else frozenset(),
                          truncated=truncated, papers=find_papers(data))


# --- Herramienta remota sin MCP: búsqueda de artículos en vivo ------------------------------

SEARCH_PAPERS = ToolSpec(
    name="search_papers",
    description=(
        "Search PubMed for articles on any biomedical topic, ranked by PubMed's Best Match. "
        "Returns title, journal, year, a short abstract, the PMID and whether the full text is "
        "in PubMed Central (full_text: true means read_paper can read it). Write the query in "
        "English as a few keywords (e.g. 'ELARA tisagenlecleucel follicular lymphoma'), not as a "
        "sentence: every word must match."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keywords in English"},
            "k": {"type": "integer", "minimum": 1, "maximum": 8, "default": 5,
                  "description": "Number of articles to return"},
            "full_text_only": {"type": "boolean", "default": False,
                               "description": "Only articles whose full text can be read"},
        },
        "required": ["query"],
    },
)

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EUROPEPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
# Aquí el abstract solo sirve para elegir qué artículo leer: el detalle lo da read_paper. Recortado
# caben 5 artículos en ~1.000 tokens (Groq Free: 8.000 tokens/min para toda la pregunta).
SEARCH_ABSTRACT_CHARS = 700


class PaperSearchTools:
    """`search_papers`: PubMed ordena, Europe PMC describe.

    El orden de Europe PMC favorece lo reciente: con "ZUMA-1 axicabtagene ciloleucel" no salía
    ningún informe del ensayo entre los 8 primeros, y el Best Match de PubMed los ponía 1.º y 3.º.
    Europe PMC da, en una sola petición para todos los PMIDs, el abstract y si el texto completo
    está en PMC. Si `esearch` falla (estuvo caído el 2026-11-05), se busca solo en Europe PMC.
    """

    def __init__(self, http: httpx.AsyncClient, api_key: str | None = None,
                 email: str | None = None) -> None:
        self._http = http
        self._ncbi = {"tool": "amr-research-agent", **({"email": email} if email else {}),
                      **({"api_key": api_key} if api_key else {})}

    def specs(self) -> list[ToolSpec]:
        return [SEARCH_PAPERS]

    async def _pubmed_ids(self, query: str, k: int, full_text_only: bool) -> list[str]:
        term = f'({query}) AND "pubmed pmc"[sb]' if full_text_only else query
        resp = await self._http.get(ESEARCH, params={
            "db": "pubmed", "term": term, "sort": "relevance", "retmax": k, "retmode": "json",
            **self._ncbi})
        resp.raise_for_status()
        data = resp.json()
        if "esearchresult" not in data or "ERROR" in data["esearchresult"]:
            raise RuntimeError(f"esearch failed: {str(data)[:200]}")  # NCBI da errores con 200
        return data["esearchresult"].get("idlist", [])

    async def _europepmc(self, query: str, k: int) -> list[dict[str, Any]]:
        resp = await self._http.get(EUROPEPMC_SEARCH, params={
            "query": query, "format": "json", "resultType": "core", "pageSize": k})
        resp.raise_for_status()
        return [r for r in resp.json().get("resultList", {}).get("result", []) if r.get("pmid")]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name != SEARCH_PAPERS.name:
            return _error(f"unknown tool {name!r}")
        query = str(arguments.get("query") or "").strip()
        if len(query) < 3:
            return _error("'query' must have at least 3 characters")
        k = max(1, min(int(arguments.get("k") or 5), 8))
        full_text_only = arguments.get("full_text_only") is True
        try:
            pmids = await self._pubmed_ids(query, k, full_text_only)
            records = await self._europepmc(
                f"({' OR '.join(f'EXT_ID:{p}' for p in pmids)}) AND SRC:MED", k) if pmids else []
            by_pmid = {r["pmid"]: r for r in records}
            records = [by_pmid[p] for p in pmids if p in by_pmid]  # en el orden de PubMed
        except (httpx.HTTPError, RuntimeError, ValueError):
            # Respaldo: solo título y abstract, porque Europe PMC también busca en el texto
            # completo y entonces ganan las revisiones que lo nombran todo de pasada.
            fallback = f"TITLE_ABS:({query}) AND SRC:MED"
            if full_text_only:
                fallback += " AND IN_PMC:y"
            records = await self._europepmc(fallback, k)

        results = []
        for r in records:
            abstract = re.sub(r"<[^>]+>", " ", r.get("abstractText") or "")
            abstract = re.sub(r"\s+", " ", abstract).strip()
            year = r.get("pubYear")
            results.append({
                "pmid": r["pmid"], "title": (r.get("title") or "").strip(),
                "journal": ((r.get("journalInfo") or {}).get("journal") or {}).get("title"),
                "year": int(year) if str(year).isdigit() else None,
                # En PMC aunque no sea de licencia abierta (p. ej. manuscritos de autor del NIH):
                # efetch db=pmc devuelve su texto completo y read_paper lo puede leer.
                "full_text": r.get("inPMC") == "Y",
                "abstract": abstract if len(abstract) <= SEARCH_ABSTRACT_CHARS
                else abstract[:SEARCH_ABSTRACT_CHARS] + "…",
            })
        if not results:
            return ToolResult(content=_dumps({"results": [], "note": "no matching articles; "
                                              "try fewer or broader keywords"}))
        content, truncated = shrink({"results": results})
        return ToolResult(content=content, pmids=frozenset(r["pmid"] for r in results),
                          truncated=truncated, papers=find_papers(results))


def drop_seen_abstracts(result: ToolResult, seen: set[str]) -> ToolResult:
    """Quita de un resultado de `search_literature` los abstracts ya mostrados en esta pregunta.

    En la primera eval completa, el agente reformulaba la misma búsqueda hasta 4 veces y casi
    siempre recibía los mismos abstracts (en lit-01, la 4.ª búsqueda no trajo ninguno nuevo):
    ~5.000 caracteres repetidos por vuelta y ninguna señal de que no avanzaba. Ahora solo ve los
    nuevos, y una nota que le dice cuándo dejar de buscar. Los PMIDs siguen contando como citas
    válidas: ya los vio antes.
    """
    if result.is_error or not (result.pmids & seen):
        return result
    try:
        data = json.loads(result.content)
    except ValueError:
        return result
    rows = data.get("results") or []
    fresh = [r for r in rows if r.get("pmid") not in seen]
    repeated = ", ".join(sorted(r["pmid"] for r in rows if r.get("pmid") in seen))
    if fresh:
        note = f"Already shown above, so omitted here: PMIDs {repeated}."
    else:
        note = (f"No new abstracts: every result (PMIDs {repeated}) was already shown above. "
                "Rephrasing the query is not finding new evidence: answer with what you have, "
                "or use another tool.")
    return replace(result, content=_dumps({"results": fresh, "note": note}))


# --- Herramientas remotas: el servidor MCP amr-mcp ------------------------------------------

def amr_mcp_server() -> StdioServerParameters:
    """amr-mcp como subproceso, con el mismo intérprete (y entorno virtual) que la API.

    El subproceso no hereda todo el entorno, solo un mínimo seguro (PATH, HOME...): le pasamos
    explícitamente las variables que usa amr-mcp.
    """
    env = {k: v for k in ("NCBI_API_KEY", "NCBI_EMAIL", "AMR_MCP_CACHE_DIR")
           if (v := os.environ.get(k))}
    return StdioServerParameters(command=sys.executable, args=["-m", "amr_mcp.server"], env=env)


class McpTools:
    """Convierte las herramientas de un servidor MCP en herramientas del agente.

    No sabe nada de AMR: sirve para cualquier servidor MCP. El `Client` lo abre y lo cierra
    quien lo crea (el lifespan de la API o el test), porque su vida es la de la aplicación.
    """

    def __init__(self, client: Client, specs: list[ToolSpec], timeout_s: float = 30.0) -> None:
        self._client = client
        self._specs = specs
        self._names = {s.name for s in specs}
        self.timeout_s = timeout_s

    @classmethod
    async def load(cls, client: Client, allowed: tuple[str, ...] = MCP_TOOLS,
                   timeout_s: float = 30.0) -> "McpTools":
        listed = {t.name: t for t in (await client.list_tools()).tools}
        missing = [n for n in allowed if n not in listed]
        if missing:  # mejor fallar al arrancar que descubrirlo en la primera pregunta
            raise RuntimeError(f"MCP server does not provide tools: {missing}")
        specs = [ToolSpec(n, listed[n].description or "", listed[n].input_schema)
                 for n in allowed]
        return cls(client, specs, timeout_s)

    def specs(self) -> list[ToolSpec]:
        return list(self._specs)

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name not in self._names:
            return _error(f"unknown tool {name!r}")
        result = await self._client.call_tool(name, arguments, read_timeout_seconds=self.timeout_s)
        if result.is_error:
            text = " ".join(getattr(c, "text", "") for c in result.content).strip()
            return _error(text or "tool failed")
        data = result.structured_content
        if data is None:  # servidores que solo devuelven texto
            data = " ".join(getattr(c, "text", "") for c in result.content)
        data = drop_keys(data)
        content, truncated = shrink(data)
        return ToolResult(content=content, pmids=find_pmids(data), truncated=truncated,
                          papers=find_papers(data))


# --- Composición -----------------------------------------------------------------------------

def unknown_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> list[str]:
    """Argumentos que el esquema no declara. Los `null` no cuentan: los modelos a veces mandan
    todos los opcionales a null, y amr-mcp también los descarta."""
    props = spec.parameters.get("properties")
    if props is None:  # esquema sin propiedades declaradas: no hay nada contra qué comprobar
        return []
    return [k for k, v in arguments.items() if v is not None and k not in props]


@dataclass
class Toolset:
    """Junta varios ToolBox, enruta cada llamada al suyo y nunca deja escapar una excepción."""

    boxes: list[ToolBox] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._route: dict[str, ToolBox] = {}
        self._specs: dict[str, ToolSpec] = {}
        for box in self.boxes:
            for spec in box.specs():
                if spec.name in self._route:
                    raise ValueError(f"duplicate tool name {spec.name!r}")
                self._route[spec.name] = box
                self._specs[spec.name] = spec

    def specs(self) -> list[ToolSpec]:
        return [s for box in self.boxes for s in box.specs()]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        t0 = time.perf_counter()
        box = self._route.get(name)
        if box is None:
            result = _error(f"unknown tool {name!r}; available: {sorted(self._route)}")
        elif unknown := unknown_arguments(self._specs[name], arguments):
            # Sin esto, un argumento inventado se ignora en silencio: el modelo mandó `top_k` a
            # search_papers y recibió 5 resultados creyendo haber pedido otro número (decisión
            # 35). Mismo mensaje que amr-mcp, que ya los rechaza (decisión 30).
            valid = ", ".join(self._specs[name].parameters.get("properties", {}))
            result = _error(f"Unknown argument(s) for {name}: {', '.join(unknown)}. "
                            f"Valid arguments: {valid}.")
        else:
            try:
                result = await box.call(name, arguments)
            except TimeoutError:
                result = _error(f"{name} timed out")
            except Exception as exc:  # red caída, NCBI devolviendo 500...: el agente sigue
                result = _error(f"{name} failed: {type(exc).__name__}: {exc}")
        ms = round((time.perf_counter() - t0) * 1000, 1)
        return replace(result, latency_ms=ms)
