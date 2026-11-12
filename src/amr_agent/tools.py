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

from amr_agent.fulltext import Article, abstract_excerpt, best_passages, parse_jats
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
    study_type: str | None = None  # "Phase 2 trial", "Meta-analysis"... (modo Papers)
    cited_by: int | None = None  # citas en Europe PMC, solo desde MIN_CITED_BY

    def merge(self, other: "Paper") -> "Paper":
        """Los datos de `other` que faltan aquí: read_paper no trae el tipo de estudio que dio
        search_papers, y no debe borrarlo al leer el artículo."""
        return replace(self, **{f: getattr(other, f) for f in ("journal", "year", "study_type",
                                                                "cited_by")
                                if getattr(self, f) is None and getattr(other, f) is not None})


@dataclass(frozen=True)
class Excerpt:
    """Texto de un artículo que devolvió una herramienta: contra él se comprueban las frases que
    cita el modelo (quotes.py, decisión 42)."""

    pmid: str
    section: str  # "Abstract" o la sección del pasaje que leyó read_paper (Results, Methods...)
    text: str


@dataclass(frozen=True)
class ToolResult:
    content: str  # lo que lee el modelo
    is_error: bool = False
    pmids: frozenset[str] = frozenset()  # PMIDs vistos: las citas válidas salen de aquí
    truncated: bool = False
    latency_ms: float = 0.0
    papers: tuple[Paper, ...] = ()  # título, revista y año de los PMIDs que los traen
    excerpts: tuple[Excerpt, ...] = ()  # abstracts y pasajes, sin recortar ni escapar


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
                journal, kind, cited = (node.get(k) for k in ("journal", "study_type",
                                                              "cited_by"))
                papers.append(Paper(str(pmid), title.strip(),
                                    journal if isinstance(journal, str) else None, year,
                                    kind if isinstance(kind, str) else None,
                                    cited if isinstance(cited, int) else None))
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
        excerpts = tuple(Excerpt(pmid, p.section or "Full text", p.text)
                         for p in passages) if pmid else ()
        return ToolResult(content=content, pmids=frozenset({pmid}) if pmid else frozenset(),
                          truncated=truncated, papers=find_papers(data), excerpts=excerpts)


# --- Herramienta remota sin MCP: búsqueda de artículos en vivo ------------------------------

SEARCH_PAPERS = ToolSpec(
    name="search_papers",
    description=(
        "Search PubMed for articles on any biomedical topic, ranked by PubMed's Best Match. "
        "Returns title, journal, year, the study type when PubMed has it (study_type), the "
        "citation count when it is 10 or more (cited_by), a short abstract, the PMID and whether "
        "the full text is in PubMed Central (full_text: true means read_paper can read it). "
        "Write the query in "
        "English as a few keywords (e.g. 'ELARA tisagenlecleucel follicular lymphoma'), not as a "
        "sentence: every word must match."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Keywords in English"},
            # `top_k` y no `k`: gpt-oss mandó `top_k` en la 1.ª llamada de las 9 tareas de la
            # primera eval (decisión 38) y cada error le costaba un paso.
            "top_k": {"type": "integer", "minimum": 1, "maximum": 8, "default": 5,
                      "description": "Number of articles to return"},
            "full_text_only": {"type": "boolean", "default": False,
                               "description": "Only articles whose full text can be read"},
        },
        "required": ["query"],
    },
)

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EUROPEPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
# El abstract sirve para elegir qué artículo leer y, en preguntas generales, para responder
# (decisión 40). Recortado caben 5 artículos en ~1.000 tokens (Groq Free: 8.000 tokens/min para
# toda la pregunta); el recorte se queda con las frases que mejor encajan con la consulta.
SEARCH_ABSTRACT_CHARS = 700
# Las citas que cuenta Europe PMC. La búsqueda trae sobre todo artículos recientes, y en 175 de
# 7 búsquedas reales casi todos tenían 0: un "0 citas" en un artículo de este año parecería una
# señal de mala calidad. Desde 10 sí distingue un ensayo de referencia (TRANSCEND: 1.990).
MIN_CITED_BY = 10
# El tipo de publicación de PubMed (MeSH) y el de PMC (JATS) lo escriben distinto: "Review" y
# "review-article". Casi la mitad de los artículos solo dice "Journal Article" (MEDLINE aún no
# los ha clasificado): sin etiqueta mejor que una genérica.
TRIAL_PHASES = {"clinical trial, phase i": "1", "clinical trial, phase ii": "2",
                "clinical trial, phase iii": "3", "clinical trial, phase iv": "4"}
OTHER_STUDY_TYPES = (
    ({"practice guideline", "guideline"}, "Guideline"),
    ({"observational study"}, "Observational study"),
    ({"scoping review"}, "Scoping review"),
    ({"review", "review-article"}, "Review"),
    ({"case reports", "case-report", "case-study"}, "Case report"),
    ({"letter"}, "Letter"), ({"editorial"}, "Editorial"), ({"comment"}, "Comment"),
    ({"preprint"}, "Preprint"),
    ({"research-article"}, "Research article"),
)


def study_type(pub_types: list[str]) -> str | None:
    """La etiqueta más informativa: metaanálisis > revisión sistemática > ensayo > el resto."""
    types = {t.strip().lower() for t in pub_types}
    if "meta-analysis" in types:
        return "Meta-analysis"
    if types & {"systematic review", "systematic-review"}:
        return "Systematic review"
    phases = sorted({TRIAL_PHASES[t] for t in types if t in TRIAL_PHASES})
    randomized = "randomized controlled trial" in types
    if phases:
        return f"Phase {'/'.join(phases)} {'randomized trial' if randomized else 'trial'}"
    if randomized:
        return "Randomized trial"
    if types & {"clinical trial", "controlled clinical trial"}:
        return "Clinical trial"
    return next((label for keys, label in OTHER_STUDY_TYPES if types & keys), None)


# PubMed lee "CAR T" como un autor ("car t"[Author]): "GD2 CAR T H3K27M glioma" daba 3 artículos
# irrelevantes y con "CAR T" entre comillas, 15 con el ensayo bueno el 2.º. "CAR T-cell" no va al
# autor, pero expande "car" a coches y proteínas. Todas las variantes pasan a la frase "CAR T".
CAR_T_RE = re.compile(r'(?<!")\bCAR[\s-]T(?:[\s-](cells?)\b|\b)(?!")', re.IGNORECASE)


def pubmed_term(query: str) -> str:
    return CAR_T_RE.sub(lambda m: '"CAR T"' + (f" {m[1]}" if m[1] else ""), query)


# Reintento con menos palabras (decisión 45). PubMed exige que estén todas y el modelo mezcla el
# estudio con el dato que busca ("... HLA-A*02 phase 1 solid cancers"): en las dos evals de
# literatura, 18 de las 38 búsquedas grabadas dieron 0 artículos. El modelo pone delante el
# ensayo, el fármaco y la enfermedad, y detrás el detalle; las palabras genéricas no distinguen un
# estudio de otro.
# Una palabra, o una frase entre comillas pegada o no a otra (GD2-"CAR T").
QUERY_TOKEN_RE = re.compile(r'[^\s"]*"[^"]*"[^\s"]*|\S+')
GENERIC_WORDS = frozenset("""
    phase i ii iii iv trial trials clinical study studies cohort randomized randomised open-label
    multicenter multicentre pivotal final interim analysis analyses update updated results report
    follow-up long-term year years month months week weeks day days one two three four five ten
    patients patient adults participants subjects cases
    efficacy safety response responses rate rates duration median mean overall survival time
    incidence proportion percentage number toxicity toxicities adverse event events grade dose
    doses outcome outcomes
    cells cell t t-cell t-cells
    a an the of in on at to for from with by and or vs versus after before during what how many
    which
""".split())
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?%?|(?:\d+|one|two|three|four|five|ten)-(?:year|month|"
                       r"week|day)s?")
# Con las 18 búsquedas vacías grabadas: las 4 primeras y luego las 2 primeras encuentran el
# artículo en 14 (las otras 4 llevan palabras que el artículo no tiene); quitando la última
# palabra cada vez, 12 con casi el doble de peticiones.
RETRY_WORDS = (4, 2)


def _generic(word: str) -> bool:
    w = word.lower().strip(",.;:()[]")
    # "*" es el comodín de PubMed: "HLA-A*02" no encuentra el artículo que lo lleva en el título.
    return w in GENERIC_WORDS or bool(NUMBER_RE.fullmatch(w)) or "*" in w


def shorter_queries(query: str) -> list[str]:
    """Versiones más cortas de una consulta sin resultados, en el orden en que se prueban."""
    words = QUERY_TOKEN_RE.findall(pubmed_term(query))
    kept = [w for w in words if not _generic(w)] or words
    out: list[str] = []
    for n in RETRY_WORDS:
        shorter = " ".join(kept[:n])
        if shorter != " ".join(words) and shorter not in out:
            out.append(shorter)
    return out


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

    async def _search(self, query: str, k: int, full_text_only: bool) -> list[dict[str, Any]]:
        try:
            pmids = await self._pubmed_ids(query, k, full_text_only)
            records = await self._europepmc(
                f"({' OR '.join(f'EXT_ID:{p}' for p in pmids)}) AND SRC:MED", k) if pmids else []
            by_pmid = {r["pmid"]: r for r in records}
            return [by_pmid[p] for p in pmids if p in by_pmid]  # en el orden de PubMed
        except (httpx.HTTPError, RuntimeError, ValueError):
            # Respaldo: solo título y abstract, porque Europe PMC también busca en el texto
            # completo y entonces ganan las revisiones que lo nombran todo de pasada.
            fallback = f"TITLE_ABS:({query}) AND SRC:MED"
            if full_text_only:
                fallback += " AND IN_PMC:y"
            return await self._europepmc(fallback, k)

    async def _pubmed_ids(self, query: str, k: int, full_text_only: bool) -> list[str]:
        query = pubmed_term(query)
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
        k = max(1, min(int(arguments.get("top_k") or 5), 8))
        full_text_only = arguments.get("full_text_only") is True
        records = await self._search(query, k, full_text_only)
        tried: list[str] = []
        if not records:
            for shorter in shorter_queries(query):
                tried.append(shorter)
                if records := await self._search(shorter, k, full_text_only):
                    break

        results = []
        for r in records:
            abstract = re.sub(r"<[^>]+>", " ", r.get("abstractText") or "")
            abstract = re.sub(r"\s+", " ", abstract).strip()
            year = r.get("pubYear")
            kind = study_type((r.get("pubTypeList") or {}).get("pubType") or [])
            cited = r.get("citedByCount")
            results.append({
                "pmid": r["pmid"], "title": (r.get("title") or "").strip(),
                "journal": ((r.get("journalInfo") or {}).get("journal") or {}).get("title"),
                "year": int(year) if str(year).isdigit() else None,
                # Solo si los hay: cada clave vacía son tokens en cada vuelta.
                **({"study_type": kind} if kind else {}),
                **({"cited_by": cited} if isinstance(cited, int) and cited >= MIN_CITED_BY
                   else {}),
                # En PMC aunque no sea de licencia abierta (p. ej. manuscritos de autor del NIH):
                # efetch db=pmc devuelve su texto completo y read_paper lo puede leer.
                "full_text": r.get("inPMC") == "Y",
                # Con la consulta del modelo, también tras el reintento: el detalle que busca
                # ("duration of response") elige las frases del abstract.
                "abstract": abstract_excerpt(abstract, query, SEARCH_ABSTRACT_CHARS),
            })
        if not results:
            note = "no matching articles: PubMed needs every keyword to match."
            if tried:  # que no repita las que ya se probaron
                note += f" Shorter versions found nothing either ({' | '.join(tried)})."
            return ToolResult(content=_dumps({"results": [], "note": (
                f"{note} Try fewer or other keywords, e.g. the trial or therapy name and the "
                "disease. This search did not count towards your limit.")}))
        # La nota va delante: el modelo (y la demo, que enseña el principio del resultado) ve
        # que la búsqueda que funcionó no es la suya.
        found = ({"note": "no article matched every keyword; these are the results for a "
                          f"shorter query: {tried[-1]}"} if tried else {})
        content, truncated = shrink({**found, "results": results})
        return ToolResult(content=content, pmids=frozenset(r["pmid"] for r in results),
                          truncated=truncated, papers=find_papers(results),
                          excerpts=tuple(Excerpt(r["pmid"], "Abstract", r["abstract"])
                                         for r in results if r["abstract"]))


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
