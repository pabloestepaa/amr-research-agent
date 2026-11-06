"""Herramientas del agente: la búsqueda en nuestra literatura y las del servidor MCP `amr-mcp`.

El agente solo ve un `ToolBox`: una lista de esquemas (`specs`) y un método `call`. Le da igual
si la herramienta es una función local o vive en otro proceso detrás del protocolo MCP.
"""

import copy
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from mcp import Client
from mcp.client.stdio import StdioServerParameters

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
class ToolResult:
    content: str  # lo que lee el modelo
    is_error: bool = False
    pmids: frozenset[str] = frozenset()  # PMIDs vistos: las citas válidas salen de aquí
    truncated: bool = False
    latency_ms: float = 0.0


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
                          truncated=truncated)


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
        return ToolResult(content=content, pmids=find_pmids(data), truncated=truncated)


# --- Composición -----------------------------------------------------------------------------

@dataclass
class Toolset:
    """Junta varios ToolBox, enruta cada llamada al suyo y nunca deja escapar una excepción."""

    boxes: list[ToolBox] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._route: dict[str, ToolBox] = {}
        for box in self.boxes:
            for spec in box.specs():
                if spec.name in self._route:
                    raise ValueError(f"duplicate tool name {spec.name!r}")
                self._route[spec.name] = box

    def specs(self) -> list[ToolSpec]:
        return [s for box in self.boxes for s in box.specs()]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        t0 = time.perf_counter()
        box = self._route.get(name)
        if box is None:
            result = _error(f"unknown tool {name!r}; available: {sorted(self._route)}")
        else:
            try:
                result = await box.call(name, arguments)
            except TimeoutError:
                result = _error(f"{name} timed out")
            except Exception as exc:  # red caída, NCBI devolviendo 500...: el agente sigue
                result = _error(f"{name} failed: {type(exc).__name__}: {exc}")
        ms = round((time.perf_counter() - t0) * 1000, 1)
        return ToolResult(result.content, result.is_error, result.pmids, result.truncated, ms)
