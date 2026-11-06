"""Tests de las herramientas del agente.

Las de amr-mcp se prueban con el protocolo MCP de verdad, pero en memoria (`Client(server.mcp)`)
y con un catálogo de ejemplo de 9 entradas: sin subproceso, sin red y sin descargar nada.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from amr_mcp import server as amr_server
from amr_mcp.catalog import Catalog
from mcp import Client

from amr_agent.tools import (
    MCP_TOOLS,
    SEARCH_LITERATURE,
    LiteratureTools,
    McpTools,
    ToolResult,
    Toolset,
    ToolSpec,
    amr_mcp_server,
    bounds_to_description,
    drop_keys,
    find_pmids,
    shrink,
)
from tests.fakes import FakeRetriever, make_hit

CATALOG = Path(__file__).parent / "fixtures" / "catalog_sample.tsv"


@pytest.fixture
def offline_catalog(monkeypatch):
    monkeypatch.setattr(amr_server.state, "catalog", Catalog.from_file(CATALOG))


@asynccontextmanager
async def amr_tools():
    # Un context manager y no una fixture con `yield`: el cliente MCP usa anyio, que exige abrir y
    # cerrar la conexión en la misma tarea, y pytest-asyncio ejecuta el final de la fixture en otra.
    async with Client(amr_server.mcp) as client:
        yield await McpTools.load(client)


# --- Utilidades ------------------------------------------------------------------------------

def test_numeric_bounds_go_to_the_description_not_the_schema():
    # Groq valida el esquema y responde 400 si el modelo se sale del rango (k=10 en lit-02):
    # el rango pasa a la descripción y la validación queda de nuestro lado.
    sent = SEARCH_LITERATURE.to_openai()["function"]["parameters"]["properties"]["k"]
    assert "maximum" not in sent and "minimum" not in sent
    assert sent["description"] == "Number of abstracts to return (minimum 1, maximum 6)"
    assert sent["type"] == "integer" and sent["default"] == 4
    assert "maximum" in SEARCH_LITERATURE.parameters["properties"]["k"]  # el original intacto
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    assert bounds_to_description(schema) == schema
    assert bounds_to_description({"type": "object"}) == {"type": "object"}


async def test_search_literature_clamps_k_instead_of_failing():
    retriever = FakeRetriever([make_hit(str(i)) for i in range(10)])
    result = await LiteratureTools(retriever).call("search_literature",
                                                   {"query": "colistin", "k": 10})
    assert not result.is_error
    assert retriever.calls[-1][1] == 6


def test_shrink_leaves_small_results_untouched():
    text, truncated = shrink({"a": [1, 2, 3]}, max_chars=100)
    assert (text, truncated) == ('{"a":[1,2,3]}', False)


def test_shrink_halves_the_biggest_list_and_keeps_valid_json():
    data = {"total": 100, "results": [{"allele": f"gene-{i}"} for i in range(100)]}
    text, truncated = shrink(data, max_chars=500)
    body, note = text.split(" [truncated", 1)
    parsed = json.loads(body)  # sigue siendo JSON válido
    assert truncated and note.startswith(":")
    assert len(body) <= 500
    assert parsed["total"] == 100
    assert parsed["results"][0] == {"allele": "gene-0"}  # conserva los primeros (más relevantes)


def test_shrink_falls_back_to_cutting_text():
    text, truncated = shrink({"note": "x" * 1000}, max_chars=50)
    assert truncated and text.startswith('{"note":"xxx') and "…" in text


def test_find_pmids_reads_comma_separated_references_at_any_depth():
    data = {"entries": [{"pubmed_reference": "12615876,37255469"}],
            "articles": [{"pmid": "26603172", "title": "not 99999999"}]}
    assert find_pmids(data) == {"12615876", "37255469", "26603172"}


def test_drop_keys_removes_noise_fields_recursively():
    data = {"results": [{"allele": "x", "genbank_protein_accession": "A1", "scope": "core"}]}
    assert drop_keys(data) == {"results": [{"allele": "x"}]}


# --- Herramientas MCP (amr-mcp en memoria) ---------------------------------------------------

async def test_mcp_tools_exposes_only_the_allowed_tools(offline_catalog):
    async with amr_tools() as mcp_tools:
        specs = mcp_tools.specs()
        assert [s.name for s in specs] == list(MCP_TOOLS)
        fn = specs[1].to_openai()["function"]
        assert fn["name"] == "get_amr_gene"
        assert fn["parameters"]["required"] == ["name"]  # el esquema viene del servidor MCP


async def test_mcp_tool_call_returns_compact_json_and_pmids(offline_catalog):
    async with amr_tools() as mcp_tools:
        result = await mcp_tools.call("get_amr_gene", {"name": "blaKPC-2"})
        data = json.loads(result.content)
        entry = data["entries"][0]
        assert not result.is_error
        assert entry["subclass"] == "CARBAPENEM"
        assert entry["refseq_protein_accession"] == "WP_004199234.1"
        assert "genbank_protein_accession" not in entry
        assert result.pmids == {"12615876", "37255469"}


async def test_mcp_tool_errors_are_returned_to_the_model(offline_catalog):
    async with amr_tools() as mcp_tools:
        result = await mcp_tools.call("get_amr_gene", {"name": "nope"})
        assert result.is_error
        assert "search_amr_genes" in result.content  # la pista del servidor llega al modelo


async def test_mcp_tools_refuses_tools_outside_the_allowlist(offline_catalog):
    async with amr_tools() as mcp_tools:
        result = await mcp_tools.call("run_amrfinder", {"fasta_path": "/etc/passwd"})
        assert result.is_error and "unknown tool" in result.content


async def test_mcp_tools_fails_fast_if_the_server_lacks_a_tool(offline_catalog):
    async with Client(amr_server.mcp) as client:
        with pytest.raises(RuntimeError, match="does_not_exist"):
            await McpTools.load(client, allowed=("get_amr_gene", "does_not_exist"))


async def test_amr_mcp_runs_as_a_stdio_subprocess():
    # El camino de producción: un proceso aparte que habla MCP por stdin/stdout. Listar las
    # herramientas no carga el catálogo, así que tampoco necesita red.
    async with Client(amr_mcp_server()) as client:
        tools = await McpTools.load(client)
    assert {s.name for s in tools.specs()} == set(MCP_TOOLS)


# --- Herramienta local de literatura ---------------------------------------------------------

async def test_search_literature_returns_abstracts_and_their_pmids():
    retriever = FakeRetriever([make_hit("111", abstract="A" * 3000), make_hit("222")])
    result = await LiteratureTools(retriever).call(
        "search_literature", {"query": "colistin resistance", "k": 2})
    data = json.loads(result.content)
    assert result.pmids == {"111", "222"}
    assert len(data["results"][0]["abstract"]) == 1501  # recortado a 1.500 + "…"
    assert retriever.calls == [("colistin resistance", 2, "hybrid")]


async def test_search_literature_clamps_k_and_validates_the_query():
    retriever = FakeRetriever([make_hit(str(i)) for i in range(10)])
    tools = LiteratureTools(retriever)
    await tools.call("search_literature", {"query": "mcr-1", "k": 50})
    assert retriever.calls[-1][1] == 6
    assert (await tools.call("search_literature", {"query": ""})).is_error


# --- Toolset ---------------------------------------------------------------------------------

class BrokenTools:
    def specs(self) -> list[ToolSpec]:
        return [ToolSpec("broken", "always fails", {"type": "object", "properties": {}})]

    async def call(self, name, arguments) -> ToolResult:
        raise ConnectionError("NCBI is down")


async def test_toolset_routes_calls_and_turns_exceptions_into_errors():
    toolset = Toolset([LiteratureTools(FakeRetriever([make_hit("111")])), BrokenTools()])
    assert [s.name for s in toolset.specs()] == ["search_literature", "broken"]

    ok = await toolset.call("search_literature", {"query": "mcr-1"})
    assert ok.pmids == {"111"} and ok.latency_ms >= 0

    broken = await toolset.call("broken", {})
    assert broken.is_error and "NCBI is down" in broken.content

    unknown = await toolset.call("nope", {})
    assert unknown.is_error and "search_literature" in unknown.content


def test_toolset_rejects_duplicate_tool_names():
    with pytest.raises(ValueError, match="duplicate"):
        Toolset([BrokenTools(), BrokenTools()])
