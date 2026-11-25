"""Grabar y reproducir las respuestas de las herramientas (el patrón "cassette" de VCR).

Por qué: las evals del agente deben medir al modelo, no a NCBI. Si NCBI va lento o está caído
(como `esearch` el 2026-11-05), o si el catálogo cambia de versión, la misma trayectoria daría
otro resultado. Con la grabación, cada llamada (herramienta + argumentos) devuelve siempre lo
mismo y la eval es reproducible.

- `replay` (por defecto): usa lo grabado; lo que falte se pide en vivo y se graba.
- `refresh`: lo pide todo en vivo y sobrescribe (p. ej., tras cambiar `shrink` o el catálogo).
- `live`: sin grabación.

Solo se graban las respuestas correctas: un error puede ser pasajero (NCBI devolviendo 500) y no
queremos congelarlo. Los errores deterministas (un gen que no existe) se repiten igual en vivo.

Se graba el `ToolResult` entero salvo la latencia: también `papers` (título, tipo de estudio,
`full_text`, que usa la decisión 50) y `excerpts` (el texto contra el que se comprueban las frases
citadas, decisión 42). Hasta el 2026-11-25 solo se guardaba `content`, y en `replay` la validez de
las citas salía 0: las filas antiguas sin esos campos se leen con tuplas vacías y
`fill_from_content` las completa.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from amr_agent.tools import Excerpt, Paper, ToolBox, ToolResult, ToolSpec, find_papers

CassetteMode = Literal["replay", "refresh", "live"]


def cassette_key(name: str, arguments: dict[str, Any]) -> str:
    # sort_keys: {"a":1,"b":2} y {"b":2,"a":1} son la misma llamada.
    return f"{name} {json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"


def to_row(name: str, arguments: dict[str, Any], result: ToolResult) -> dict[str, Any]:
    """Una línea de la grabación. Sin `papers`/`excerpts` vacíos: el fichero de AMR no cambia."""
    row: dict[str, Any] = {"tool": name, "arguments": arguments, "content": result.content,
                           "pmids": sorted(result.pmids), "truncated": result.truncated}
    if result.papers:
        row["papers"] = [asdict(p) for p in result.papers]
    if result.excerpts:
        row["excerpts"] = [asdict(e) for e in result.excerpts]
    return row


def from_row(row: dict[str, Any]) -> ToolResult:
    return ToolResult(content=row["content"], pmids=frozenset(row["pmids"]),
                      truncated=row["truncated"],
                      papers=tuple(Paper(**p) for p in row.get("papers", ())),
                      excerpts=tuple(Excerpt(**e) for e in row.get("excerpts", ())))


class CassetteTools:
    def __init__(self, inner: ToolBox, path: Path, mode: CassetteMode = "replay") -> None:
        self._inner = inner
        self.path = path
        self.mode = mode
        self._store: dict[str, dict] = {}
        if mode != "live" and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._store[cassette_key(row["tool"], row["arguments"])] = row
        self.hits = 0
        self.misses = 0

    def specs(self) -> list[ToolSpec]:
        return self._inner.specs()

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        key = cassette_key(name, arguments)
        if self.mode == "replay" and key in self._store:
            self.hits += 1
            return from_row(self._store[key])
        self.misses += 1
        result = await self._inner.call(name, arguments)
        if self.mode != "live" and not result.is_error:
            self._store[key] = to_row(name, arguments, result)
        return result

    def save(self) -> None:
        if self.mode == "live":
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Ordenado por clave: el fichero no cambia de orden entre ejecuciones y el diff de git
        # solo muestra las llamadas nuevas.
        lines = [json.dumps(self._store[k], ensure_ascii=False) for k in sorted(self._store)]
        self.path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")



def _parse_prefix(content: str) -> Any:
    """El JSON de un resultado, aunque `shrink` lo cortara: se queda con los objetos completos
    de la lista (`results` o `passages`) que llegaron al modelo."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    end = len(content)
    while (end := content.rfind("}", 0, end)) != -1:
        try:
            return json.loads(content[:end + 1] + "]}")
        except json.JSONDecodeError:
            continue
    return None


def fill_from_content(row: dict[str, Any]) -> dict[str, Any]:
    """Completa `papers`/`excerpts` de una fila grabada antes de guardarlos, sin red: salen del
    propio `content`, es decir, de lo que leyó el modelo (abstracts y pasajes tal como los vio,
    ya recortados). Repetir la llamada no sirve: PubMed cambia de orden y de citas, y el recorte
    de pasajes cambió con la decisión 46."""
    if "papers" in row or "excerpts" in row or row["tool"] not in ("search_papers", "read_paper"):
        return row
    data = _parse_prefix(row["content"])
    if not isinstance(data, dict):
        return row
    if row["tool"] == "search_papers":
        excerpts = [Excerpt(str(r["pmid"]), "Abstract", r["abstract"])
                    for r in data.get("results", []) if r.get("pmid") and r.get("abstract")]
    else:
        excerpts = [Excerpt(str(data["pmid"]), p.get("section") or "Full text", p["text"])
                    for p in data.get("passages", []) if data.get("pmid") and p.get("text")]
    result = ToolResult(row["content"], pmids=frozenset(row["pmids"]), truncated=row["truncated"],
                        papers=find_papers(data), excerpts=tuple(excerpts))
    return to_row(row["tool"], row["arguments"], result)
