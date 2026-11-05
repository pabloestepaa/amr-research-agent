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
"""

import json
from pathlib import Path
from typing import Any, Literal

from amr_agent.tools import ToolBox, ToolResult, ToolSpec

CassetteMode = Literal["replay", "refresh", "live"]


def cassette_key(name: str, arguments: dict[str, Any]) -> str:
    # sort_keys: {"a":1,"b":2} y {"b":2,"a":1} son la misma llamada.
    return f"{name} {json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"


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
            row = self._store[key]
            return ToolResult(content=row["content"], pmids=frozenset(row["pmids"]),
                              truncated=row["truncated"])
        self.misses += 1
        result = await self._inner.call(name, arguments)
        if self.mode != "live" and not result.is_error:
            self._store[key] = {"tool": name, "arguments": arguments, "content": result.content,
                                "pmids": sorted(result.pmids), "truncated": result.truncated}
        return result

    def save(self) -> None:
        if self.mode == "live":
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Ordenado por clave: el fichero no cambia de orden entre ejecuciones y el diff de git
        # solo muestra las llamadas nuevas.
        lines = [json.dumps(self._store[k], ensure_ascii=False) for k in sorted(self._store)]
        self.path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
