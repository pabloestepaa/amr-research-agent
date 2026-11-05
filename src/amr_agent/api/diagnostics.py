"""Memoria del proceso para los logs de arranque.

Render Free no enseña métricas de memoria (son de pago) y la instancia tiene 512 MB: si un
despliegue falla, la única pista es lo que escribamos en los logs.

Dos fuentes, las dos de Linux (Render, Docker); en macOS devuelven None:
- el cgroup del contenedor: la memoria que el kernel le cuenta y la que compara con el límite.
  Es la cifra que importa.
- el RSS de cada proceso (/proc): sirve para ver el reparto entre la API y amr-mcp, pero no se
  puede sumar, porque cuenta dos veces las librerías compartidas (en CI: 450 MB sumando RSS
  frente a 368 MiB reales del contenedor).
"""

import os
import resource
import sys
from pathlib import Path

PROC = Path("/proc")
CGROUP = Path("/sys/fs/cgroup")
MB = 1024 * 1024


def _read_bytes(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None  # "max" = sin límite


def container_memory(root: Path = CGROUP) -> tuple[float, float | None, float | None] | None:
    """(uso actual, pico, límite) del contenedor en MB, o None fuera de un cgroup.

    cgroup v2 (lo habitual hoy) y v1 tienen ficheros distintos. El pico puede faltar (v2 lo
    añadió en el kernel 5.19) y el límite es None si no hay ninguno.
    """
    for current, peak, limit in (
        ("memory.current", "memory.peak", "memory.max"),  # v2
        ("memory/memory.usage_in_bytes", "memory/memory.max_usage_in_bytes",
         "memory/memory.limit_in_bytes"),  # v1
    ):
        used = _read_bytes(root / current)
        if used is not None:
            top, cap = _read_bytes(root / peak), _read_bytes(root / limit)
            # En v1, "sin límite" es un número enorme en vez de "max".
            cap = cap if cap is not None and cap < 1 << 50 else None
            return (used / MB, top / MB if top is not None else None,
                    cap / MB if cap is not None else None)
    return None


def rss_mb(pid: int, proc: Path = PROC) -> float | None:
    """Memoria residente actual (VmRSS) de un proceso, en MB."""
    try:
        for line in (proc / str(pid) / "status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024  # viene en kB
    except (OSError, ValueError, IndexError):
        return None
    return None


def child_pids(pid: int, proc: Path = PROC) -> list[int]:
    """Procesos hijos directos (p. ej., el subproceso amr-mcp)."""
    children = []
    try:
        entries = list(proc.iterdir())
    except OSError:
        return []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            # /proc/<pid>/stat: "pid (nombre) estado ppid ..."; el nombre puede tener espacios,
            # así que partimos después del último ")".
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[1]) == pid:
                children.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    return children


def peak_rss_mb() -> float:
    """Pico de memoria del proceso de la API (sin amr-mcp); se usa donde no hay cgroup."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024  # bytes vs kB


def memory_summary(proc: Path = PROC, cgroup: Path = CGROUP) -> str:
    """Ej.: "container 368 MB (peak 381 MB) of 512 MB · RSS api 354 MB, children 95 MB"."""
    parts = []
    if (mem := container_memory(cgroup)) is not None:
        used, peak, limit = mem
        text = f"container {used:.0f} MB"
        if peak is not None:
            text += f" (peak {peak:.0f} MB)"
        if limit is not None:
            text += f" of {limit:.0f} MB"
        parts.append(text)
    me = os.getpid()
    api = rss_mb(me, proc)
    if api is None:
        parts.append(f"RSS n/a (api peak {peak_rss_mb():.0f} MB)")
    else:
        kids = [m for p in child_pids(me, proc) if (m := rss_mb(p, proc)) is not None]
        parts.append(f"RSS api {api:.0f} MB, children {sum(kids):.0f} MB")
    return " · ".join(parts)
