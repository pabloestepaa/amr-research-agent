"""Memoria del proceso para los logs de arranque.

Render Free no enseña métricas de memoria (son de pago) y la instancia tiene 512 MB: si un
despliegue falla, la única pista es lo que escribamos en los logs. Leemos /proc, que existe en
Linux (Render, Docker); en macOS las funciones devuelven None y el log dice "n/a".
"""

import os
import resource
import sys
from pathlib import Path

PROC = Path("/proc")


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
    """Pico de memoria de este proceso desde que arrancó (lo que cuenta para un límite)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024  # bytes vs kB


def memory_summary() -> str:
    """Ej.: "RSS api 331 MB (peak 352 MB) + children 95 MB = 426 MB"."""
    me = os.getpid()
    api = rss_mb(me)
    if api is None:
        return f"RSS n/a (peak {peak_rss_mb():.0f} MB)"
    kids = [m for p in child_pids(me) if (m := rss_mb(p)) is not None]
    total = api + sum(kids)
    return (f"RSS api {api:.0f} MB (peak {peak_rss_mb():.0f} MB) + children "
            f"{sum(kids):.0f} MB = {total:.0f} MB")
