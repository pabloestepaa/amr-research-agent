"""Robustez del arranque: timeouts con la base de datos y diagnóstico de memoria."""

import time

import psycopg
import pytest

from amr_agent.api.diagnostics import child_pids, memory_summary, rss_mb
from amr_agent.db import apply_schema_async


async def test_unreachable_database_fails_fast_instead_of_hanging():
    # 10.255.255.1 es una dirección privada sin nadie detrás: sin connect_timeout, la conexión
    # esperaría lo que dicte TCP (minutos). libpq redondea los timeouts por debajo de 2 s a 2 s.
    start = time.perf_counter()
    with pytest.raises(psycopg.OperationalError):
        await apply_schema_async("postgresql://u@10.255.255.1:5432/db", connect_timeout=2)
    assert time.perf_counter() - start < 10


def fake_proc(tmp_path, pid: int, ppid: int, name: str, rss_kb: int) -> None:
    d = tmp_path / str(pid)
    d.mkdir()
    (d / "stat").write_text(f"{pid} ({name}) S {ppid} 1 1 0 -1\n")
    (d / "status").write_text(f"Name:\t{name}\nVmPeak:\t999 kB\nVmRSS:\t{rss_kb} kB\n")


def test_reads_rss_and_children_from_proc(tmp_path):
    fake_proc(tmp_path, 1, 0, "uvicorn", 340 * 1024)
    fake_proc(tmp_path, 7, 1, "python -m amr_mcp) x", 95 * 1024)  # nombre con espacios y ")"
    fake_proc(tmp_path, 8, 3, "other", 10 * 1024)
    (tmp_path / "self").mkdir()  # entradas no numéricas: se ignoran
    assert rss_mb(1, tmp_path) == 340
    assert child_pids(1, tmp_path) == [7]
    assert rss_mb(7, tmp_path) == 95


def test_missing_proc_is_not_an_error(tmp_path):
    assert rss_mb(123, tmp_path) is None  # macOS: no hay /proc
    assert child_pids(123, tmp_path / "nope") == []
    assert "MB" in memory_summary()  # en Linux da cifras; en macOS, "n/a" y el pico
