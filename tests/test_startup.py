"""Robustez del arranque: timeouts con la base de datos y diagnóstico de memoria."""

import time

import psycopg
import pytest

from amr_agent.api.diagnostics import child_pids, container_memory, memory_summary, rss_mb
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


def test_missing_proc_and_cgroup_are_not_errors(tmp_path):
    # macOS: no hay /proc ni cgroup
    assert rss_mb(123, tmp_path) is None
    assert child_pids(123, tmp_path / "nope") == []
    assert container_memory(tmp_path) is None
    assert memory_summary(tmp_path, tmp_path).startswith("RSS n/a (api peak ")


MiB = 1024 * 1024


def test_reads_container_memory_from_cgroup_v2(tmp_path):
    (tmp_path / "memory.current").write_text(f"{368 * MiB}\n")
    (tmp_path / "memory.peak").write_text(f"{381 * MiB}\n")
    (tmp_path / "memory.max").write_text(f"{512 * MiB}\n")
    assert container_memory(tmp_path) == (368, 381, 512)
    assert memory_summary(tmp_path / "proc", tmp_path).startswith(
        "container 368 MB (peak 381 MB) of 512 MB · ")


def test_cgroup_v2_without_peak_or_limit(tmp_path):
    (tmp_path / "memory.current").write_text(f"{100 * MiB}")
    (tmp_path / "memory.max").write_text("max")  # sin límite
    assert container_memory(tmp_path) == (100, None, None)


def test_reads_container_memory_from_cgroup_v1(tmp_path):
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "memory.usage_in_bytes").write_text(f"{200 * MiB}")
    (mem / "memory.max_usage_in_bytes").write_text(f"{250 * MiB}")
    (mem / "memory.limit_in_bytes").write_text("9223372036854771712")  # v1: "sin límite"
    assert container_memory(tmp_path) == (200, 250, None)
