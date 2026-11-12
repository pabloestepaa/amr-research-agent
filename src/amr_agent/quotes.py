"""Frases literales que respaldan cada cita en el modo Papers (decisión 42).

El modelo termina la respuesta con un bloque como este:

    Quotes:
    [PMID:36538619] "Insulin independence occurred in 78.6% of recipients."

El código separa el bloque de la respuesta y busca cada frase en el texto que devolvieron las
herramientas para ese PMID (el extracto del abstract o los pasajes de read_paper). Es el mismo
guardarraíl que con los PMIDs: un PMID solo vale si una herramienta lo devolvió, y una frase solo
vale si está en lo que el modelo leyó. Sin red ni LLM.
"""

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from amr_agent.tools import Excerpt

# "Quotes:", también con negrita o como título de Markdown. Si contesta en español puede
# traducir la cabecera aunque se le pida "Quotes:".
HEADER_RE = re.compile(
    r"^[ \t]*(?:#+[ \t]*)?\**[ \t]*(?:quotes|citas(?:[ \t]+textuales)?|frases)[ \t]*\**"
    r"[ \t]*:?[ \t]*\**[ \t]*$", re.IGNORECASE | re.MULTILINE)
_OPEN, _CLOSE = "\"“«„", "\"”»“"
# Una línea con un PMID y una frase entre comillas, en cualquier orden:
#   - [PMID:1] "frase"    [PMID:1]: “frase”    "frase" [PMID:1]
LINE_RE = re.compile(
    rf"^[ \t]*(?:[-*•>][ \t]*)?(?:\[PMID:(?P<a>\d+)\][ \t]*[:\-–—]?[ \t]*[{_OPEN}](?P<qa>.+)"
    rf"[{_CLOSE}]|[{_OPEN}](?P<qb>.+)[{_CLOSE}][ \t]*[\-–—]?[ \t]*\[PMID:(?P<b>\d+)\])"
    r"[ \t]*\.?[ \t]*$", re.MULTILINE)
# "…", "..." o "[…]": el modelo puede cortar la frase; cada trozo tiene que estar, en orden.
ELLIPSIS_RE = re.compile(r"\s*\[?\s*(?:…|\.{3})\s*\]?\s*")
# Sin contar espacios. Un "78.6%" suelto aparece en muchos sitios y no dice de dónde sale.
MIN_QUOTE_CHARS = 20
# Con el bloque de frases, gpt-oss a veces sacaba las citas del texto: notas [^1] o una lista de
# PMIDs ("**Citas**", "- [PMID:1]") justo antes. Con las frases ya no dicen nada: se quitan.
FOOTNOTE_RE = re.compile(r"[ \t]*\[\^\d+\]")
PMID_LINE_RE = re.compile(r"[ \t]*(?:[-*•][ \t]*|\d+[.)][ \t]*)?(?:\[PMID:\d+\][ \t,;]*)+")
LIST_HEADER_RE = re.compile(
    r"[ \t]*(?:#+[ \t]*)?\**[ \t]*(?:citas|referencias|references|sources|fuentes)[ \t]*\**"
    r"[ \t]*:?[ \t]*\**[ \t]*", re.IGNORECASE)


@dataclass(frozen=True)
class RawQuote:
    pmid: str
    text: str


def split_quotes(answer: str) -> tuple[str, list[RawQuote]]:
    """Separa la respuesta del bloque de frases. Sin cabecera, toma las líneas de cita del final."""
    headers = list(HEADER_RE.finditer(answer))
    if headers:
        last = headers[-1]
        body, block = answer[:last.start()], answer[last.end():]
        quotes = [_raw(m) for m in LINE_RE.finditer(block)]
    else:
        lines = answer.rstrip().split("\n")
        quotes = []
        while lines and (m := LINE_RE.fullmatch(lines[-1])):
            quotes.insert(0, _raw(m))
            lines.pop()
        body = "\n".join(lines)
    unique = list(dict.fromkeys(q for q in quotes if q.text))
    if unique:
        body = _drop_reference_list(FOOTNOTE_RE.sub("", body))
    return body.rstrip(), unique


def _drop_reference_list(body: str) -> str:
    lines = body.rstrip().split("\n")
    end = len(lines)
    while end and (PMID_LINE_RE.fullmatch(lines[end - 1]) or not lines[end - 1].strip()):
        end -= 1
    if end == len(lines):
        return body
    if end and LIST_HEADER_RE.fullmatch(lines[end - 1]):
        end -= 1
    return "\n".join(lines[:end]) if any(x.strip() for x in lines[:end]) else body


def _raw(m: re.Match[str]) -> RawQuote:
    return RawQuote(m["a"] or m["b"], (m["qa"] or m["qb"] or "").strip())


def _norm(text: str) -> str:
    # NFKC convierte el guion de gpt-oss (U+2011) en U+2010 y los espacios finos en espacios;
    # los espacios se quitan del todo: "78.6 %" y "78.6%" son la misma frase.
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[‐-―−]", "-", text)
    text = re.sub(r"[‘’‚‛]", "'", re.sub(r"[“”„‟«»]", '"', text))
    return re.sub(r"\s+", "", text)


def locate(quote: RawQuote, excerpts: Iterable[Excerpt]) -> Excerpt | None:
    """El texto de ese PMID en el que aparece la frase, o None si no aparece (o es muy corta)."""
    parts = [_norm(p).strip(".,;:") for p in ELLIPSIS_RE.split(quote.text)]
    parts = [p for p in parts if p]
    if sum(len(p) for p in parts) < MIN_QUOTE_CHARS:
        return None
    for excerpt in excerpts:
        if excerpt.pmid != quote.pmid:
            continue
        haystack, pos = _norm(excerpt.text), 0
        for part in parts:
            found = haystack.find(part, pos)
            if found < 0:
                break
            pos = found + len(part)
        else:
            return excerpt
    return None
