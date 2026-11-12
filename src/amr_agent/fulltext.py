"""Texto completo de un artículo (JATS de PMC): trozos con su sección y búsqueda BM25 dentro.

El dato exacto que busca un investigador (una dosis, una tasa de respuesta, un tamaño de muestra)
casi nunca está en el abstract: en 40 artículos de terapia celular, el 96 % de las cifras de
Resultados y tablas no aparecía en él (docs/decisions.md, decisión 33). Este módulo no usa la red
ni la base de datos: recibe el XML y devuelve pasajes ordenados. La descarga está en tools.py.

¿Por qué BM25 y no embeddings? Un artículo son ~50 trozos y calcular sus embeddings en vivo
cuesta ~9 s en un portátil (más en Render). Medido con LitQA2 y con preguntas parafraseadas,
BM25 encuentra el pasaje con la respuesta entre los 10 primeros el 92 % de las veces, igual que
la combinación con embeddings.
"""

import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass

CHUNK_CHARS = 1200

# Palabras vacías: no ayudan a distinguir un trozo de otro y, sin IDF alto, solo meten ruido.
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were "
    "what which with how does do did than into their these those whose who when where why after "
    "before between during".split())

# Secciones que no aportan datos del estudio.
SKIP_SECTION = re.compile(r"acknowledg|funding|author contribution|competing interest|"
                          r"conflict of interest|data availability|code availability", re.I)


@dataclass(frozen=True)
class Block:
    section: str  # "Results > Safety", "Figure 2", "Table 1"...
    text: str


@dataclass(frozen=True)
class Article:
    pmcid: str | None
    pmid: str | None
    title: str
    journal: str | None
    year: int | None
    blocks: tuple[Block, ...]


def _text(el: ET.Element | None) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip() if el is not None else ""


def _own_text(p: ET.Element) -> str:
    """Texto de un párrafo sin las tablas y figuras que lleve dentro (esas van en su bloque)."""
    parts = [p.text or ""]
    for child in p:
        if child.tag not in ("table-wrap", "fig"):
            parts.append("".join(child.itertext()))
        parts.append(child.tail or "")
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _walk(sec: ET.Element, path: tuple[str, ...], out: list[Block]) -> None:
    """Recorre una sección: sus párrafos llevan el camino de títulos ("Results > Safety")."""
    title = _text(sec.find("title"))
    if title and SKIP_SECTION.search(title):
        return
    path = (*path, title) if title else path
    for child in sec:
        if child.tag == "sec":
            _walk(child, path, out)
        elif child.tag == "p":
            # Una tabla o figura dentro de un párrafo se guarda aparte, con su etiqueta.
            for fl in child.findall(".//table-wrap") + child.findall(".//fig"):
                _float(fl, out)
            out.append(Block(" > ".join(path) or "Body", _own_text(child)))
        elif child.tag in ("fig", "table-wrap"):
            _float(child, out)


def _float(el: ET.Element, out: list[Block]) -> None:
    label = _text(el.find("label")) or ("Table" if el.tag == "table-wrap" else "Figure")
    out.append(Block(label, _text(el)))


def parse_jats(xml: bytes) -> Article:
    """Lee el XML JATS que devuelve `efetch db=pmc` (o Europe PMC `fullTextXML`).

    Incluye los párrafos del cuerpo (estén o no dentro de una <sec>: hay artículos sin
    secciones), los pies de figura y las tablas, también los que van en <floats-group> al final
    (Nature). No se deduce nada del nombre de la sección: Nature no tiene "Results", cada sección
    se titula con su hallazgo.
    """
    root = ET.fromstring(xml)
    meta = root.find(".//article-meta")
    ids = {el.get("pub-id-type"): (el.text or "").strip()
           for el in (meta.findall("article-id") if meta is not None else [])}
    pmcid = ids.get("pmc") or ids.get("pmcid")
    if pmcid and not pmcid.startswith("PMC"):
        pmcid = f"PMC{pmcid}"
    year = next((int(y.text) for y in root.iterfind(".//article-meta/pub-date/year")
                 if y.text and y.text.strip().isdigit()), None)

    blocks: list[Block] = []
    body = root.find(".//body")
    if body is not None:
        _walk(body, (), blocks)
    floats = root.find(".//floats-group")
    if floats is not None:
        for el in floats:
            if el.tag in ("fig", "table-wrap"):
                _float(el, blocks)
    seen: set[str] = set()
    unique = []
    for b in blocks:  # sin duplicados ni restos ("Fig. 2", celdas vacías)
        if len(b.text) > 40 and b.text not in seen:
            seen.add(b.text)
            unique.append(b)
    return Article(
        pmcid=pmcid or None, pmid=ids.get("pmid") or None,
        title=_text(root.find(".//article-meta/title-group/article-title")),
        journal=_text(root.find(".//journal-meta//journal-title")) or None,
        year=year, blocks=tuple(unique))


def chunk(blocks: tuple[Block, ...] | list[Block], size: int = CHUNK_CHARS) -> list[Block]:
    """Junta bloques seguidos de la misma sección hasta ~`size` caracteres.

    Un párrafo largo no se parte: cortarlo a mitad de frase separaría el dato de su contexto.
    El tamaño apenas cambia el resultado (600, 1.200 y 2.000 caracteres dieron lo mismo).
    """
    out: list[Block] = []
    for b in blocks:
        last = out[-1] if out else None
        if last and last.section == b.section and len(last.text) + len(b.text) <= size:
            out[-1] = Block(last.section, f"{last.text}\n{b.text}")
        else:
            out.append(b)
    return out


def tokenize(text: str) -> list[str]:
    # "CAR-T", "IL-13Rα2", "2.7" o "blaNDM-1" quedan como un solo término. [^\W_] = cualquier
    # letra o cifra, también griegas: en biomedicina están por todas partes (TNF-α, HIF-1α).
    return re.findall(r"[^\W_]+(?:[.-][^\W_]+)*", text.lower())


def bm25_rank(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[int]:
    """Índices de `docs` del más al menos relevante para `query`.

    BM25: para cada término de la consulta suma IDF (raro en el artículo = más informativo) por
    una frecuencia saturada (k1: la 5.ª aparición vale poco más que la 2.ª) y normalizada por la
    longitud del trozo (b: un trozo largo no gana solo por ser largo). El IDF se calcula dentro
    del artículo: "CAR" no distingue nada en un paper de CAR-T, "tocilizumab" sí.
    """
    terms = [t for t in tokenize(query) if t not in STOPWORDS]
    toks = [tokenize(d) for d in docs]
    if not docs or not terms:
        return list(range(len(docs)))
    avg = sum(map(len, toks)) / len(toks) or 1.0
    df = Counter(t for ts in toks for t in set(ts))
    n = len(docs)

    def score(ts: list[str]) -> float:
        tf = Counter(ts)
        return sum(math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                   * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(ts) / avg))
                   for t in terms if t in tf)

    scores = [score(ts) for ts in toks]
    return sorted(range(n), key=lambda i: -scores[i])  # sorted es estable: empate = orden original


# Fin de frase: punto, cierre o interrogación y luego mayúscula o paréntesis. Los abstracts
# estructurados de Europe PMC pegan el título de la sección ("...P = 0.03.ResultsCrude..."),
# así que no se exige espacio. No corta tras "vs.", "e.g.", "i.e." ni "et al.".
SENTENCE_END_RE = re.compile(
    r"(?<=[.!?])(?<!\bvs\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bal\.)\s*(?=[A-Z(\[])")


def abstract_excerpt(abstract: str, query: str, limit: int) -> str:
    """Las frases del abstract que mejor encajan con la consulta, en su orden y sin pasar de
    `limit` caracteres; "…" donde se ha quitado algo (decisión 40).

    Cortar por el principio dejaba fuera los resultados: en una pregunta real sobre trasplante
    de islotes, el "78.6%" estaba en el carácter 1.046 de un abstract de 1.892."""
    if len(abstract) <= limit:
        return abstract
    sentences = [x.strip() for x in SENTENCE_END_RE.split(abstract) if x.strip()]
    keep: set[int] = set()
    used = 0
    for i in bm25_rank(query, sentences):
        if used + len(sentences[i]) + 2 > limit:
            continue  # una frase muy larga no tapa a las siguientes
        keep.add(i)
        used += len(sentences[i]) + 2
    if not keep:  # ni una frase cabe: el principio, como antes
        return abstract[:limit] + "…"
    parts, prev = [], -1
    for i in sorted(keep):
        if i != prev + 1:
            parts.append("…")
        parts.append(sentences[i])
        prev = i
    if prev != len(sentences) - 1:
        parts.append("…")
    return " ".join(parts)


def best_passages(article: Article, question: str, k: int) -> list[Block]:
    passages = chunk(article.blocks)
    order = bm25_rank(question, [f"{p.section} {p.text}" for p in passages])
    return [passages[i] for i in order[:k]]
