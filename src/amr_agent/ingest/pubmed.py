"""Cliente mínimo de NCBI E-utilities para descargar abstracts de PubMed.

Flujo: `esearch` (consulta -> lista de PMIDs) y después `efetch` por lotes (PMIDs -> XML).
El parseo del XML es una función pura para poder testearla sin red.
"""

import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass

import httpx

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedError(RuntimeError):
    pass


@dataclass(frozen=True)
class Article:
    pmid: str
    title: str
    abstract: str
    journal: str | None
    year: int | None

    def text_for_embedding(self) -> str:
        return f"{self.title}\n\n{self.abstract}"


def _text(el: ET.Element | None) -> str:
    # itertext() recoge también el texto de etiquetas internas como <i> o <sup>.
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def _parse_year(pub_date: ET.Element | None) -> int | None:
    if pub_date is None:
        return None
    year = _text(pub_date.find("Year"))
    if not year:  # a veces viene como <MedlineDate>2019 Nov-Dec</MedlineDate>
        match = re.search(r"\d{4}", _text(pub_date.find("MedlineDate")))
        year = match.group(0) if match else ""
    return int(year) if year.isdigit() else None


def parse_pubmed_xml(xml: str | bytes) -> list[Article]:
    """Convierte la respuesta XML de efetch en artículos; descarta los que no tienen abstract."""
    articles: list[Article] = []
    for node in ET.fromstring(xml).iter("PubmedArticle"):
        pmid = _text(node.find("MedlineCitation/PMID"))
        art = node.find("MedlineCitation/Article")
        if art is None:
            continue
        title = _text(art.find("ArticleTitle"))
        parts = []
        for section in art.findall("Abstract/AbstractText"):
            text = _text(section)
            label = section.get("Label")  # abstracts estructurados: BACKGROUND, METHODS...
            if text:
                parts.append(f"{label.capitalize()}: {text}" if label else text)
        abstract = "\n".join(parts)
        if not (pmid and title and abstract):
            continue
        articles.append(
            Article(
                pmid=pmid,
                title=title,
                abstract=abstract,
                journal=_text(art.find("Journal/Title")) or None,
                year=_parse_year(art.find("Journal/JournalIssue/PubDate")),
            )
        )
    return articles


class PubMedClient:
    def __init__(
        self,
        http: httpx.Client,
        api_key: str | None = None,
        email: str | None = None,
        tool: str = "amr-research-agent",
    ) -> None:
        self._http = http
        self._base = {"tool": tool, **({"email": email} if email else {})}
        if api_key:
            self._base["api_key"] = api_key
        # NCBI permite 3 peticiones/s sin key y 10/s con key.
        self._delay = 0.11 if api_key else 0.34

    def _request(self, method: str, endpoint: str, data: dict) -> httpx.Response:
        params = {**self._base, **data}
        for attempt in range(4):
            if method == "GET":
                resp = self._http.get(f"{EUTILS}/{endpoint}", params=params)
            else:
                resp = self._http.post(f"{EUTILS}/{endpoint}", data=params)
            if resp.status_code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(2**attempt)  # backoff exponencial: 1 s, 2 s, 4 s
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError("unreachable")

    def search(self, term: str, max_results: int) -> list[str]:
        params = {"db": "pubmed", "term": term, "retmax": max_results, "retmode": "json",
                  "sort": "relevance"}
        for attempt in range(4):
            result = self._request("GET", "esearch.fcgi", params).json()["esearchresult"]
            # NCBI a veces responde 200 con {"ERROR": "..."} cuando su backend falla.
            if "idlist" in result:
                return result["idlist"]
            if attempt < 3:
                time.sleep(2 ** (attempt + 1))
        raise PubMedError(f"esearch failed: {result.get('ERROR', result)}")

    def fetch(self, pmids: list[str]) -> list[Article]:
        # POST en lugar de GET: con cientos de IDs la URL sería demasiado larga.
        resp = self._request(
            "POST",
            "efetch.fcgi",
            {"db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "xml"},
        )
        return parse_pubmed_xml(resp.content)

    def iter_articles(self, term: str, max_results: int, batch: int = 200) -> Iterator[Article]:
        pmids = self.search(term, max_results)
        for i in range(0, len(pmids), batch):
            time.sleep(self._delay)
            yield from self.fetch(pmids[i : i + batch])
