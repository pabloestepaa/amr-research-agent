"""Dobles de test: sustituyen al modelo de embeddings, al LLM y al buscador.

Así los tests son rápidos, deterministas y no necesitan red ni API keys.
"""

import hashlib
import re

import numpy as np

from amr_agent.llm import Completion
from amr_agent.retrieval import Hit


class HashingEmbedder:
    """Embedding "de juguete": bolsa de palabras proyectada en 384 dimensiones con un hash.

    No entiende sinónimos, pero textos que comparten palabras quedan cerca, que es
    suficiente para comprobar que el SQL de búsqueda vectorial funciona.
    """

    dim = 384

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = np.linalg.norm(v)
        if norm == 0:  # evitamos el vector cero: su distancia coseno no está definida
            v[0], norm = 1.0, 1.0
        return v / norm

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system: str, user: str) -> Completion:
        self.calls.append((system, user))
        return Completion(self.reply, "fake-model", 100, 20, 1.0)


class FakeRetriever:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int, str]] = []

    async def search(self, query: str, k: int, mode: str = "hybrid") -> list[Hit]:
        self.calls.append((query, k, mode))
        return self.hits[:k]


def make_hit(pmid: str, title: str = "A title", abstract: str = "An abstract.") -> Hit:
    return Hit(pmid=pmid, title=title, abstract=abstract, journal="J Test", year=2024, score=0.5)
