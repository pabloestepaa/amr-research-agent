"""Dobles de test: sustituyen al modelo de embeddings, al LLM y al buscador.

Así los tests son rápidos, deterministas y no necesitan red ni API keys.
"""

import copy
import hashlib
import json
import re

import numpy as np

from amr_agent.llm import ChatTurn, Completion, ToolCall
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
    """Devuelve siempre `reply`; si se le da una lista, una respuesta por llamada."""

    def __init__(self, reply: str | list[str]) -> None:
        self.replies = [reply] if isinstance(reply, str) else list(reply)
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system: str, user: str, *, json_mode: bool = False) -> Completion:
        self.calls.append((system, user))
        reply = self.replies[min(len(self.calls), len(self.replies)) - 1]
        return Completion(reply, "fake-model", 100, 20, 1.0)


class FakeRetriever:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int, str]] = []

    async def search(self, query: str, k: int, mode: str = "hybrid") -> list[Hit]:
        self.calls.append((query, k, mode))
        return self.hits[:k]


def make_hit(pmid: str, title: str = "A title", abstract: str = "An abstract.") -> Hit:
    return Hit(pmid=pmid, title=title, abstract=abstract, journal="J Test", year=2024, score=0.5)


def tool_turn(*calls: tuple[str, dict | str], tokens: int = 100) -> ChatTurn:
    """Turno en el que el modelo pide herramientas. Un str como argumentos = JSON en bruto."""
    return ChatTurn(
        text="",
        tool_calls=[ToolCall.parse(f"call_{i}", name, args if isinstance(args, str)
                                   else json.dumps(args)) for i, (name, args) in enumerate(calls)],
        model="fake-model", prompt_tokens=tokens, completion_tokens=10, latency_ms=1.0,
    )


def text_turn(text: str, tokens: int = 100) -> ChatTurn:
    return ChatTurn(text=text, model="fake-model", prompt_tokens=tokens, completion_tokens=10,
                    latency_ms=1.0)


class ScriptedLLM:
    """LLM con guion: devuelve los turnos en orden y guarda lo que recibió en cada llamada.

    Prueba el bucle del agente (no al modelo): qué mensajes se le envían, cuándo se le prohíben
    las herramientas, qué pasa si pide algo raro...
    """

    def __init__(self, *turns: ChatTurn) -> None:
        self.turns = list(turns)
        self.calls: list[dict] = []

    async def chat(self, messages, tools, *, tool_choice="auto") -> ChatTurn:
        self.calls.append({"messages": copy.deepcopy(messages), "tools": tools,
                           "tool_choice": tool_choice})
        return self.turns[min(len(self.calls), len(self.turns)) - 1]
