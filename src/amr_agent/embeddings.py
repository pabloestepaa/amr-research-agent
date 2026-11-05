"""Embeddings: convertir texto en vectores para la búsqueda semántica."""

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[np.ndarray]: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedEmbedder:
    """Modelo local (ONNX, sin GPU) a través de fastembed.

    bge-small distingue entre documentos y consultas: `query_embed` añade el prefijo de
    instrucción que el modelo espera para las preguntas, y eso mejora la recuperación.
    """

    def __init__(self, model_name: str, threads: int | None = None) -> None:
        from fastembed import TextEmbedding  # import perezoso: los tests no lo cargan

        self._model = TextEmbedding(model_name, threads=threads)

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return list(self._model.embed(texts))

    def embed_query(self, text: str) -> np.ndarray:
        return next(iter(self._model.query_embed(text)))
