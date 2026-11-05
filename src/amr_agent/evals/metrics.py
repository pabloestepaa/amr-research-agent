"""Métricas de recuperación con relevancia binaria (un abstract es relevante o no).

Son funciones puras de pocas líneas a propósito: se entienden, se testean y se pueden
explicar en una entrevista. Ragas o DeepEval calculan cosas parecidas, pero son una caja
negra más grande de lo que este proyecto necesita.
"""

import math
from collections.abc import Sequence


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Qué fracción de los documentos relevantes aparece entre los k primeros."""
    if not relevant:
        raise ValueError("recall is undefined without relevant documents")
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    """1 / posición del primer relevante (0 si no aparece). Su media es el MRR."""
    for position, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """nDCG@k: premia que los relevantes estén arriba; 1.0 es la ordenación perfecta.

    DCG = Σ rel_i / log2(i + 1). Se divide por el DCG ideal (todos los relevantes primero)
    para que el valor quede entre 0 y 1 sea cual sea el número de relevantes.
    """
    dcg = sum(1.0 / math.log2(i + 1) for i, doc in enumerate(ranked[:k], start=1)
              if doc in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
