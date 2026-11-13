"""Abstención del agente: el modelo escribe una frase fija y el código la convierte en un mensaje
para el usuario (decisión 48).

El modelo sigue respondiendo con la frase exacta en inglés que pide el prompt (NO_EVIDENCE o
NO_EVIDENCE_PAPERS): es fácil de detectar y el prompt no cambia. Pero esa frase sola tenía tres
problemas: salía en inglés aunque la pregunta fuera en español, decía "no encontré evidencia"
también cuando no había buscado nada (la capital de Australia) y no contaba qué había mirado.
Aquí se distinguen los dos casos por la trayectoria (fuera de ámbito = ninguna herramienta; en
las evals, 5 de 5 con 0 llamadas frente a 4 de 4 con 3 o más) y se escribe el mensaje en el
idioma de la pregunta, con las búsquedas y lecturas que hizo de verdad.
"""

import re
from typing import Literal

from amr_agent.rag import NO_EVIDENCE, NO_EVIDENCE_PAPERS

AbstentionKind = Literal["out_of_scope", "not_found"]

# La misma detección que la demo (index.html, `SPANISH`): signos y tildes del español o palabras
# frecuentes en las preguntas. Solo español e inglés: cualquier otro idioma recibe el inglés.
SPANISH_RE = re.compile(
    r"[¿¡ñáéíóú]|\b(qué|cuál|cómo|dónde|cuántos?|cuántas?|genes? de|resistencia|especies?|"
    r"bacterias|pacientes|porcentaje|tratamiento)\b", re.I)

NO_EVIDENCE_ES = {
    NO_EVIDENCE: "No he encontrado evidencia sobre esto en la literatura indexada.",
    NO_EVIDENCE_PAPERS: "No he encontrado evidencia sobre esto en los artículos que encontré.",
}


def is_spanish(text: str) -> bool:
    return bool(SPANISH_RE.search(text))


def is_abstention(answer: str) -> bool:
    """La respuesta contiene la frase de abstención de algún modo (sin el punto final)."""
    return any(m.rstrip(".").lower() in answer.lower() for m in (NO_EVIDENCE, NO_EVIDENCE_PAPERS))


def _join(items: list[str], spanish: bool) -> str:
    if len(items) < 2:
        return "".join(items)
    return f"{', '.join(items[:-1])} {'y' if spanish else 'and'} {items[-1]}"


def abstention_message(kind: AbstentionKind, *, question: str, topic: tuple[str, str],
                       no_evidence: str, searched: list[str], read: list[str]) -> str:
    """El mensaje para el usuario. `topic` es el ámbito del modo en inglés y en español;
    `searched` y `read`, las consultas y los artículos de las llamadas que funcionaron."""
    es = is_spanish(question)
    if kind == "out_of_scope":
        if es:
            return (f"Solo respondo preguntas sobre {topic[1]}, así que no he buscado nada sobre "
                    "esto.")
        return f"I only answer questions about {topic[0]}, so I did not search for this."
    base = NO_EVIDENCE_ES.get(no_evidence, no_evidence) if es else no_evidence
    queries = [f"«{q}»" if es else f'"{q}"' for q in searched]
    papers = [f"PMID {p}" if p.isdigit() else p for p in read]
    if es:
        did = ([f"Busqué {_join(queries, True)}"] if queries else []) + (
            [f"leí el texto completo de {_join(papers, True)}"] if papers else [])
        tail = " y ".join(did)
        return f"{base} {tail[0].upper()}{tail[1:]}, pero nada de eso responde a la pregunta." \
            if did else base
    did = ([f"I searched for {_join(queries, False)}"] if queries else []) + (
        [f"read the full text of {_join(papers, False)}"] if papers else [])
    tail = " and ".join(did)
    return f"{base} {tail[0].upper()}{tail[1:]}, but none of it answers the question." \
        if did else base
