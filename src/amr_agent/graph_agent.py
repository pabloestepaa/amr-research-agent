"""El mismo agente que agent.py, con el bucle hecho en LangGraph (bloque C de la fase 8).

Mismas herramientas, prompts, límites y reglas: el grafo usa las piezas de `Agent` (`_start`,
`_must_answer`, `_explore_turn`, `_final_turn`, `_run_tools`, `_answer`). Lo único que cambia es
quién decide el orden: en agent.py un `for` con `if`; aquí un grafo de nodos y flechas.

         ┌──────────── next_round ────────────┐
         │                                    ▼
    START ─► next_round ─► explore ─┬─ pidió herramientas ─► tools ─┘
                  │                 └─ respondió ─────────────────────► END
                  └─ límite (pasos o tokens) ─► final ──────────────────► END

LangGraph no va en la imagen de Docker (es una dependencia de desarrollo): se usa en la eval
(`amr-eval-agent --mode literature --engine graph`) para comparar las dos versiones.
"""

import time
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from amr_agent.agent import Agent, AgentAnswer, EventSink, Progress, _ignore


class State(TypedDict):
    """El estado que recorre el grafo: la vuelta en la que va y lo acumulado (`Progress`)."""

    step: int
    progress: Progress


class GraphAgent(Agent):
    """`Agent` con `_run` en LangGraph. `run()`, las trazas y la memoria de lectura son las de
    `Agent`, así que la API y la eval lo usan igual."""

    async def _run(self, question: str, emit: EventSink) -> AgentAnswer:
        t0 = time.perf_counter()
        state = await self.graph(question, emit).ainvoke(
            {"step": 1, "progress": self._start(question)},
            # Cada vuelta son 2 nodos (explore + tools); LangGraph corta a los 25 por defecto.
            {"recursion_limit": 2 * self.max_steps + 2},
        )
        return self._answer(question, state["progress"], t0)

    def graph(self, question: str = "", emit: EventSink = _ignore) -> CompiledStateGraph:
        """El grafo de una pregunta, listo para ejecutar. Para verlo dibujado:
        `GraphAgent(llm, tools).graph().get_graph().draw_mermaid()` (o `draw_ascii()` con grandalf).
        """
        schemas = [s.to_openai() for s in self._tools.specs()]

        # Nodos: cada uno hace una cosa y devuelve qué cambia del estado.
        async def explore(state: State) -> dict[str, Any]:
            emit("round", {"step": state["step"], "final": False})
            await self._explore_turn(state["progress"], schemas)
            return {}

        async def tools(state: State) -> dict[str, Any]:
            p = state["progress"]
            await self._run_tools(p.turns[-1], state["step"], p, emit)
            return {"step": state["step"] + 1}

        async def final(state: State) -> dict[str, Any]:
            p = state["progress"]
            p.stop = self._must_answer(question, p, state["step"]) or "answer"
            emit("round", {"step": state["step"], "final": True})
            await self._final_turn(question, p)
            return {}

        # Flechas condicionales: miran el estado y eligen el siguiente nodo (sin cambiar nada).
        def next_round(state: State) -> Literal["explore", "final"]:
            if self._must_answer(question, state["progress"], state["step"]):
                return "final"
            return "explore"

        def after_explore(state: State) -> Literal["tools", "__end__"]:
            return "tools" if state["progress"].turns[-1].tool_calls else END

        graph = StateGraph(State)
        graph.add_node("explore", explore)
        graph.add_node("tools", tools)
        graph.add_node("final", final)
        graph.add_conditional_edges(START, next_round)
        graph.add_conditional_edges("explore", after_explore)
        graph.add_conditional_edges("tools", next_round)
        graph.add_edge("final", END)
        return graph.compile()
