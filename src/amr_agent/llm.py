"""Cliente de LLM agnóstico del proveedor.

Groq, Gemini, Ollama, OpenRouter... casi todos exponen una API compatible con la de OpenAI.
Usando el SDK de OpenAI con otra `base_url`, cambiar de proveedor es cambiar dos variables
de entorno, sin tocar código.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import openai
from openai import AsyncOpenAI


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    raw_arguments: str  # tal cual los escribió el modelo: hay que reenviarlos así en el historial
    arguments: dict[str, Any] | None  # None si no eran un objeto JSON válido

    @classmethod
    def parse(cls, id: str, name: str, raw: str | None) -> "ToolCall":
        try:
            args = json.loads(raw or "{}")
        except json.JSONDecodeError:
            args = None
        return cls(id, name, raw or "{}", args if isinstance(args, dict) else None)


@dataclass(frozen=True)
class ChatTurn:
    """Una respuesta del modelo en una conversación con herramientas."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float = 0.0

    def as_message(self) -> dict[str, Any]:
        """El mensaje `assistant` que hay que añadir al historial antes de los resultados.

        Lo construimos a mano en vez de reenviar el objeto del SDK: así solo viajan los campos
        estándar (gpt-oss añade otros, como su razonamiento, que no hace falta reenviar).
        """
        msg: dict[str, Any] = {"role": "assistant", "content": self.text or None}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name, "arguments": c.raw_arguments}}
                for c in self.tool_calls
            ]
        return msg


ToolChoice = Literal["auto", "none"]


class LLMClient(Protocol):
    async def complete(self, system: str, user: str, *, json_mode: bool = False) -> Completion:
        ...


class ChatLLM(Protocol):
    """Lo que necesita el agente: conversación con varios mensajes y herramientas."""

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                   *, tool_choice: ToolChoice = "auto") -> ChatTurn:
        ...


RETRYABLE = (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError)


def retry_delay(exc: Exception, attempt: int) -> float:
    """Lo que pide el servidor en `Retry-After` o, si no dice nada, espera exponencial."""
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            return max(0.0, float(response.headers.get("retry-after", "")))
        except ValueError:
            pass
    return min(0.5 * 2**attempt, 8.0)


class OpenAICompatibleLLM:
    def __init__(
        self, base_url: str, api_key: str, model: str, timeout_s: float = 30.0,
        max_retries: int = 2, max_retry_wait_s: float = 120.0, http_client: Any = None,
    ) -> None:
        # Los reintentos los hacemos nosotros (ver `_create`), no el SDK: el SDK respeta un
        # `Retry-After` de hasta 2 minutos, que está bien en las evals pero no en una petición web.
        self._client = AsyncOpenAI(
            base_url=base_url, api_key=api_key or "missing", timeout=timeout_s, max_retries=0,
            http_client=http_client,  # solo en tests: un cliente con respuestas simuladas
        )
        self.model = model
        self.max_retries = max_retries
        self.max_retry_wait_s = max_retry_wait_s

    async def _create(self, **kwargs: Any) -> Any:
        """Llama al proveedor reintentando 429, 5xx y errores de red, pero solo si compensa.

        Si el servidor pide esperar más de `max_retry_wait_s` (con la cuota diaria de Groq
        agotada pide minutos), fallamos ya: en la API es mejor un 503 inmediato que un visitante
        mirando "Working..." durante minutos para recibir el mismo error. Las evals, en cambio,
        pueden permitirse esperar.
        """
        attempt = 0
        while True:
            try:
                return await self._client.chat.completions.create(**kwargs)
            except RETRYABLE as exc:
                wait = retry_delay(exc, attempt)
                if attempt >= self.max_retries or wait > self.max_retry_wait_s:
                    raise
                await asyncio.sleep(wait)
                attempt += 1

    async def complete(self, system: str, user: str, *, json_mode: bool = False) -> Completion:
        start = time.perf_counter()
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = await self._create(
            model=self.model,
            temperature=0.0,  # respuestas lo más deterministas posible: mejor para evaluar
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **extra,
        )
        usage = resp.usage
        return Completion(
            text=(resp.choices[0].message.content or "").strip(),
            model=resp.model,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                   *, tool_choice: ToolChoice = "auto") -> ChatTurn:
        # Sin herramientas no enviamos `tools` ni `tool_choice`: es una conversación normal. Así
        # pide el agente la respuesta final (ver `Agent._final_messages`).
        start = time.perf_counter()
        extra = {"tools": tools, "tool_choice": tool_choice} if tools else {}
        resp = await self._create(model=self.model, temperature=0.0, messages=messages, **extra)
        msg = resp.choices[0].message
        usage = resp.usage
        return ChatTurn(
            text=(msg.content or "").strip(),
            tool_calls=[ToolCall.parse(tc.id, tc.function.name, tc.function.arguments)
                        for tc in msg.tool_calls or []],
            model=resp.model,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
