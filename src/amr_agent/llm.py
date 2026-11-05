"""Cliente de LLM agnóstico del proveedor.

Groq, Gemini, Ollama, OpenRouter... casi todos exponen una API compatible con la de OpenAI.
Usando el SDK de OpenAI con otra `base_url`, cambiar de proveedor es cambiar dos variables
de entorno, sin tocar código.
"""

import time
from dataclasses import dataclass
from typing import Protocol

from openai import AsyncOpenAI


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float


class LLMClient(Protocol):
    async def complete(self, system: str, user: str, *, json_mode: bool = False) -> Completion:
        ...


class OpenAICompatibleLLM:
    def __init__(
        self, base_url: str, api_key: str, model: str, timeout_s: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        # El SDK reintenta solo los 429 y 5xx respetando la cabecera `retry-after`. En la API
        # bastan 2 reintentos; las evals usan más porque mandan muchas peticiones seguidas.
        self._client = AsyncOpenAI(
            base_url=base_url, api_key=api_key or "missing", timeout=timeout_s,
            max_retries=max_retries,
        )
        self.model = model

    async def complete(self, system: str, user: str, *, json_mode: bool = False) -> Completion:
        start = time.perf_counter()
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = await self._client.chat.completions.create(
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
