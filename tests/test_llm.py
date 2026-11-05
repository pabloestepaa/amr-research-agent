"""Tests del cliente del LLM contra respuestas HTTP simuladas (sin red ni claves).

El SDK de OpenAI 3.x usa httpx2 (no httpx), así que respx no lo intercepta: le pasamos un cliente
httpx2 con un MockTransport que devuelve las respuestas que queremos, en orden.
"""

import time

import httpx2 as httpx
import openai
import pytest

from amr_agent.llm import OpenAICompatibleLLM

BASE = "https://llm.test/v1"


def ok(content: str = "hi", tool_calls: list | None = None) -> httpx.Response:
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return httpx.Response(200, json={
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    })


def rate_limited(retry_after: str) -> httpx.Response:
    return httpx.Response(429, headers={"retry-after": retry_after},
                          json={"error": {"message": "slow down"}})


class Server:
    """Simula el proveedor: devuelve las respuestas en orden (la última se repite)."""

    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.call_count = 0

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.call_count += 1
        return self.responses[min(self.call_count, len(self.responses)) - 1]

    def llm(self, **kwargs) -> OpenAICompatibleLLM:
        client = httpx.AsyncClient(transport=httpx.MockTransport(self.handle))
        return OpenAICompatibleLLM(BASE, "test-key", "m", http_client=client, **kwargs)


async def test_retries_when_the_server_asks_for_a_short_wait():
    server = Server(rate_limited("0.01"), ok("hello"))
    result = await server.llm(max_retry_wait_s=5).complete("sys", "user")
    assert result.text == "hello"
    assert server.call_count == 2


async def test_fails_fast_when_the_server_asks_for_a_long_wait():
    # Con la cuota diaria de Groq agotada, Retry-After es de minutos: la API no debe quedarse
    # esperando, debe fallar ya (y el endpoint lo convierte en un 503).
    server = Server(rate_limited("90"))
    start = time.perf_counter()
    with pytest.raises(openai.RateLimitError):
        await server.llm(max_retry_wait_s=5).complete("sys", "user")
    assert time.perf_counter() - start < 1
    assert server.call_count == 1


async def test_gives_up_after_max_retries():
    server = Server(rate_limited("0"))
    with pytest.raises(openai.RateLimitError):
        await server.llm(max_retries=2).complete("sys", "user")
    assert server.call_count == 3  # el intento inicial + 2 reintentos


async def test_client_errors_are_not_retried():
    server = Server(httpx.Response(400, json={"error": {}}))
    with pytest.raises(openai.BadRequestError):
        await server.llm().complete("sys", "user")
    assert server.call_count == 1


async def test_chat_parses_tool_calls_and_keeps_raw_arguments():
    server = Server(ok("", tool_calls=[
        {"id": "c1", "type": "function",
         "function": {"name": "get_amr_gene", "arguments": '{"name": "mcr-1"}'}},
        {"id": "c2", "type": "function", "function": {"name": "x", "arguments": "{oops"}},
    ]))
    turn = await server.llm().chat([{"role": "user", "content": "q"}], tools=[])
    assert [c.name for c in turn.tool_calls] == ["get_amr_gene", "x"]
    assert turn.tool_calls[0].arguments == {"name": "mcr-1"}
    assert turn.tool_calls[1].arguments is None  # JSON roto: el agente responderá con un error
    msg = turn.as_message()
    assert msg["content"] is None
    assert msg["tool_calls"][1]["function"]["arguments"] == "{oops"  # se reenvía tal cual
