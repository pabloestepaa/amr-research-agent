import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from amr_agent.api.main import app, get_rag, get_retriever
from amr_agent.rag import RagService
from tests.fakes import FakeLLM, FakeRetriever, make_hit


@pytest.fixture
def client():
    # Sin `with TestClient(...)` no se ejecuta el lifespan: no hace falta BD ni modelo.
    retriever = FakeRetriever([make_hit("1001", title="mcr-1")])
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_rag] = lambda: RagService(retriever, FakeLLM("Yes [PMID:1001]."))
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_answer_with_citations(client):
    resp = client.post("/ask", json={"question": "What does mcr-1 do?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["citations"][0]["pmid"] == "1001"
    assert body["usage"]["model"] == "fake-model"


@pytest.mark.parametrize("payload", [
    {"question": "hi"},  # demasiado corta
    {"question": "valid question", "k": 50},  # k fuera de rango
    {"question": "valid question", "mode": "magic"},  # modo inexistente
])
def test_ask_validates_input(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_search_does_not_need_llm(client):
    resp = client.get("/search", params={"q": "colistin", "k": 1})
    assert resp.status_code == 200
    assert resp.json()[0]["pmid"] == "1001"


def test_rate_limit_from_llm_becomes_503(client):
    class RateLimitedLLM:
        async def complete(self, system, user):
            request = httpx.Request("POST", "https://x")
            raise openai.RateLimitError(
                "slow down", response=httpx.Response(429, request=request),
                body=None)

    app.dependency_overrides[get_rag] = lambda: RagService(
        FakeRetriever([make_hit("1")]), RateLimitedLLM())
    assert client.post("/ask", json={"question": "colistin?"}).status_code == 503
