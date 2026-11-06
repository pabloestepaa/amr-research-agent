import json

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from amr_agent.agent import Agent, AgentAnswer
from amr_agent.api.main import app, get_agent, get_agent_quota, get_quota, get_rag, get_retriever
from amr_agent.api.ratelimit import AskQuota
from amr_agent.rag import RagService
from amr_agent.tools import LiteratureTools, Toolset
from tests.fakes import FakeLLM, FakeRetriever, ScriptedLLM, make_hit, text_turn, tool_turn


@pytest.fixture
def client():
    # Sin `with TestClient(...)` no se ejecuta el lifespan: no hace falta BD ni modelo.
    retriever = FakeRetriever([make_hit("1001", title="mcr-1")])
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_rag] = lambda: RagService(retriever, FakeLLM("Yes [PMID:1001]."))
    quota = AskQuota(per_ip_per_minute=100, per_day=100)
    app.dependency_overrides[get_quota] = lambda: quota
    agent_quota = AskQuota(per_ip_per_minute=100, per_day=100)
    app.dependency_overrides[get_agent_quota] = lambda: agent_quota
    llm = ScriptedLLM(tool_turn(("search_literature", {"query": "mcr-1"})),
                      text_turn("mcr-1 confers colistin resistance [PMID:1001]."))
    app.state.agent = Agent(llm, Toolset([LiteratureTools(retriever)]))
    yield TestClient(app)
    app.dependency_overrides.clear()
    del app.state.agent


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


def test_demo_page_is_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert '"/ask"' in resp.text and '"/agent"' in resp.text
    assert resp.headers["cache-control"] == "no-cache"  # sin página vieja tras un deploy


def test_demo_page_embeds_a_valid_recorded_agent_run(client):
    html = client.get("/").text
    start = '<script type="application/json" id="recorded">'
    blob = html.split(start, 1)[1].split("</script>", 1)[0]
    recorded = json.loads(blob)
    assert recorded["question"] and recorded["recorded_on"]
    AgentAnswer.model_validate(recorded["response"])  # mismo esquema que devuelve /agent


def test_ask_is_rate_limited_per_ip(client):
    quota = AskQuota(per_ip_per_minute=2, per_day=100)
    app.dependency_overrides[get_quota] = lambda: quota
    codes = [client.post("/ask", json={"question": "colistin?"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    resp = client.post("/ask", json={"question": "colistin?"})
    assert int(resp.headers["retry-after"]) > 0


def test_agent_returns_answer_and_trajectory(client):
    resp = client.post("/agent", json={"question": "What does mcr-1 do?"})
    assert resp.status_code == 200
    body = resp.json()
    assert [s["tool"] for s in body["steps"]] == ["search_literature"]
    assert body["steps"][0]["arguments"] == {"query": "mcr-1"}
    assert [c["pmid"] for c in body["citations"]] == ["1001"]
    assert body["stop_reason"] == "answer"
    assert body["usage"]["llm_calls"] == 2


def test_agent_validates_input(client):
    assert client.post("/agent", json={"question": "hi"}).status_code == 422


def test_agent_unavailable_is_503_and_does_not_spend_quota(client):
    app.state.agent = None  # amr-mcp no arrancó
    quota = AskQuota(per_ip_per_minute=1, per_day=100)
    app.dependency_overrides[get_agent_quota] = lambda: quota
    codes = [client.post("/agent", json={"question": "mcr-1?"}).status_code for _ in range(3)]
    assert codes == [503, 503, 503]  # nunca 429: no ha gastado cuota
    assert client.post("/ask", json={"question": "mcr-1?"}).status_code == 200  # /ask sigue


def test_agent_has_its_own_quota(client):
    quota = AskQuota(per_ip_per_minute=1, per_day=100)
    app.dependency_overrides[get_agent_quota] = lambda: quota
    assert client.post("/agent", json={"question": "mcr-1?"}).status_code == 200
    assert client.post("/agent", json={"question": "mcr-1?"}).status_code == 429
    assert client.post("/ask", json={"question": "mcr-1?"}).status_code == 200


def test_agent_llm_rate_limit_becomes_503(client):
    class RateLimitedLLM:
        async def chat(self, messages, tools, *, tool_choice="auto"):
            request = httpx.Request("POST", "https://x")
            raise openai.RateLimitError(
                "slow down", response=httpx.Response(429, request=request), body=None)

    app.dependency_overrides[get_agent] = lambda: Agent(RateLimitedLLM(), Toolset([]))
    assert client.post("/agent", json={"question": "mcr-1?"}).status_code == 503
