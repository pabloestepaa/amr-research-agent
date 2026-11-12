import json

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from amr_agent.agent import LITERATURE_MODE, Agent, AgentAnswer
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
    resp = client.post("/ask", json={"question": "colistin?"})
    assert resp.status_code == 503 and "in a minute" in resp.json()["detail"]


def test_daily_llm_limit_is_reported_as_daily(client):
    class DailyLimitedLLM:
        async def complete(self, system, user):
            request = httpx.Request("POST", "https://x")
            raise openai.RateLimitError(
                "Rate limit reached on tokens per day (TPD): Limit 200000",
                response=httpx.Response(429, request=request), body=None)

    app.dependency_overrides[get_rag] = lambda: RagService(
        FakeRetriever([make_hit("1")]), DailyLimitedLLM())
    resp = client.post("/ask", json={"question": "colistin?"})
    assert resp.status_code == 503 and "daily" in resp.json()["detail"]


def test_demo_page_is_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert '"/ask"' in resp.text and '"/agent/stream"' in resp.text
    assert resp.headers["cache-control"] == "no-cache"  # sin página vieja tras un deploy


@pytest.mark.parametrize("script_id", ["recorded", "recorded-papers"])
def test_demo_page_embeds_valid_recorded_runs(client, script_id):
    html = client.get("/").text
    start = f'<script type="application/json" id="{script_id}">'
    blob = html.split(start, 1)[1].split("</script>", 1)[0]
    recorded = json.loads(blob)
    assert recorded["question"] and recorded["recorded_on"]
    answer = AgentAnswer.model_validate(recorded["response"])  # mismo esquema que /agent
    assert answer.citations and not answer.invalid_citations
    if script_id == "recorded-papers":  # la grabación tiene que enseñar frases comprobadas
        assert all(q.verified for c in answer.citations for q in c.quotes)


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


# --- Modo literatura de /agent (pestaña "Papers") -------------------------------------------

@pytest.fixture
def literature(client):
    """Un agente de literatura con guion y cuotas reales (sin el override de la fixture)."""
    app.dependency_overrides.pop(get_agent_quota)
    app.state.agent_quota = AskQuota(per_ip_per_minute=100, per_day=100)
    app.state.literature_quota = AskQuota(per_ip_per_minute=1, per_day=100)
    llm = ScriptedLLM(text_turn("A literature answer."), text_turn("Again."))
    app.state.literature_agent = Agent(llm, Toolset([]), mode=LITERATURE_MODE)
    yield llm
    del app.state.agent_quota, app.state.literature_quota, app.state.literature_agent


def test_literature_mode_uses_its_own_agent(client, literature):
    resp = client.post("/agent", json={"question": "CAR-T dose?", "mode": "literature"})
    assert resp.status_code == 200 and resp.json()["answer"] == "A literature answer."
    system = literature.calls[0]["messages"][0]["content"]
    assert system.startswith("You are a research assistant for the biomedical literature.")


def test_literature_mode_has_its_own_quota(client, literature):
    lit = {"question": "CAR-T dose?", "mode": "literature"}
    assert client.post("/agent", json=lit).status_code == 200
    assert client.post("/agent", json=lit).status_code == 429  # 1 por minuto
    assert client.post("/agent", json={"question": "mcr-1?"}).status_code == 200  # AMR, aparte


def test_literature_mode_unavailable_is_503_without_spending_quota(client, literature):
    app.state.literature_agent = None
    lit = {"question": "CAR-T dose?", "mode": "literature"}
    assert [client.post("/agent", json=lit).status_code for _ in range(2)] == [503, 503]


def test_agent_rejects_an_unknown_mode(client):
    assert client.post("/agent", json={"question": "mcr-1?", "mode": "x"}).status_code == 422


def test_amr_is_the_default_mode_in_the_schema(client):
    schema = client.get("/openapi.json").json()["components"]["schemas"]["AgentRequest"]
    assert schema["properties"]["mode"]["default"] == "amr"


# --- /agent/stream: los pasos en directo (decisión 44) -------------------------------------

def sse_events(text: str) -> list[tuple[str, dict]]:
    """Los eventos de una respuesta SSE: (nombre, datos), sin los pings."""
    events = []
    for frame in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in frame.splitlines()
                      if line and not line.startswith(":"))
        if "event" in fields:
            events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_agent_stream_tells_each_round_and_step_then_the_same_answer(client):
    resp = client.post("/agent/stream", json={"question": "What does mcr-1 do?"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = sse_events(resp.text)
    assert [name for name, _ in events] == ["round", "step", "round", "answer"]
    assert events[0][1] == {"step": 1, "final": False}
    assert events[1][1]["tool"] == "search_literature" and events[1][1]["pmids"] == ["1001"]
    answer = AgentAnswer.model_validate(events[-1][1])  # el mismo esquema que /agent
    assert [c.pmid for c in answer.citations] == ["1001"]


def test_agent_stream_reports_llm_errors_as_an_event(client):
    class RateLimitedLLM:
        async def chat(self, messages, tools, *, tool_choice="auto"):
            request = httpx.Request("POST", "https://x")
            raise openai.RateLimitError(
                "Rate limit reached: tokens per day", body=None,
                response=httpx.Response(429, request=request))

    app.dependency_overrides[get_agent] = lambda: Agent(RateLimitedLLM(), Toolset([]))
    events = sse_events(client.post("/agent/stream", json={"question": "mcr-1?"}).text)
    assert events[-1] == ("error", {"status": 503, "detail":
                                    "LLM free-tier daily token limit reached, try again later"})


def test_agent_stream_tells_when_it_waits_for_the_llm_quota(client):
    from amr_agent.llm import RETRY_WAIT

    class SlowLLM:  # el cliente real avisa así antes de dormir (llm.py)
        async def chat(self, messages, tools, *, tool_choice="auto"):
            RETRY_WAIT.get()(23.04)
            return text_turn("mcr-1 confers colistin resistance.")

    app.dependency_overrides[get_agent] = lambda: Agent(SlowLLM(), Toolset([]))
    events = sse_events(client.post("/agent/stream", json={"question": "mcr-1?"}).text)
    assert [name for name, _ in events] == ["round", "wait", "answer"]
    assert events[1][1] == {"seconds": 23.0}


def test_agent_stream_keeps_http_errors_before_the_first_event(client):
    assert client.post("/agent/stream", json={"question": "hi"}).status_code == 422
    quota = AskQuota(per_ip_per_minute=1, per_day=100)
    app.dependency_overrides[get_agent_quota] = lambda: quota
    assert client.post("/agent/stream", json={"question": "mcr-1?"}).status_code == 200
    assert client.post("/agent/stream", json={"question": "mcr-1?"}).status_code == 429
    app.state.agent = None
    assert client.post("/agent/stream", json={"question": "mcr-1?"}).status_code == 503
