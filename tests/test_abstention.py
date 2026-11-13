"""Abstención (decisión 48): el modelo escribe la frase fija y el código la convierte en un
mensaje en el idioma de la pregunta, distinguiendo "fuera de ámbito" de "no encontrado"."""

import pytest

from amr_agent.abstention import is_spanish
from amr_agent.agent import AMR_MODE, LITERATURE_MODE, Agent
from amr_agent.evals.agent_eval import score_task
from amr_agent.evals.dataset import AgentTask
from amr_agent.rag import NO_EVIDENCE_PAPERS
from amr_agent.tools import PaperSearchTools, ReadPaperTools, Toolset
from tests.fakes import ScriptedLLM, text_turn, tool_turn
from tests.test_eval_agent import answer
from tests.test_literature_mode import epmc_record, network


@pytest.mark.parametrize("question, spanish", [
    ("¿Cuál es la capital de Australia?", True),
    ("Qué porcentaje de pacientes deja de necesitar insulina", True),  # sin signos ni tildes
    ("Which bacteria carry blaKPC-2?", False),
    ("What share of patients needed tocilizumab in ZUMA-1?", False),
])
def test_spanish_detection_matches_the_demo(question, spanish):
    assert is_spanish(question) is spanish


@pytest.mark.parametrize("question, expected", [
    ("¿Cuál es la capital de Australia?",
     "Solo respondo preguntas sobre investigación biomédica y de ciencias de la vida, así que no "
     "he buscado nada sobre esto."),
    ("Write a Python function that parses a FASTA file.",
     "I only answer questions about biomedical and life-science research, so I did not search "
     "for this."),
])
async def test_out_of_scope_says_so_in_the_question_language(question, expected):
    # En la eval, "¿Cuál es la capital de Australia?" recibía "I could not find evidence for
    # this in the articles I found": en inglés y como si hubiera buscado.
    llm = ScriptedLLM(text_turn(NO_EVIDENCE_PAPERS))
    result = await Agent(llm, Toolset([]), mode=LITERATURE_MODE).run(question)
    assert result.answer == expected
    assert (result.abstained, result.abstention) == (True, "out_of_scope")


async def test_not_found_tells_what_was_searched_and_read():
    http, _ = network([epmc_record("11111111")])
    tools = Toolset([PaperSearchTools(http), ReadPaperTools(http)])
    llm = ScriptedLLM(
        tool_turn(("search_papers", {"query": "chronic GVHD incidence"})),
        tool_turn(("read_paper", {"paper_id": "11111111", "question": "incidence"}),
                  ("read_paper", {"paper_id": "abc", "question": "incidence"})),  # falla: no sale
        text_turn(NO_EVIDENCE_PAPERS))
    result = await Agent(llm, tools, mode=LITERATURE_MODE).run(
        "¿Con qué frecuencia aparece la EICH crónica?")
    assert result.answer == (
        "No he encontrado evidencia sobre esto en los artículos que encontré. Busqué «chronic "
        "GVHD incidence» y leí el texto completo de PMID 11111111, pero nada de eso responde a "
        "la pregunta.")
    assert (result.abstained, result.abstention) == (True, "not_found")


async def test_in_english_the_message_still_starts_with_the_fixed_sentence():
    # Así las respuestas guardadas de antes y las de ahora se comparan igual.
    http, _ = network([epmc_record("1")])
    llm = ScriptedLLM(tool_turn(("search_papers", {"query": "a b"})),
                      tool_turn(("search_papers", {"query": "c d"})),
                      text_turn(NO_EVIDENCE_PAPERS))
    tools = Toolset([PaperSearchTools(http)])
    result = await Agent(llm, tools, mode=LITERATURE_MODE).run("What dose was used?")
    assert result.answer == (f'{NO_EVIDENCE_PAPERS} I searched for "a b" and "c d", but none of it '
                             "answers the question.")


async def test_a_partial_answer_that_mentions_the_sentence_is_kept():
    http, _ = network([epmc_record("1")])
    text = f"Response was 40% [PMID:1]. {NO_EVIDENCE_PAPERS}"
    llm = ScriptedLLM(tool_turn(("search_papers", {"query": "x y"})), text_turn(text))
    result = await Agent(llm, Toolset([PaperSearchTools(http)]), mode=LITERATURE_MODE).run("?")
    assert result.answer == text and result.abstained  # cuenta igual que antes en las evals


def test_the_eval_reads_the_field_whatever_the_language():
    oos = AgentTask(id="oos", category="out_of_scope", question="¿Capital de Australia?",
                    should_abstain=True)
    spanish = answer("Solo respondo preguntas sobre investigación biomédica.", [])
    assert not score_task(oos, spanish)["success"]  # sin el campo, no se reconoce...
    spanish.abstained, spanish.abstention = True, "out_of_scope"
    assert score_task(oos, spanish)["success"]  # ...con el campo, sí


def test_each_mode_has_its_scope_in_both_languages():
    assert AMR_MODE.topic[0] == "antimicrobial resistance research"
    assert LITERATURE_MODE.topic[1].startswith("investigación biomédica")
