from datetime import date
from pathlib import Path

import pytest
from ask_examples import EDGES, NODES, NOT_LINEAGE
from helpers import TEST_PROJECT

from asklineage.answer.history import (
    HistorySentence,
    HistorySummary,
    Retrieved,
    answer_history,
    build_prompt,
    check_summary,
)
from asklineage.api.flow import run_ask
from asklineage.api.models import AskRequest, HistoryResponse
from asklineage.corpus.docs import Corpus, Document
from asklineage.incidents.chunks import spellings
from asklineage.incidents.store import index_incidents
from asklineage.llm.scripted import ScriptedProvider

INCIDENTS = (
    Retrieved(
        "inc-001",
        "Orders went missing",
        (("Root cause", "raw.orders stopped loading."), ("Fix", "Re-ran the load for stg_orders.")),
    ),
    Retrieved("inc-002", "Duplicate payments", (("Fix", "Added a dedup to fct_payments."),)),
)
KNOWN = spellings(
    [
        ("raw.orders", "source.p.raw.orders"),
        ("stg_orders", "model.p.stg_orders"),
        ("fct_payments", "model.p.fct_payments"),
    ]
)


def summary(*sentences: tuple[str, list[str]], answered: bool = True) -> HistorySummary:
    return HistorySummary(
        answered=answered,
        sentences=[HistorySentence(text=text, cites=cites) for text, cites in sentences],
    )


def rules(result: HistorySummary) -> list[str]:
    return [v.rule for v in check_summary(result, INCIDENTS, KNOWN)]


def test_a_cited_grounded_summary_passes():
    good = summary(("raw.orders stopped, so stg_orders was re-run (inc-001).", ["inc-001"]))
    assert rules(good) == []


def test_saying_nothing_answers_passes_only_without_sentences():
    assert rules(summary(answered=False)) == []
    assert rules(summary(("Something.", ["inc-001"]), answered=False)) == [
        "sentences_when_unanswered"
    ]
    assert rules(summary()) == ["no_sentences"]


@pytest.mark.parametrize(
    ("sentence", "rule"),
    [
        (("A load was re-run.", []), "no_citation"),
        (("A load was re-run.", ["inc-404"]), "unknown_citation"),
        (("Like inc-002, a load was re-run.", ["inc-001"]), "uncited_incident"),
        (("fct_payments was re-run.", ["inc-001"]), "node_not_in_citation"),
        (("   ", ["inc-001"]), "empty_sentence"),
    ],
)
def test_each_rule(sentence, rule):
    assert rules(summary(sentence)) == [rule]


def test_a_node_is_allowed_when_any_cited_incident_names_it():
    both = summary(("stg_orders and fct_payments were fixed.", ["inc-001", "inc-002"]))
    assert rules(both) == []


def test_the_prompt_carries_ids_and_sections():
    prompt = build_prompt("Has stg_orders broken?", INCIDENTS)
    assert '<incident id="inc-001" date="">' in prompt
    assert "Fix: Re-ran the load for stg_orders." in prompt


def test_no_provider_or_no_incidents_means_no_summary():
    assert answer_history("q", INCIDENTS, None, KNOWN).summary is None
    assert answer_history("q", (), ScriptedProvider([]), KNOWN).summary is None


def test_a_failing_reply_is_retried_once_then_dropped():
    bad = {"answered": True, "sentences": [{"text": "It broke.", "cites": []}]}
    provider = ScriptedProvider([bad, bad])
    result = answer_history("q", INCIDENTS, provider, KNOWN)
    assert result.summary is None
    assert len(result.attempts) == 2


BODY = """## Summary

stg_orders lost rows because raw.orders was late.

## Root cause

The orders export stopped.

## Fix

Restarted the orders export and rebuilt stg_orders.
"""


def history_reply():
    return {**NOT_LINEAGE, "intent": "history"}


@pytest.fixture
def indexed(conn, load_graph):
    load_graph(NODES, EDGES)
    doc = Document(
        id="inc-001",
        type="incident",
        title="Orders missing",
        date=date(2025, 3, 1),
        mention="exact",
        node_roles=(),
        body=BODY,
        path=Path("corpus/test/incidents/inc-001.md"),
        meta={"severity": "sev2"},
    )
    index_incidents(conn, Corpus(TEST_PROJECT, 1, (doc,)), None)


def ask(conn, provider, question):
    request = AskRequest(project=TEST_PROJECT, question=question)
    return run_ask(conn, provider, request, max_named=20)


def test_a_history_question_lists_incidents_and_a_cited_summary(conn, indexed):
    reply = {
        "answered": True,
        "sentences": [
            {"text": "The orders export stopped; it was restarted.", "cites": ["inc-001"]}
        ],
    }
    response = ask(conn, ScriptedProvider([history_reply(), reply]), "Has stg_orders broken?")

    assert isinstance(response, HistoryResponse)
    assert response.mode == "keyword"
    assert [(i.doc_id, i.date, i.severity) for i in response.incidents] == [
        ("inc-001", "2025-03-01", "sev2")
    ]
    assert response.incidents[0].url == "corpus/test/incidents/inc-001.md"
    assert response.answered is True
    assert response.summary[0].cites == ["inc-001"]
    assert [a.stage for a in response.attempts] == ["parse", "history"]


def test_the_model_can_say_nothing_answers(conn, indexed):
    reply = {"answered": False, "sentences": []}
    response = ask(conn, ScriptedProvider([history_reply(), reply]), "Did stg_orders ever break?")

    assert response.answered is False
    assert response.summary == []
    assert "only the closest matches" in response.message


def test_no_matching_incident_says_so_without_a_model_call(conn, indexed):
    provider = ScriptedProvider([history_reply()])
    response = ask(conn, provider, "Did the billing invoice ever fail?")

    assert response.incidents == []
    assert response.message == "No recorded incident matches the question."
    assert [a.stage for a in response.attempts] == ["parse"]


DATED = (
    Retrieved(
        "inc-008",
        "Stops dropped",
        (("Summary", "On 7 May about 6% of stops went missing."),),
        "2025-05-07",
    ),
)


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        ("On 7 May 2025 about 6% of stops went missing.", True),
        ("On 07 May, stops went missing.", True),
        ("On 7 May 2023 stops went missing.", False),
        ("About 12% of stops went missing.", False),
        ("inc-008 lost stops on 7 May.", True),
    ],
)
def test_numbers_must_come_from_the_cited_incident_or_its_date(text, ok):
    result = check_summary(summary((text, ["inc-008"])), DATED, KNOWN)
    assert [v.rule for v in result] == ([] if ok else ["number_not_in_citation"])


def test_the_prompt_gives_each_incident_its_date():
    assert '<incident id="inc-008" date="2025-05-07">' in build_prompt("q", DATED)
