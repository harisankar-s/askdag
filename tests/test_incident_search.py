from datetime import date
from pathlib import Path

import pytest
from helpers import TEST_PROJECT, make_node

from asklineage.corpus.docs import Corpus, Document, load_corpus
from asklineage.embed.hashing import HashingEmbedder
from asklineage.eval.incident_eval import (
    IncidentSetError,
    check_against_corpus,
    load_incident_set,
    run_incident_eval,
)
from asklineage.incidents.search import DocHit, NoVectorsError, find_incidents, rrf
from asklineage.incidents.store import index_incidents

ROOT = Path(__file__).parent.parent
COMMITTED = ROOT / "config" / "incident_questions.calitp.yml"


def hit(doc_id: str, section: str = "Fix") -> DocHit:
    return DocHit("p", doc_id, doc_id.upper(), 0.0, section, f"text of {doc_id}")


def test_rrf_rewards_agreement_and_keeps_the_best_ranked_section():
    fused = rrf([[hit("a", "Summary"), hit("b")], [hit("b", "Root cause"), hit("c")]])
    assert [h.doc_id for h in fused] == ["b", "a", "c"]
    assert fused[0].section == "Root cause"
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_committed_questions_point_at_incidents_in_the_committed_corpus():
    incident_set = load_incident_set(COMMITTED)
    corpus, problems = load_corpus(ROOT / "corpus" / "calitp")
    assert problems == []
    check_against_corpus(incident_set, corpus)
    assert {q.category for q in incident_set.questions} == {
        "named",
        "symptom",
        "loose",
        "decoy",
        "none",
    }


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "questions.yml"
    path.write_text(f"version: 1\nproject: {TEST_PROJECT}\nquestions:\n{body}")
    return path


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("  - {id: a, category: named, question: q, expected: []}\n", "only category none"),
        ("  - {id: a, category: none, question: q, expected: [inc-001]}\n", "only category none"),
        ("  - {id: a, category: vague, question: q, expected: [inc-001]}\n", "category 'vague'"),
        (
            "  - {id: a, category: none, question: q}\n  - {id: a, category: none, question: r}\n",
            "duplicate id",
        ),
    ],
)
def test_bad_question_sets_are_rejected(tmp_path, body, expected):
    with pytest.raises(IncidentSetError, match=expected):
        load_incident_set(write(tmp_path, body))


def incident(doc_id: str, body: str) -> Document:
    return Document(
        id=doc_id,
        type="incident",
        title=f"Incident {doc_id}",
        date=date(2025, 1, int(doc_id[-1])),
        mention="exact",
        node_roles=(),
        body=body,
        path=Path(f"{doc_id}.md"),
        meta={"severity": "sev3"},
    )


DOCS = (
    incident("inc-001", "## Summary\n\ndim_x lost rows.\n\n## Fix\n\nFixed the timezone cast."),
    incident("inc-002", "## Summary\n\nstg_x had duplicate rows.\n\n## Fix\n\nAdded a dedup."),
    incident("inc-003", "## Summary\n\nThe feeds table had a timezone problem.\n\n## Fix\n\nNone."),
)


def test_stale_questions_are_rejected(tmp_path):
    incident_set = load_incident_set(
        write(tmp_path, "  - {id: a, category: named, question: q, expected: [inc-404]}\n")
    )
    with pytest.raises(IncidentSetError, match="no incident 'inc-404'"):
        check_against_corpus(incident_set, Corpus(TEST_PROJECT, 1, DOCS))


@pytest.fixture
def indexed(conn, load_graph):
    raw = make_node("x", "source", "raw")
    stg = make_node("stg_x")
    dim = make_node("dim_x")
    load_graph([raw, stg, dim], [(raw.unique_id, stg.unique_id), (stg.unique_id, dim.unique_id)])
    embedder = HashingEmbedder(768)
    index_incidents(conn, Corpus(TEST_PROJECT, 1, DOCS), embedder)
    return embedder


def ids(conn, question, method, embedder):
    return [
        h.doc_id for h in find_incidents(conn, question, method, embedder, project=TEST_PROJECT)
    ]


def test_names_follow_lineage_upstream_and_rank_by_hops(conn, indexed):
    hits = find_incidents(conn, "Has dim_x broken?", "names", indexed, project=TEST_PROJECT)
    assert [(h.doc_id, h.score) for h in hits] == [("inc-001", 1.0), ("inc-002", 0.5)]


def test_names_find_nothing_when_no_node_is_named(conn, indexed):
    assert ids(conn, "Has the feeds table broken?", "names", indexed) == []


def test_keyword_matches_any_word_and_ranks_by_best_chunk(conn, indexed):
    hits = find_incidents(conn, "timezone problem", "keyword", indexed, project=TEST_PROJECT)
    assert [h.doc_id for h in hits] == ["inc-003", "inc-001"]
    assert hits[1].section == "Fix"


def test_vector_and_hybrid_return_documents_once(conn, indexed):
    for method in ("vector", "hybrid"):
        found = ids(conn, "timezone problem in the feeds table", method, indexed)
        assert sorted(found) == ["inc-001", "inc-002", "inc-003"]
        assert found[0] == "inc-003"


def test_routed_puts_name_links_first_then_hybrid(conn, indexed):
    found = ids(conn, "Did dim_x have a timezone problem?", "routed", indexed)
    assert found == ["inc-001", "inc-002", "inc-003"]


def test_vector_search_needs_an_embedder_and_vectors(conn, indexed):
    with pytest.raises(NoVectorsError, match="EMBED_PROVIDER"):
        ids(conn, "timezone", "vector", None)
    other = HashingEmbedder(768)
    other.model = "other-model"
    with pytest.raises(NoVectorsError, match="load-docs"):
        ids(conn, "timezone", "hybrid", other)


def test_eval_scores_each_method(conn, indexed, tmp_path):
    incident_set = load_incident_set(
        write(
            tmp_path,
            "  - {id: n1, category: named, question: 'Has dim_x broken?', expected: [inc-001]}\n"
            "  - {id: x1, category: none, question: 'billing export late?'}\n",
        )
    )
    results = run_incident_eval(conn, incident_set, indexed, ("names", "keyword"))
    assert results["names"]["summary"]["overall"]["hit@1"] == 1
    assert results["keyword"]["summary"]["overall"]["n"] == 1
