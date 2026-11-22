"""Seeded document skeletons from a manifest: the facts in front-matter, the prose left as TODO.

The facts are drawn so the corpus looks like a real one (ADR 0014): breakage is noticed in
marts, a few busy marts break often (Zipf-like weights), root causes sit upstream of symptoms
and match the cause, some incidents repeat an earlier symptom with a different cause, and
some documents must avoid exact node names. The same manifest, seed and counts always give
the same skeletons.
"""

import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from fnmatch import fnmatch
from pathlib import Path

from asklineage.corpus.docs import (
    CORPUS_VERSION,
    FOLDERS,
    SECTIONS,
    SOURCE_CAUSES,
    TODO,
    node_ids,
    render_document,
)
from asklineage.eval.truth import TruthGraph, truth_graph

START = date(2025, 1, 6)
END = date(2026, 10, 30)
HOT_NODES = 60
MAX_ROOT_HOPS = 4
SYMPTOM_PATHS = ("models/mart/*",)
CAUSE_WEIGHTS = {
    "source_outage": 0.3,
    "logic_bug": 0.25,
    "schema_change": 0.2,
    "duplicate_rows": 0.15,
    "incremental_gap": 0.1,
}
TEST_TYPES = {
    "source_outage": ("not_null", "relationships", "not_null_proportion"),
    "schema_change": ("not_null", "accepted_values", "expression_is_true"),
    "duplicate_rows": ("unique", "unique_combination_of_columns", "unique_proportion"),
    "logic_bug": ("accepted_values", "relationships", "expression_is_true", "accepted_range"),
    "incremental_gap": ("not_null", "unique", "unique_combination_of_columns"),
}
SEVERITY_WEIGHTS = {"sev1": 0.1, "sev2": 0.35, "sev3": 0.55}
INCIDENT_INFORMAL = 0.2
SLACK_INFORMAL = 0.35
SECOND_SYMPTOM = 0.3
FAILED_TEST = 0.75
FIRST_DECOY = 5  # a decoy needs earlier incidents to copy
INCIDENT_CHANNELS = ("#data-incidents", "#data-alerts")
QUESTION_CHANNELS = ("#data-help", "#analytics", "#transit-data")


@dataclass(frozen=True)
class Skeleton:
    folder: str
    id: str
    meta: dict
    body: str

    @property
    def relative_path(self) -> Path:
        return Path(self.folder) / f"{self.id}.md"


@dataclass(frozen=True)
class Project:
    graph: TruthGraph
    ids: dict[str, str]
    names: dict[str, str]  # unique_id -> name used in documents
    tests: dict[str, list[tuple[str, str]]]  # node unique_id -> [(test name, test type)]
    incremental: frozenset[str]
    paths: dict[str, str]  # unique_id -> original_file_path


def read_project(manifest: Mapping, project: str) -> Project:
    ids = node_ids(manifest, project)
    tests: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for node in manifest["nodes"].values():
        if node["resource_type"] == "test" and node.get("attached_node"):
            test_type = (node.get("test_metadata") or {}).get("name") or "singular"
            tests[node["attached_node"]].append((node["name"], test_type))
    incremental = frozenset(
        unique_id
        for unique_id, node in manifest["nodes"].items()
        if (node.get("config") or {}).get("materialized") == "incremental"
    )
    return Project(
        graph=truth_graph(manifest),
        ids=ids,
        names={unique_id: name for name, unique_id in ids.items()},
        tests={node: sorted(found) for node, found in tests.items()},
        incremental=incremental,
        paths={
            uid: node.get("original_file_path") or ""
            for uid, node in manifest["nodes"].items()
            if node["resource_type"] == "model"
        },
    )


def depths(graph: TruthGraph, start: str, step: Mapping[str, frozenset[str]], limit: int):
    """Node -> fewest hops from start (start excluded), up to limit hops."""
    found: dict[str, int] = {}
    frontier = [start]
    for hops in range(1, limit + 1):
        frontier = sorted(
            {n for node in frontier for n in step.get(node, ()) if n not in found and n != start}
        )
        for node in frontier:
            found[node] = hops
    return found


def hot_models(project: Project, symptom_paths: Sequence[str]) -> list[str]:
    """Marts ranked by how much they feed and how many tests watch them."""
    graph = project.graph
    models = [
        uid
        for uid in project.names
        if uid.startswith("model.")
        and any(fnmatch(project.paths.get(uid, ""), pattern) for pattern in symptom_paths)
    ]
    score = {
        uid: len(depths(graph, uid, graph.children, 10)) + 3 * len(project.tests.get(uid, ()))
        for uid in models
        if any(p.startswith("source.") for p in depths(graph, uid, graph.parents, 10))
    }
    return sorted(score, key=lambda uid: (-score[uid], uid))[:HOT_NODES]


def zipf_choice(rng: random.Random, ranked: list[str]) -> str:
    return rng.choices(ranked, weights=[1 / (rank + 1) ** 0.8 for rank in range(len(ranked))])[0]


def weighted(rng: random.Random, weights: Mapping[str, float]) -> str:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def root_candidates(project: Project, symptom: str, cause: str) -> dict[str, int]:
    above = depths(project.graph, symptom, project.graph.parents, MAX_ROOT_HOPS)
    if cause not in SOURCE_CAUSES:
        above = {symptom: 0, **above}
    keep = {}
    for uid, hops in above.items():
        if uid not in project.names:
            continue
        is_source = uid.startswith("source.")
        if cause in SOURCE_CAUSES and not is_source:
            continue
        if cause not in SOURCE_CAUSES and not uid.startswith("model."):
            continue
        if cause == "incremental_gap" and uid not in project.incremental:
            continue
        keep[uid] = hops
    return keep


def pick_test(rng: random.Random, project: Project, path: set[str], cause: str) -> list[str]:
    on_path = [test for uid in sorted(path) for test in project.tests.get(uid, ())]
    preferred = [name for name, kind in on_path if kind in TEST_TYPES[cause]]
    pool = preferred or [name for name, _ in on_path]
    if not pool or rng.random() > FAILED_TEST:
        return []
    return [rng.choice(pool)]


def draw_incident(
    rng: random.Random, project: Project, symptom: str, avoid: tuple[str, str] | None = None
) -> dict | None:
    """A cause and root cause for one symptom model, or None if no cause fits it."""
    causes = dict(CAUSE_WEIGHTS)
    while causes:
        cause = weighted(rng, causes)
        roots = root_candidates(project, symptom, cause)
        roots = {uid: h for uid, h in roots.items() if avoid != (project.names[uid], cause)}
        if not roots:
            del causes[cause]
            continue
        ordered = sorted(roots)
        root = rng.choices(ordered, weights=[1 / (1 + roots[uid]) for uid in ordered])[0]
        symptoms = [symptom]
        if rng.random() < SECOND_SYMPTOM:
            graph = project.graph
            below = depths(graph, root, graph.children, 2)
            others = sorted(
                u for u in below if u.startswith("model.") and u in project.names and u != symptom
            )
            if others:
                symptoms.append(rng.choice(others))
        below = set(depths(project.graph, root, project.graph.children, 50)) | {root}
        above: set[str] = set()
        for uid in symptoms:
            above |= set(depths(project.graph, uid, project.graph.parents, 50)) | {uid}
        path = below & above
        return {
            "path": path,
            "cause": cause,
            "symptom_nodes": [project.names[uid] for uid in symptoms],
            "root_cause_node": project.names[root],
            "failed_tests": pick_test(rng, project, path, cause),
        }
    return None


def spread_dates(rng: random.Random, count: int) -> list[date]:
    span = (END - START).days
    return sorted(START + timedelta(days=rng.randrange(span)) for _ in range(count))


def incident_body() -> str:
    return "\n\n".join(f"## {section}\n\n{TODO}" for section in SECTIONS["incident"])


def runbook_body() -> str:
    return "\n\n".join(f"## {section}\n\n{TODO}" for section in SECTIONS["runbook"])


def scaffold(
    manifest: Mapping,
    project_name: str,
    seed: int,
    incidents: int = 40,
    decoys: int = 6,
    runbooks: int = 12,
    incident_threads: int = 15,
    question_threads: int = 10,
    symptom_paths: Sequence[str] = SYMPTOM_PATHS,
) -> list[Skeleton]:
    rng = random.Random(seed)
    project = read_project(manifest, project_name)
    hot = hot_models(project, symptom_paths)
    if not hot:
        raise ValueError(f"no models under {list(symptom_paths)} with a source upstream")

    drawn: list[dict] = []
    decoy_at = set(rng.sample(range(FIRST_DECOY, incidents), decoys))
    for position, when in enumerate(spread_dates(rng, incidents)):
        facts = None
        if position in decoy_at:
            original = rng.choice(drawn)
            symptom = project.ids[original["symptom_nodes"][0]]
            avoid = (original["root_cause_node"], original["cause"])
            facts = draw_incident(rng, project, symptom, avoid)
            if facts:
                facts["decoy_of"] = original["id"]
                on_path = {n for uid in facts["path"] for n, _ in project.tests.get(uid, ())}
                if set(original["failed_tests"]) <= on_path:
                    facts["failed_tests"] = list(original["failed_tests"])
        while facts is None:
            facts = draw_incident(rng, project, zipf_choice(rng, hot))
        facts["id"] = f"inc-{len(drawn) + 1:03d}"
        facts["date"] = when
        facts["severity"] = weighted(rng, SEVERITY_WEIGHTS)
        facts["mention"] = "informal" if rng.random() < INCIDENT_INFORMAL else "exact"
        drawn.append(facts)

    families = Counter((f["cause"], f["root_cause_node"]) for f in drawn)
    chosen = [key for key, _ in sorted(families.items(), key=lambda kv: (-kv[1], kv[0]))]
    chosen = chosen[:runbooks]
    skeletons: list[Skeleton] = []
    runbook_of: dict[tuple[str, str], str] = {}
    for number, (cause, root) in enumerate(chosen, start=1):
        runbook_id = f"rb-{number:02d}"
        runbook_of[(cause, root)] = runbook_id
        related = [f for f in drawn if (f["cause"], f["root_cause_node"]) == (cause, root)]
        nodes = list(dict.fromkeys([root, *(n for f in related for n in f["symptom_nodes"])]))[:4]
        meta = {
            "id": runbook_id,
            "type": "runbook",
            "title": TODO,
            "date": related[0]["date"] + timedelta(days=7),
            "mention": "exact",
            "cause": cause,
            "nodes": nodes,
        }
        skeletons.append(Skeleton(FOLDERS["runbook"], runbook_id, meta, runbook_body()))

    for facts in drawn:
        meta = {
            "id": facts["id"],
            "type": "incident",
            "title": TODO,
            "date": facts["date"],
            "severity": facts["severity"],
            "mention": facts["mention"],
            "cause": facts["cause"],
            "symptom_nodes": facts["symptom_nodes"],
            "root_cause_node": facts["root_cause_node"],
            "failed_tests": facts["failed_tests"],
        }
        runbook = runbook_of.get((facts["cause"], facts["root_cause_node"]))
        if runbook:
            meta["runbook"] = runbook
        if facts.get("decoy_of"):
            meta["decoy_of"] = facts["decoy_of"]
        skeletons.append(Skeleton(FOLDERS["incident"], facts["id"], meta, incident_body()))

    by_severity = sorted(drawn, key=lambda f: (f["severity"], f["id"]))
    about = sorted(by_severity[:incident_threads], key=lambda f: f["id"])
    threads = [
        {
            "date": f["date"] + timedelta(days=rng.randrange(2)),
            "channel": rng.choice(INCIDENT_CHANNELS),
            "nodes": list(f["symptom_nodes"]),
            "incident": f["id"],
        }
        for f in about
    ]
    for when in spread_dates(rng, question_threads):
        threads.append(
            {
                "date": when,
                "channel": rng.choice(QUESTION_CHANNELS),
                "nodes": [project.names[zipf_choice(rng, hot)]],
            }
        )
    threads.sort(key=lambda t: (t["date"], t.get("incident", "")))
    for number, thread in enumerate(threads, start=1):
        thread_id = f"sl-{number:03d}"
        meta = {
            "id": thread_id,
            "type": "slack",
            "title": TODO,
            "date": thread["date"],
            "mention": "informal" if rng.random() < SLACK_INFORMAL else "exact",
            "channel": thread["channel"],
            "nodes": thread["nodes"],
        }
        if thread.get("incident"):
            meta["incident"] = thread["incident"]
        skeletons.append(Skeleton(FOLDERS["slack"], thread_id, meta, f"- 00:00 **todo**: {TODO}"))
    return skeletons


def write_scaffold(directory: Path, project: str, seed: int, skeletons: list[Skeleton]) -> int:
    """Writes every skeleton, or nothing if any file (or corpus.yml) already exists."""
    settings = directory / "corpus.yml"
    targets = [directory / s.relative_path for s in skeletons]
    existing = [p for p in [settings, *targets] if p.exists()]
    if existing:
        raise FileExistsError(f"{len(existing)} file(s) already exist, e.g. {existing[0]}")
    for skeleton, target in zip(skeletons, targets, strict=True):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_document(skeleton.meta, skeleton.body))
    settings.write_text(
        f"# Synthetic corpus for {project} (ADR 0014). Fixed once the prose is written.\n"
        f"version: {CORPUS_VERSION}\nproject: {project}\nseed: {seed}\n"
    )
    return len(skeletons)
