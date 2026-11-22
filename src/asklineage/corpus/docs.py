"""Incident, runbook and Slack documents: Markdown with YAML front-matter (ADR 0014).

Front-matter is the ground truth: which nodes a document is about, and for incidents the root
cause and symptoms. `check_corpus` holds every document to the manifest and its lineage, so a
fact in the corpus can't contradict the project. Pure: reads files and a manifest dict.
"""

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Literal, get_args

import yaml

from asklineage.eval.truth import TruthGraph, truth_graph

DocType = Literal["incident", "runbook", "slack"]
Cause = Literal["source_outage", "schema_change", "duplicate_rows", "logic_bug", "incremental_gap"]
Severity = Literal["sev1", "sev2", "sev3"]
Mention = Literal["exact", "informal"]
Role = Literal["symptom", "root_cause", "about"]

DOC_TYPES: tuple[str, ...] = get_args(DocType)
CAUSES: tuple[str, ...] = get_args(Cause)
SEVERITIES: tuple[str, ...] = get_args(Severity)
MENTIONS: tuple[str, ...] = get_args(Mention)
SOURCE_CAUSES = frozenset({"source_outage", "schema_change"})
CORPUS_VERSION = 1
FOLDERS = {"incident": "incidents", "runbook": "runbooks", "slack": "slack"}
SECTIONS = {
    "incident": ("Summary", "Impact", "Root cause", "Fix", "Follow-ups"),
    "runbook": ("When to use", "Steps", "Check"),
    "slack": (),
}
TODO = "<!-- TODO -->"
SLACK_MESSAGE = re.compile(r"^- \d{2}:\d{2} \*\*[a-z][a-z0-9._-]*\*\*: \S", re.MULTILINE)
SLACK_THREAD_DAYS = 2


class CorpusError(ValueError):
    pass


@dataclass(frozen=True)
class Document:
    id: str
    type: DocType
    title: str
    date: date
    mention: Mention
    node_roles: tuple[tuple[str, Role], ...]  # (node name, role), in front-matter order
    body: str
    path: Path
    meta: Mapping = field(repr=False)

    @property
    def nodes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(name for name, _ in self.node_roles))

    @property
    def is_draft(self) -> bool:
        return TODO in self.body or TODO in self.title


@dataclass(frozen=True)
class Corpus:
    project: str
    seed: int
    documents: tuple[Document, ...]


@dataclass(frozen=True)
class CorpusReport:
    corpus: Corpus
    problems: tuple[str, ...]
    drafts: tuple[str, ...]


def split_front_matter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        raise CorpusError("no front-matter (the file must start with ---)")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise CorpusError("front-matter is not closed with ---")
    meta = yaml.safe_load(text[4:end]) or {}
    if not isinstance(meta, dict):
        raise CorpusError("front-matter is not a mapping")
    return meta, text[end + 5 :].strip() + "\n"


def render_document(meta: Mapping, body: str) -> str:
    front = yaml.safe_dump(dict(meta), sort_keys=False, allow_unicode=True, width=100)
    return f"---\n{front}---\n\n{body.strip()}\n"


def node_roles(meta: Mapping) -> tuple[tuple[str, Role], ...]:
    if meta.get("type") == "incident":
        symptoms = meta.get("symptom_nodes") or ()
        roles: list[tuple[str, Role]] = [(str(n), "symptom") for n in symptoms]
        if meta.get("root_cause_node"):
            roles.append((str(meta["root_cause_node"]), "root_cause"))
        return tuple(roles)
    return tuple((str(n), "about") for n in meta.get("nodes") or ())


def load_corpus(directory: Path) -> tuple[Corpus, list[str]]:
    """Every document under directory, and the problems that stop a file being read at all."""
    settings_path = directory / "corpus.yml"
    if not settings_path.exists():
        raise CorpusError(f"{settings_path}: not found")
    settings = yaml.safe_load(settings_path.read_text()) or {}
    if settings.get("version") != CORPUS_VERSION or not settings.get("project"):
        raise CorpusError(f"{settings_path}: expected version {CORPUS_VERSION} and a project")

    problems: list[str] = []
    documents: list[Document] = []
    for path in sorted(directory.glob("*/*.md")):
        label = path.relative_to(directory).as_posix()
        try:
            meta, body = split_front_matter(path.read_text())
        except (CorpusError, yaml.YAMLError) as exc:
            problems.append(f"{label}: {exc}")
            continue
        doc_type = meta.get("type")
        when = meta.get("date")
        if isinstance(when, str):
            try:
                when = date.fromisoformat(when)
            except ValueError:
                when = None
        if doc_type not in DOC_TYPES or not isinstance(when, date):
            problems.append(f"{label}: needs a type ({', '.join(DOC_TYPES)}) and a YYYY-MM-DD date")
            continue
        documents.append(
            Document(
                id=str(meta.get("id", "")),
                type=doc_type,
                title=str(meta.get("title", "")).strip(),
                date=when,
                mention=meta.get("mention"),
                node_roles=node_roles(meta),
                body=body,
                path=path,
                meta=meta,
            )
        )
    corpus = Corpus(str(settings["project"]), int(settings.get("seed", 0)), tuple(documents))
    return corpus, problems


def node_ids(manifest: Mapping, project: str) -> dict[str, str]:
    """Name -> unique_id: models, seeds and snapshots by name, sources as source_name.name."""
    ids = {
        node["name"]: unique_id
        for unique_id, node in manifest["nodes"].items()
        if node["resource_type"] in ("model", "seed", "snapshot")
        and node["package_name"] == project
    }
    for unique_id, source in (manifest.get("sources") or {}).items():
        if source["package_name"] == project:
            ids[f"{source['source_name']}.{source['name']}"] = unique_id
    return ids


def tests_by_name(manifest: Mapping) -> dict[str, str | None]:
    """Test name -> the node it is attached to."""
    return {
        node["name"]: node.get("attached_node")
        for node in manifest["nodes"].values()
        if node["resource_type"] == "test"
    }


def reach(graph: TruthGraph, start: str, step: Mapping[str, frozenset[str]]) -> set[str]:
    """start and every node reachable from it, with no depth limit."""
    seen = {start}
    frontier = [start]
    while frontier:
        frontier = [n for node in frontier for n in step.get(node, ()) if n not in seen]
        seen.update(frontier)
    return seen


def exact_names(name: str) -> tuple[str, ...]:
    """Spellings that count as naming a node exactly. A source's bare table name counts only
    when it has an underscore; "feeds" or "stops" alone is ordinary English."""
    if "." in name:
        table = name.split(".", 1)[1]
        return (name, table) if "_" in table else (name,)
    return (name,)


def names_in(text: str, name: str) -> bool:
    return any(
        re.search(rf"(?<![\w.]){re.escape(spelling)}(?![\w])", text, re.IGNORECASE)
        for spelling in exact_names(name)
    )


def check_corpus(directory: Path, manifest: Mapping) -> CorpusReport:
    corpus, problems = load_corpus(directory)
    ids = node_ids(manifest, corpus.project)
    if not ids:
        raise CorpusError(f"the manifest has nothing in project {corpus.project!r}")
    graph = truth_graph(manifest)
    tests = tests_by_name(manifest)
    nodes = manifest["nodes"]
    by_id: dict[str, Document] = {}
    for doc in corpus.documents:
        if doc.id in by_id:
            problems.append(f"{doc.id}: duplicate id")
        by_id[doc.id] = doc

    for doc in corpus.documents:
        say = problems.append
        label = doc.id or doc.path.name
        if doc.path.stem != doc.id:
            say(f"{label}: id must match the file name {doc.path.stem!r}")
        if doc.path.parent.name != FOLDERS[doc.type]:
            say(f"{label}: a {doc.type} belongs in {FOLDERS[doc.type]}/")
        if not doc.title:
            say(f"{label}: no title")
        if doc.mention not in MENTIONS:
            say(f"{label}: mention must be one of {', '.join(MENTIONS)}")
        if not doc.node_roles:
            say(f"{label}: names no nodes")
        # An incident's root cause may also be a symptom; a repeat within one role is a slip.
        if len(set(doc.node_roles)) != len(doc.node_roles):
            say(f"{label}: a node is listed twice")
        unknown = [name for name in doc.nodes if name not in ids]
        say_each(say, label, [f"no node named {name!r}" for name in unknown])

        if doc.type == "incident":
            check_incident(doc, by_id, ids, graph, tests, nodes, say)
        elif doc.type == "slack":
            check_slack(doc, by_id, say)
        elif doc.meta.get("cause") not in CAUSES:
            say(f"{label}: cause must be one of {', '.join(CAUSES)}")

        if not doc.is_draft and not unknown:
            check_body(doc, say)

    drafts = tuple(doc.id for doc in corpus.documents if doc.is_draft)
    return CorpusReport(corpus, tuple(problems), drafts)


def say_each(say, label: str, messages: list[str]) -> None:
    for message in messages:
        say(f"{label}: {message}")


def check_incident(doc, by_id, ids, graph, tests, nodes, say) -> None:
    meta = doc.meta
    label = doc.id
    cause = meta.get("cause")
    if cause not in CAUSES:
        say(f"{label}: cause must be one of {', '.join(CAUSES)}")
    if meta.get("severity") not in SEVERITIES:
        say(f"{label}: severity must be one of {', '.join(SEVERITIES)}")
    symptoms = [str(n) for n in meta.get("symptom_nodes") or ()]
    root = meta.get("root_cause_node")
    if not symptoms or not root:
        say(f"{label}: needs symptom_nodes and a root_cause_node")
        return
    if any(name not in ids for name in [*symptoms, root]):
        return

    root_id = ids[root]
    is_source = root_id.startswith("source.")
    if cause in SOURCE_CAUSES and not is_source:
        say(f"{label}: a {cause} starts at a source, not {root!r}")
    if cause not in SOURCE_CAUSES and is_source:
        say(f"{label}: a {cause} starts at a model, not the source {root!r}")
    if cause == "incremental_gap" and not is_source:
        if (nodes[root_id].get("config") or {}).get("materialized") != "incremental":
            say(f"{label}: an incremental_gap needs an incremental model, {root!r} is not")

    below_root = reach(graph, root_id, graph.children)
    above_symptoms: set[str] = set()
    for name in symptoms:
        if ids[name] not in below_root:
            say(f"{label}: root cause {root!r} is not upstream of symptom {name!r}")
        above_symptoms |= reach(graph, ids[name], graph.parents)
    on_path = below_root & above_symptoms
    for test in meta.get("failed_tests") or ():
        if test not in tests:
            say(f"{label}: no test named {test!r}")
        elif tests[test] not in on_path:
            say(f"{label}: test {test!r} is not on a node between the root cause and a symptom")

    runbook = meta.get("runbook")
    if runbook and getattr(by_id.get(runbook), "type", None) != "runbook":
        say(f"{label}: runbook {runbook!r} is not a runbook in the corpus")
    decoy_of = meta.get("decoy_of")
    if decoy_of:
        original = by_id.get(decoy_of)
        if original is None or original.type != "incident":
            say(f"{label}: decoy_of {decoy_of!r} is not an incident in the corpus")
        else:
            shared = set(symptoms) & set(original.meta.get("symptom_nodes") or ())
            same = (original.meta.get("root_cause_node"), original.meta.get("cause")) == (
                root,
                cause,
            )
            if not shared or same:
                say(f"{label}: a decoy shares a symptom with {decoy_of} but not its root cause")
            if original.date > doc.date:
                say(f"{label}: a decoy can't be dated before its original {decoy_of}")


def check_slack(doc, by_id, say) -> None:
    label = doc.id
    if not str(doc.meta.get("channel", "")).startswith("#"):
        say(f"{label}: channel must start with #")
    incident = doc.meta.get("incident")
    if not incident:
        return
    original = by_id.get(incident)
    if original is None or original.type != "incident":
        say(f"{label}: incident {incident!r} is not an incident in the corpus")
        return
    if not original.date <= doc.date <= original.date + timedelta(days=SLACK_THREAD_DAYS):
        say(f"{label}: a thread about {incident} must start within {SLACK_THREAD_DAYS} days of it")


def check_body(doc: Document, say) -> None:
    label = doc.id
    headings = re.findall(r"^## (.+?)\s*$", doc.body, re.MULTILINE)
    if doc.type == "slack":
        if len(SLACK_MESSAGE.findall(doc.body)) < 2:
            say(f"{label}: a thread needs at least two '- HH:MM **handle**: text' messages")
    elif headings != list(SECTIONS[doc.type]):
        say(f"{label}: sections must be {', '.join(SECTIONS[doc.type])} in that order")
    text = f"{doc.title}\n{doc.body}"
    if doc.mention == "exact":
        missing = [name for name in doc.nodes if not names_in(text, name)]
        say_each(say, label, [f"mention is exact but {name!r} is not named" for name in missing])
    elif doc.mention == "informal":
        named = [name for name in doc.nodes if names_in(text, name)]
        say_each(say, label, [f"mention is informal but {name!r} is named" for name in named])


def describe(report: CorpusReport) -> dict[str, object]:
    docs = report.corpus.documents
    incidents = [d for d in docs if d.type == "incident"]
    per_node = Counter(name for d in incidents for name in d.nodes)
    return {
        "types": dict(Counter(d.type for d in docs)),
        "informal": sum(1 for d in docs if d.mention == "informal"),
        "decoys": sum(1 for d in incidents if d.meta.get("decoy_of")),
        "causes": dict(Counter(d.meta.get("cause") for d in incidents)),
        "busiest": per_node.most_common(5),
    }
