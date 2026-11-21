# 8. LangGraph orchestrates /api/ask; plain functions would do for now

**Status:** accepted, 2026-11-21

## Context
Step 6 turns a question into an answer: read the question (or take structured fields), find the node, walk the lineage, write the answer. Three paths end early: the question isn't about lineage (abstain), the name matches several nodes or none (clarification), or no LLM is available for free text (abstain).

As built in Step 6a, the flow is linear with two branch points and no loop of its own. The only loop is the check-and-retry inside `generate_checked`, bounded at two attempts. A 30-line function of `if`s would express it completely, and was the first recommendation.

LangGraph was adopted anyway, for two reasons:
1. **Learning the tool and its trade-offs** is a goal of this repo, and a small graph is the cheapest place to learn it.
2. **Two scenarios on the roadmap need what it adds** (Step 6b): resuming a clarification without re-reading the question, and follow-up questions that reuse the previous node. Both need per-conversation state that survives between requests, which LangGraph provides as checkpoints. Phase 3's multi-part questions (plan, run, re-plan) add a real loop.

## Decision
- `/api/ask` runs a compiled `StateGraph` (`api/flow.py`): `parse` or `read_fields` → `resolve` → `lineage` → `answer`, with conditional edges to `abstain` and `clarify`. Drawn by `ASK_GRAPH.get_graph().draw_mermaid()`; dotted edges are conditional:

  ```mermaid
  graph TD;
      __start__([start]) -.-> parse;
      __start__ -.-> read_fields;
      parse -.-> abstain;
      parse -.-> resolve;
      read_fields --> resolve;
      resolve -.-> clarify;
      resolve -.-> lineage;
      lineage --> answer;
      abstain --> __end__([end]);
      clarify --> __end__;
      answer --> __end__;
  ```
- **Nodes are thin.** Each wraps a pure function that existed before LangGraph or is tested without it (`parse_question`, `resolve_node`, `lineage`, `answer_lineage`). The graph decides the order; the functions do the work.
- **Only the graph runtime is used, not LangChain's model layer.** LLM calls still go through our `LLMProvider` and `generate_checked` (ADR 0006, 0007).
- **Dependencies go through the runtime context, not the state.** The connection, provider and settings are passed as `AskContext`; the state holds only data (request, query, resolution, result, response), so it can be checkpointed in 6b.
- **Attempts collect through a reducer** (`Annotated[list, operator.add]`): each node returns only its own attempts.
- **Every loop is bounded in code**: two attempts per LLM call, 25 hops per walk. `recursion_limit=10` is a backstop that turns a wiring bug into an error; the longest run is 4 steps.
- **The LLM reads the question but never picks a node.** It copies a name from the question, and code checks that the name appears in the question as a whole name (`dim_feeds` does not match inside `dim_feeds_v2`); `resolve_node` decides (ADR 0004).
- **Every outcome is HTTP 200 with a `kind`**: `lineage`, `clarification` (`ambiguous` / `not_found`, with candidates), or `abstain` (`not_lineage`, `not_understood`, `llm_off`, `llm_unavailable`). Bad requests are 422. With `LLM_PROVIDER=none`, structured requests (`node`, `direction`, `depth`) work; free text abstains with `llm_off`.

## What plain functions would have looked like
```python
def ask(conn, provider, request):
    query, abstained = read_question(provider, request)
    if abstained:
        return abstained
    resolution = resolve_node(conn, query.entity, request.project)
    if not isinstance(resolution, Resolved):
        return clarification(resolution)
    result = lineage(conn, resolution.node.unique_id, query.direction, query.depth)
    return lineage_response(answer_lineage(result, resolution.node, provider, ...))
```
Same functions, same tests, no dependency. Moving between the two is a change to the orchestrator only.

## Trade-offs
| | Plain functions | **LangGraph** |
|---|---|---|
| Dependencies | none | `langgraph` 1.2.24 adds 20 packages to the lock: 6 LangChain/LangGraph packages (`langgraph`, `-checkpoint`, `-prebuilt`, `-sdk`, `langchain-core`, `langchain-protocol`), `langsmith` (pulled in by `langchain-core`, with `requests`, `zstandard` and others), and serialisers (`orjson`, `ormsgpack`) |
| Reading the flow | top to bottom in one function | node functions plus an edge list; `ASK_GRAPH.get_graph().draw_mermaid()` draws it |
| State | local variables, typed by the checker | a `TypedDict` the nodes partially update; a key a node didn't set is a runtime `KeyError`, not a type error |
| Testing | call the function | the same pure functions are tested directly; the graph is tested end to end |
| Pause and resume (6b) | build it: a sessions table, serialised state, resume logic | `interrupt()` plus a Postgres checkpointer |
| Loops (Phase 3) | hand-written, with its own bound | graph cycles, bounded by our counters and `recursion_limit` |

## Update after 6b (ADR 0009)
6b added a `begin` node (it clears the turn's keys, because a checkpoint carries them into the next question) and edges `parse → lineage` (follow-ups) and `clarify → lineage` (a resumed choice). What the checkpointer cost in our code:
- `langgraph-checkpoint-postgres` and its tables, a second pool with different connection settings, and `saver.setup()` in `init-db`.
- An explicit serializer allowlist of 20 state types, because LangGraph's default lets unregistered types through with a warning and says it will block them.
- Reasoning about `interrupt()`: the node runs again from its first line on resume, so nothing before the call may have side effects.
- A rule for every state key: turn-scoped (cleared by `begin`) or thread-scoped (`previous`).

## Consequences
- The state must stay serializable: no connections, clients or providers in it.
- `langsmith` (LangChain's tracing client) is installed but sends nothing unless `LANGSMITH_TRACING` and an API key are set. Deployments should leave them unset, or set them deliberately; tracing would send questions and answers to a third party. Tracing in this repo is OpenTelemetry (Phase 6). It also registers a pytest plugin (`plugins: langsmith-0.16.0` in every test run), inactive without those settings.
- `langgraph-prebuilt` and `langgraph-sdk` (agent helpers and the hosted-platform client) come with `langgraph` and are unused.
- Server-side state arrives with 6b, and with it checkpoint storage, retention and per-thread concurrency. The stateless path (pass `node` as a `unique_id`) stays, so no client has to depend on it.
- If 6b and Phase 3 don't end up using checkpoints or cycles, this ADR should be revisited and the graph replaced by the function above.

## First runs: qwen2.5:7b on Cal-ITP (6a)
Every LLM call passed on the first attempt. Reading the question costs about 0.6 s and 600 tokens in / 20–24 out, mostly the rules, examples and schema in the system prompt.

| Question | Outcome | Parse | Highlights |
|---|---|---|---|
| "How do I write a window function?" | `abstain` / `not_lineage` | 539 ms | — |
| "What does dim_schedule_feedz depend on?" | `clarification` / `not_found`; `query` kept the typo; `dim_schedule_feeds` top suggestion (0.89) | 616 ms | — |
| "What breaks downstream if I change dim_schedule_feeds?" | `lineage`, downstream, default depth 5: 192 nodes, 4 teams, truncated | 659 ms | 3.5 s, 1.6k in / 166 out, `passed` |

- The model copied the typo instead of correcting it, which is the behaviour ADR 0004 needs; a corrected name would have failed the entity check.
- The highlights were all true but repeated ADR 0007's run 5 weakness: the three alphabetically first gtfs-team direct children (`dim_agency`, `dim_agency_latest`, `dim_areas`), although payments-team, tides-team and transit-database-team nodes were named. The rule to prefer other-team nodes is not followed. Usefulness is measured in Step 7 before changing the prompt or moving the choice into code.
- With no depth in the question, the default of 5 returned 192 rows. The full table stays in the response because it is the complete answer; clients collapse it.

## Known gaps
- `depth` from the LLM is range-checked (1..25) but not checked against the question, so a model could pick a depth the user didn't ask for. The response echoes `direction` and `depth`, so the reading is visible.
- A name written with spaces where dbt uses underscores ("dim schedule feeds") passes the check but resolves to `not_found`, with suggestions.
- Ownership-only questions ("who owns X?") abstain for now; the lineage answer covers owners of the nodes it lists.
