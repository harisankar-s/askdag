# 13. Intent in the parse call; owner answers from SQL

**Status:** accepted, 2026-11-21

## Context
`/api/ask` answered only lineage questions. Every other question abstained, including "who owns X?", which the database already answers exactly. Phase 2 also adds search, which needs questions to be told apart: does this one name a node, or describe data?

## Decision
- **One call does both.** `ParsedQuestion.is_lineage: bool` becomes `intent: "lineage" | "owner" | "search" | "other"`, in the same parse call. There is no separate classifier and no second LLM call.
- **Owner questions are answered by code.** The flow parses, resolves the node (clarifying if the name is ambiguous or unknown), then runs an `owner` node. That node reads `owner` and `owner_source` from `nodes` and returns an `OwnerResponse` with the meaning of the source, or says that no owner is recorded. No LLM writes any of the text.
- **Owner follow-ups reuse the previous node** ("who owns it?"). An owner turn keeps the previous direction and depth, so a later "and upstream?" still works.
- **The parse check applies the entity rules to owner questions too.** The direction rule stays lineage only.
- **`not_lineage` is renamed `out_of_scope`.** Search questions use it until step 2d.2 adds search to `/api/ask`.

## Consequences
- **The golden set moves to version 2.** "Who owns X?" is now expected to get an owner answer, and three owner items plus an owner follow-up are added (one about a node with no owner). The sha changes, so runs are not comparable with the version 1 baseline (55 correct, 1 acceptable of 56).
- **A misread intent sends a question down the wrong route.** The eval reports owner reading separately. An owner answer about the wrong node counts as `silent_wrong`, the same as a wrong lineage answer.
- **Old checkpointed threads do not load.** They hold a `ParsedQuestion` with `is_lineage`. Run `prune-threads` to clear them.
- **The web page does not show `kind: owner` yet** (step 2d.3).

## Alternatives considered
- **A separate classifier call:** each step is easier to test on its own, but it adds latency and a second failure point for no gain the eval has shown.
- **A rule for owner questions** (a regex for "who owns"): free, but brittle across phrasings ("which team is responsible for"), and the parse call already reads the question.

## First run (Cal-ITP, qwen2.5:7b, golden set v2 8ef4d9130445, 58 turns)
- 56 correct, 1 acceptable, 1 safe_wrong, 0 silent wrong. Owner questions: read as owner 4/4, right node and owner 4/4. Off-topic questions abstained 4/4. Lineage reading: 24/25, with 100% node, direction and depth on the turns that were answered.
- **The one miss:** in `t-new-node-and-direction` turn 2 ("And what does ntd_report_validation.all_ntdreports feed?"), the model set `same_node: true` even though it named a new node. The check caught it (`same_node_entity`). The retry dropped the name but kept `same_node`, and was caught again (`same_node_names_another_node`), so the flow abstained. Nothing false reached the user. This turn passed in the last version 1 run. The new "Who owns it?" example (`same_node: true`) is a likely reason the model now leans toward `same_node`.
- **Repair: a named node wins over `same_node`** (`named_node_wins`). When no attempt passes, and one failed only because it set `same_node` next to a node name copied from the question, the flow uses that name with `same_node` false. The name comes from the question, so nothing is guessed (ADR 0004). It runs after the retry, as `recover_reading` does, so the retry's cost is still paid.
- **Rerun with the repair:** 57 correct, 1 acceptable, 0 wrong of 58. `t-new-node-and-direction` turn 2 is now correct.

## Addendum: search in /api/ask (step 2d.2)
- `intent: search` routes to a `search` node. It runs the semantic search with lineage-aware ranking (ADR 0012), or keyword search when there is no embedder or no vectors. It returns a `SearchResponse`: the hits with owner, the chunk kind that matched and `via`, plus the mode and a message written by code. No LLM writes any text.
- **`low_confidence` is a flag, not an abstain.** It is set when the semantic top score is under `SEARCH_LOW_CONFIDENCE_BELOW` (0.667), and always in keyword mode. Abstaining on a cutoff would have blocked 6 of 30 answerable questions to catch all 5 unanswerable ones (ADR 0012).
- **The golden set moves to version 3** with two project-neutral search questions. They check only that the intent is read as search; `eval search` measures retrieval.
- A search answer does not set the thread's previous node, so "who owns the first one?" is not supported yet.
- **Golden set v3 run (qwen2.5:7b):** 59 correct, 1 acceptable, 0 wrong of 60. Both search questions were read as search. Through the API, the u01 question returns `fct_breakdowns_by_agency` first, via its staging model.
