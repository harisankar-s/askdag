# 19. Highlights are chosen and written in code; lineage answers need no LLM

**Status:** accepted, 2026-11-22. Supersedes the highlight part of ADR 0007.

## Context
A lineage answer has a table and a summary paragraph, both from code (ADR 0007), plus up to three highlights: the relationships worth noticing before changing the root. Until now the LLM chose and wrote the highlights. Its prompt gave the selection rule in words: prefer nodes owned by a team other than most of the others, or by no one; otherwise prefer the furthest. Code then checked every sentence (real names, real edge or path in the stated direction, no numbers, no owner wording) and retried once.

The last golden run (qwen2.5:7b, 2026-11-21, 35 answers with highlights) showed:
- **Truth was not the problem:** the manifest contradicted 0 of 101 kept highlights. 1 of 35 first attempts stated a false relation, and the check caught it.
- **Choice was:** where another team was involved, the LLM picked one of its nodes in 3 of 17 answers. Taking the first named nodes did so in 17 of 17, and a seeded random pick in 16 of 17.
- **Cost:** p50 4.5 s and p95 6.1 s per answer, the largest part of a lineage turn's wall time (p50 3.9 s per turn over all turns).

## Decision
- **Code applies the rule the LLM was given** (`answer/highlights.py`):
  1. one node for each team other than the majority, and for unowned nodes: its nearest node, smallest teams first, unowned before a team of the same size;
  2. with slots left, the furthest node, if it is more than one hop away;
  3. then a direct node.
  Ties break by name. Candidates are every node in the walk, not a capped list.
- **The first version picked only one other team**, as the prompt's wording could be read. On `dim_schedule_feeds` (107 nodes downstream, 2 hops) that named transit-database-team and then two gtfs-team nodes, the team the summary already names with 101 nodes; payments-team and tides-team went unnamed. Now the three highlights are those three teams.
- **A template writes each sentence** from what the walk proved: "`x` reads directly from `root`" or "`x` depends on `root` through other nodes", mirrored for upstream, with "the source" before a source. Owner tags are added after each name, as before.
- **Lineage answers make no LLM call.** The LLM stays where language is needed: reading a free-text question (parse) and summarising incidents with citations (ADR 0016).
- **Removed:** the highlight prompt and its worked examples, the sentence checker (`grounding.py`), the highlight eval, `highlight_status`, the `highlights` attempt stage, `LLM_MAX_NAMED_NODES` and the capped list of named nodes, and `lineage --summarize -v`.

## Why no new measurement
The decision rests on two facts, neither of which needs a run:
- **The rule is fully specified,** so following it is computation, not judgment. The 3 of 17 above is the LLM failing its own written rule, not a disagreement about taste. Comparing a code picker with the usefulness proxy would be circular, because the proxy is the same rule.
- **The sentence carried no information beyond (upstream, downstream, hops).** The check forbade numbers, owners and other names, so a template writes the same content with no way to be wrong.

What the change gives up is anything the LLM might have added outside the rule, such as noticing that a node's name suggests a dashboard. No measurement in this build could show that, and the check would have blocked most of it anyway.

## Options compared
| | **Code picks, template writes (chosen)** | Code picks, LLM writes | LLM picks and writes (before) |
|---|---|---|---|
| Follows the stated rule | always | always | 3 of 17 where it applied |
| False relations | none possible | caught by the check | 1 in 35 first attempts, caught |
| Added latency per answer | none | about the same as before | p50 4.5 s, p95 6.1 s |
| Needs a model | no | yes | yes |
| Code to keep | ~70 lines | prompt, checker, retry | prompt, checker, retry, eval |

## Consequences
- **A lineage answer is the same with or without an LLM.** A request that names the node (`{"node": ...}`) never touches a model, and `LLM_PROVIDER=none` loses only free-text questions and history summaries.
- **Highlights are deterministic,** so tests assert exact sentences, and a direct highlight is checked to be an edge of the result.
- **The rule is crude.** It does not know which nodes are dashboards, exports or heavily used. A better rule would need data this build doesn't load (exposures, query logs), and that data would go into the rule, not into a prompt.
- **The API changed:** `highlight_status` is gone from `LineageResponse`, and attempt stages are `parse` and `history`. Threads checkpointed before this change may not load; clear them with `asklineage prune-threads --older-than-hours 0`.
- The golden-run cost table now reports `parse` and `history`, and the prompt hashes cover the parse and history prompts.
- The README's "highlights deferred to Phase 6 with hand labels" line is stale; the Phase 6 README rewrite replaces it.
