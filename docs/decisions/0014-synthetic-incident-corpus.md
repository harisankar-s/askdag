# 14. A synthetic incident corpus: seeded scenarios, hand-written prose, ground truth in front-matter

**Status:** accepted, 2026-11-22

## Context
Phase 3 answers "has this broken before, and how was it fixed?". That needs incident write-ups, runbooks and Slack threads. Cal-ITP publishes none, and a real company's would be private. So the corpus is synthetic, and how it is made decides what the retrieval eval can measure:
- **The text must not be the ground truth.** If the eval reads "which incident is about which node" out of the prose, a retriever that matches the prose gets graded by its own method.
- **Fully templated text flatters keyword search.** Every incident would say "`not_null` test on X failed", the questions would repeat that wording, and BM25 would look as good as embeddings.
- **Facts must agree with the project.** An incident whose "root cause" is a model downstream of the one that broke teaches the answer step to say false things, and the eval can't tell.
- **Real corpora are skewed.** A few busy models break again and again; most never do. The same symptom recurs with different causes. Retrieval is only hard when the corpus looks like this.

## Decision
- **Two stages.** `asklineage corpus scaffold` (seeded, from the manifest) writes one Markdown file per document with its facts in YAML front-matter and empty sections marked `TODO`. The prose is then written by hand (by Claude, in this build) from those facts. The scaffold refuses to overwrite a file, so hand-written prose is never lost to a rerun.
- **Front-matter is the ground truth.** Incidents: `symptom_nodes`, `root_cause_node`, `cause`, `failed_tests`, `severity`, `runbook`, `decoy_of`. Runbooks and Slack threads: `nodes`, plus the `incident` a thread is about. The eval in 3c uses only these fields.
- **`asklineage corpus check` enforces what the scaffold promised, on every edit:**
  - every node and test exists in the manifest;
  - the root cause equals or is upstream of every symptom node, using the independent BFS from ADR 0010;
  - a failed test is attached to a node on a path from the root cause to a symptom;
  - the cause matches the node kind (`source_outage` and `schema_change` start at a source, `incremental_gap` at an incremental model, the others at a model);
  - links point at documents of the right type, and a decoy shares a symptom with its original but has a different root cause or cause;
  - `mention: exact` documents name every node exactly; `mention: informal` documents name none of them ("the feeds dim"). This forces the hard cases to exist rather than hoping the prose contains some.
- **Breakage is noticed in marts.** The first symptom of every incident is a mart (`models/mart/*`); the root cause is that mart or up to 4 hops above it. A first draw that ranked all models put 23 of 40 symptoms on staging and intermediate models (staging models have the most downstream nodes), which is not where people notice problems, and drew no `incremental_gap` at all.
- **Skew on purpose.** Symptom marts are drawn with Zipf-like weights, ranked by downstream size and test count, so a few nodes collect several incidents. Exactly 6 incidents are decoys (a fixed count; a 15% chance per incident gave only 2 on the first Cal-ITP draw): same symptom and test as an earlier one, a different cause.
- **Size:** 40 incidents, 12 runbooks, 25 Slack threads (15 about an incident, 10 unrelated questions as noise). Committed under `corpus/calitp/`. They are our own text; they only use Cal-ITP model and test names, so nothing AGPL is copied.

## Options compared
| | **Seeded scaffold + hand prose (chosen)** | Templates only | LLM-generated end to end |
|---|---|---|---|
| Facts agree with lineage | checked in code | checked in code | must be checked after the fact |
| Varied wording | yes | no; keyword search looks too good | yes, but one model's style |
| Reproducible | the facts are; the prose is committed | fully | no |
| Eval independence | ground truth in front-matter | same | the generator and the answerer may be the same model family |
| Cost | hours of writing | minutes | a few runs |

## Consequences
- The prose has one author's style. Questions in 3c are written separately, and a share of them are phrased loosely, to limit how much that helps any retriever.
- The scaffold's seed and counts are recorded in `corpus/calitp/corpus.yml`. Changing them means starting the prose again, so they are fixed once the writing starts.
- Changing the manifest can make the corpus stale; `corpus check` names each broken reference, as `eval search-set` does for questions.
- Dates are synthetic (January 2025 to October 2026) and don't match Cal-ITP's real history.

## Result (Cal-ITP, seed 14)
- 40 incidents (14 source outages, 11 logic bugs, 8 duplicate rows, 7 schema changes; 11 informal; 6 decoys), 12 runbooks, 25 Slack threads (15 about incidents, 10 unrelated questions). `corpus check` passes on all 77.
- `dim_schedule_feeds` is the symptom of 9 incidents with four different causes, which is the hardest case for "how was it fixed last time?".
- No `incremental_gap` was drawn. Cal-ITP's 62 incremental models are mostly real-time and quality models, a long way from the busy marts, so the 10% weight rarely finds a candidate within 4 hops. We kept that rather than forcing one, so the cause mix follows the project.
- The prose sometimes refers to earlier incidents ("the same failure mode as inc-003"). That is realistic, and it gives step 3e cross-references to follow.
