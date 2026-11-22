# 15. Incidents: a search index copied from the system of record, linked to nodes by detected names

**Status:** accepted, 2026-11-22

## Context
Phase 3 answers "has this broken before, and how was it fixed?". In a real team the incidents live in a ticketing tool (Jira, ServiceNow, PagerDuty) or a postmortem space, and that tool stays the system of record. Here, the Markdown corpus (ADR 0014) stands in for an export of it. Two questions follow: should we copy incidents into our own index, or query the tool at question time? And how does an incident get linked to dbt nodes, when real tickets rarely tag models?

## Decision
- **Index, not record.** `asklineage load-docs` copies incidents into `documents` (one row each, with `source` and `url` pointing back at the record) and `doc_chunks` (one chunk per section, each starting with a header of id, date, severity and title). Answers link to `url`; nothing is edited here.
- **Node links come from names detected in the text**, exact model names or `source_name.table`, stored in `documents.node_ids`. The front-matter's `symptom_nodes` and `root_cause_node` are the eval's ground truth and are never read by the app. Reading them would grade the system against its own input, and real tickets don't have them.
- **Only incidents are indexed.** Runbooks and Slack threads stay in the corpus, unloaded: the same loader would take them, but each type needs its own chunking and its own eval questions, and incidents alone already carry the trade-off (informal write-ups, decoys).
- **Full replace on each load, after embedding has succeeded.** Forty incidents are about 200 chunks, a few seconds to embed, so tracking which chunks changed isn't worth the code. A failed embed leaves the old index in place.
- **Vectors are optional.** With `EMBED_PROVIDER=none`, chunks are stored without vectors and keyword search still works (the generated `tsvector` column), as in Phase 2.

## Options compared
| | **Index copy (chosen)** | Query the ticketing tool per question |
|---|---|---|
| Semantic search | yes | no; ticket search is keyword only |
| Join with lineage ("incidents on anything upstream of X") | one SQL query | not possible inside the tool |
| Match a single section (Root cause, Fix) | yes | whole ticket only |
| Freshness | as of the last sync | live |
| Permissions | must be copied or filtered | enforced by the tool |
| Cost per question | local | an API call, subject to rate limits |

The lineage join decides it: "has anything upstream of `dim_schedule_feeds` broken?" needs the incident links and the lineage edges in one place.

## Consequences
- **Name detection misses informal write-ups** ("the feeds dimension"). That is the gap step 3c measures: name links against keyword, vector and hybrid search.
- **What a production build would add, not built here:**
  - an adapter per tool that syncs tickets changed since the last run, keyed on the issue key, instead of a full replace;
  - ticket permissions copied onto rows and filtered at query time;
  - deletes and merges propagated from the record.
- **Field mapping from a ticket:** issue key → `doc_id`, summary → `title`, description and resolution → sections, priority → `severity`, created → `doc_date`, browse link → `url`.
- `documents` and `doc_chunks` are keyed by project and id, so two projects can use the same ids.

## Result (Cal-ITP)
- `load-docs` indexed 40 incidents as 200 chunks in 3 s. Name detection fully links all 29 incidents that name their nodes exactly (63 links), and links none of the 11 informal ones. So name links alone can't find 11 of 40 incidents (28%), whatever the question.

## Addendum: which search finds incidents (step 3c)
17 hand-written questions (set edc781727e1c): 4 name a node, 5 describe a symptom, 3 are written loosely, 3 are decoys, and 2 have no answer. Cal-ITP, nomic-embed-text.

| method | hit@1 | hit@3 | MRR | p50 ms |
|---|---|---|---|---|
| names (detected names + 4 hops upstream) | 3/15 | 4/15 | 0.233 | 1 |
| keyword (full-text, any word) | **14/15** | 15/15 | **0.956** | 1 |
| vector | 11/15 | 15/15 | 0.844 | 16 |
| hybrid (RRF of keyword and vector) | 13/15 | 15/15 | 0.922 | 23 |
| routed (names first, then hybrid) | 12/15 | 15/15 | 0.889 | 29 |

- **Name links alone fail.** They answer only the 4 named questions and find nothing for the other 11. Even on named questions they return a set, not a ranking: ordering by date put a newer, less relevant incident first. Names are a filter, not a ranker, so `routed` comes out worse than plain hybrid.
- **Keyword search wins here, and the reason is known:** the same author wrote the prose and the questions, so they share words ("backfill", "duplicate feed versions"). ADR 0014 predicted this bias. Real tickets are written by many people, and that is where vectors earn their place.
- **The eval saturates at hit@3:** every method except names puts a right incident in the top 3 for all 15 answerable questions. With 40 incidents, only hit@1 and MRR tell the methods apart.
- **No method abstains.** All of them return results for both unanswerable questions. Two questions are too few to set a cutoff, so step 3d leaves that judgment to the answer step: it must say when the retrieved incidents don't answer the question.

**Decision: step 3d uses hybrid.** It is one question behind keyword on a set biased towards keyword, it holds up when the wording differs, and it falls back to keyword alone when `EMBED_PROVIDER=none`. `names` and `routed` stay in the eval as baselines and are not used by the app.
