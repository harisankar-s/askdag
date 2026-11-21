# 4. Name resolution returns candidates instead of guessing

**Status:** accepted, 2026-11-21

## Context
Every lineage answer starts by turning what the user typed ("trips", "fct_ordrs", "settlements") into one node. In the Cal-ITP project, 13 names are shared by two nodes each (for example `settlements` exists under two Littlepay sources), and users misspell long names like `stg_gtfs_schedule__trips`. Whatever node we pick, the traversal and the grounding check then faithfully describe *that* node. A wrong pick here gives a correct-looking table for the wrong model, and the grounding check (Step 5) cannot catch it, because the summary does match the query result.

## Decision
Only an exact match resolves: a `unique_id`, a name (ignoring case), or `source_name.table` for sources. Several exact matches come back as `Ambiguous`. No exact match runs pg_trgm fuzzy matching (`greatest(similarity, word_similarity)`) and returns up to 5 suggestions as `NotFound`. A fuzzy match is never used to answer, however high its score.

## Consequences
- A typo costs the user one clarification turn ("did you mean `fct_orders`?"). The LangGraph flow (Step 6) needs a clarification path.
- Name-resolution accuracy can be measured on its own in the eval, separately from traversal.
- pg_trgm is a trusted extension, so the database owner can create it without superuser; the schema stays idempotent.

## Alternatives considered
- **Auto-pick the top fuzzy match above a threshold:** fewer turns, but any threshold is a guess, and a confident mistake here is the hardest kind of error to detect downstream.
- **Let the LLM pick from candidates:** moves the guess into a less predictable component, against the rule that the LLM never decides the lineage list.
- **Embedding similarity:** better for "the trips model" style descriptions; that's Phase 2 semantic search, not name lookup.
