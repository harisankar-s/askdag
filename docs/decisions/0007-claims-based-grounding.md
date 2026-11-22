# 7. Code writes the lineage facts; the LLM adds checked highlights

**Status:** superseded in part by ADR 0019 (2026-11-22): code now chooses and writes the highlights, so the LLM highlight, its prompt and its check are gone. The summary written in code stands. Originally accepted 2026-11-21, revised the same day after four runs on Cal-ITP; history below.

## Context
A lineage answer is the full table from the SQL traversal (ADR 0001, 0005) plus a short text summary. The table can't be wrong in ways an LLM causes, but LLM-written text can: it can name a model that isn't in the result, reverse an edge, miscount, invent an owner, or leave out that the result was cut off at the depth limit. The provider is swappable (ADR 0006), down to a 7B local model.

## Decision
- **Code writes the summary paragraph** from `LineageFacts`: the total and per-depth counts, the teams affected (downstream) or owning the nodes (upstream), where owners come from (`owner_source`), the unowned count, and truncation. These facts can't be wrong and are always present, with or without an LLM.
- **The LLM may add 1 to 3 highlights**, each a typed `relation` claim (`upstream`, `downstream`, `hops`, `text`). It sees only the root, the direction and up to `LLM_MAX_NAMED_NODES` named nodes (default 20) with their owners, with no counts to restate. Names rotate across owners, so every team and the unowned nodes get named even when one team owns almost everything.
- **The LLM picks nodes; code states their owners.** Owners are used to choose which nodes to highlight, but the highlight text may not mention owners at all. When a highlight is shown, code adds each result node's owner after its first mention: `` `fct_payments_rides_v2` (payments-team) ``, or `(no owner recorded)`.
- **Each highlight is checked in code:**
  1. *The relationship is true*: `direct` is an edge and `indirect` a path, in the stated direction.
  2. *The sentence matches it*: every backticked name is in the claim (or is the root); real node names without backticks are flagged; no numbers at all; no owner wording (owner, owned, team, or a team's name); "feeds" vs "reads from" agrees with the fields.
- **On failure:** one retry carrying the previous reply and the violations, then the highlights are dropped. The summary and table are always returned. `LineageAnswer.status` is `passed`, `passed_after_retry`, `rejected`, `off` (no LLM) or `skipped` (empty result), with every attempt recorded.
- **The retry loop is generic** (`answer/generate.py`, `generate_checked`): prompt, schema and check go in; checked output or `None`, plus attempts, come out. Phase 3 answers reuse it with their own schema and check.

## Options compared before choosing

| | A: LLM writes all claims | **B: code facts + LLM highlights** | C: template only | D: A on a bigger model |
|---|---|---|---|---|
| Facts can be wrong | Only if the check misses it | No | No | As A, less often |
| False alarms / fallbacks | High (heuristic phrase lists) | Low (checks are mostly exact) | None | Fewer real errors, same false alarms |
| Latency, local 7B | 6–18 s per attempt, up to ~33 s with retry | Facts instant; highlights after | ms | Hosted 2–5 s |
| Tokens per answer | ~3k in / 500 out, sometimes twice | ~1.5k in / ~150 out | 0 | As A, higher price |
| Model portability | Fragile; a prompt tweak flipped pass to fail | Moderate | Total | Better, still prompt-sensitive |
| Eval signal | Hallucinations mixed with check false alarms | Mostly real errors | n/a | As A |
| API shape (Step 6) | One prose string | `facts` fields + `highlights[]` | `facts` | As A |

C is B with the LLM off (`LLM_PROVIDER=none`), so one design covers both. D stays an eval row (Step 7), not the fix.

This does not shrink the RAG part of the project: for lineage, the SQL traversal *is* the retrieval and its result is complete, so generation adds least. The LLM's main jobs are elsewhere: understanding the question (Step 6), semantic search over descriptions (Phase 2), and synthesising incidents and runbooks with citations (Phase 3). The rule: compute what is computable; generate only what needs language.

## History: three runs with qwen2.5:7b on Cal-ITP (`dim_schedule_feeds`)
1. **First design (A), one `count` claim with optional `depth`.** All four attempts failed on `scope: "depth"` with no `depth`. Ollama compiles the schema into a grammar, and an optional field is one the model may skip. Split into required-field kinds. A retry given only the problems rewrote the reply and dropped a correct truncation claim, so the retry now carries the previous reply. Lesson: put every rule a schema can express into the schema.
2. **After that fix.** Downstream (107 nodes) passed on the first attempt. Upstream (9 nodes) failed once on a real hallucination, an edge from `external_gtfs_schedule.agency` to `int_gtfs_schedule__joined_feed_outcomes` that doesn't exist; the check caught it and the retry passed. But that passing summary described the depth counts backwards ("2 nodes read directly from the sources"), copying the only worked example, which was downstream. Added an example per direction and a direction rule for count sentences.
3. **After that.** Both questions fell back. Some rejections were real (hop counts off by one, a missing owner claim); others were my rule's false alarms (passive "2 are read directly" read as the wrong direction), and "48 read from nodes at depth 1" was true but broke the strict number rule. The retry repeated the same sentences. Across all runs, nearly every failure was in count, owner or truncation sentences, which the template already writes correctly; relation claims failed once, on a real error.

4. **Option B, with the LLM allowed to state owners.** Every relationship stated was true. Every failure was an owner statement, and the check caught all three: downstream, "`tides_agency`, with no owner recorded" (tides-team owns it), then a passing retry with two correct cross-team highlights (payments-team); upstream, two sources called unowned on both attempts (gtfs-team owns all 9 nodes). Two prompt causes: the upstream example said "the source `raw.payments`, which has no owner recorded", which the model generalised to all sources; and the rule to prefer unowned or other-team nodes pushed it to invent one when none existed. Owners moved out of the model's text into code-added tags, the examples lost their owner phrases, and the preference rule now falls back to the furthest nodes.
5. **Owners as code-added tags.** Both questions passed on the first attempt (about 6 s each, about 1.2–1.6k tokens in and 170–200 out), and every highlight was true. But the downstream highlights were the three alphabetically first gtfs-team models, even though payments-team, tides-team and transit-database-team nodes were named: the check proves highlights are true, not that they're useful. Measuring usefulness is the eval's job (Step 7).

## Consequences
- The summary's facts can't be wrong, and they're the same with any model or none.
- Highlights are thin until the model has more to reason with (descriptions in Phase 2, incidents in Phase 3); most lineage insights, such as cross-team impact, are computed in code.
- The text reads more like a report than prose.
- Known gaps: the direction phrase lists cover common English phrasings only; a highlight using none of them is checked on its fields alone. The root gets no owner tag, because the root isn't one of the result's nodes.
- The first-attempt pass rate of highlights per model is the eval's hallucination metric (Step 7).

## Alternatives considered
- **Keep A and extend the phrase lists:** each list fixed one phrasing and broke another.
- **A second LLM as judge at runtime:** a probabilistic check on a probabilistic answer; useful offline in the eval.
- **Show failed text with a warning:** defeats the rule that the LLM never decides what the lineage is.
- **`{{placeholder}}` templates filled by the model:** small models handle template syntax poorly, and it still can't say which relationships to pick.
