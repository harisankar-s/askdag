# 12. A separate embedding provider interface, with task prefixes and a fixed dimension

**Status:** accepted, 2026-11-21

## Context
Phase 2 adds semantic search: questions that describe a node ("which models have bus stop locations?") rather than name one. That needs an embedding model for node text and for questions. ADR 0006 already put the chat model behind a narrow interface, with `none` as the default. Embeddings need the same freedom, but they differ from chat in three ways:
- They are often set up separately in real deployments: a hosted chat gateway with a local embedder, or the reverse.
- Many retrieval models embed questions and indexed text differently. `nomic-embed-text` expects `search_document: ` and `search_query: ` prefixes; e5 uses `passage: ` and `query: `; OpenAI's models use none. Leaving the prefixes out raises no error, and recall just drops.
- The vector column has a fixed dimension. A different model usually means a different dimension, so the whole table has to be embedded again.

## Decision
- **Its own Protocol**, `EmbeddingProvider.embed(texts, kind)`, with `kind` being `document` or `query`. It returns `Embeddings` (vectors, model, input tokens, latency). It is not a method on `LLMProvider`.
- **Its own settings** (`EMBED_PROVIDER`, `EMBED_BASE_URL`, `EMBED_MODEL`, `EMBED_API_KEY`, `EMBED_DIMENSIONS`, `EMBED_BATCH_SIZE`), so the chat and embedding models can live on different servers.
- **Prefixes are settings** (`EMBED_DOCUMENT_PREFIX`, `EMBED_QUERY_PREFIX`), with nomic's as the defaults. The adapter adds them; callers only say which kind of text they send.
- **The adapter checks every vector's length against `EMBED_DIMENSIONS`** and stops with an error naming that setting. Otherwise the mismatch would show up later as a Postgres error on insert.
- **One adapter for the OpenAI embeddings API**, reusing the `openai` client, so there's no new dependency. It requests `encoding_format="float"` (the client's default is base64, which some compatible servers don't send). Errors map to `EmbeddingError`, as chat errors map to `LLMError`.
- **`EMBED_PROVIDER=none` is the default.** Search then uses keywords only (step 2c), so a fresh clone still works.
- **`HashingEmbedder`** (feature hashing of words, deterministic, offline) lets CI test real pgvector queries. It shares words, not meaning, so it never measures search quality.
- **Default model: `nomic-embed-text` on Ollama.** It has 768 dimensions, reads up to 8k tokens, and is about 270 MB.

## Options compared
| | **nomic-embed-text (chosen)** | mxbai-embed-large | bge-m3 |
|---|---|---|---|
| Dimensions | 768 | 1024 | 1024 |
| Context | 8k tokens | 512 tokens | 8k tokens |
| Size | ~270 MB | ~670 MB | ~1.2 GB |
| Fit | long descriptions with columns fit; runs on a laptop | slightly better benchmark scores, but too short for a model with its columns | multilingual; heavier than an English dbt project needs |

## Consequences
- **Switching to a model with a different dimension is a migration.** Change `EMBED_DIMENSIONS`, change the column type by hand (the schema file is `IF NOT EXISTS` and won't alter it), and embed every node again. Step 2b stores the model name on each row, so a model switch can be detected rather than silently mixing vectors from two models.
- **A wrong prefix is invisible.** It can only be caught by the search eval (step 2e), which is one reason the questions are written before chunking is chosen.
- **First reading, one question and two documents (`embed-check`, nomic-embed-text on Ollama):** with prefixes, the stops document scored 0.665 and the fares document 0.503 (gap 0.162). Without prefixes, they scored 0.735 and 0.526 (gap 0.209). So on this one pair, the prefixes narrowed the gap rather than widening it. One pair proves nothing either way, and scores from different settings can't be compared directly (the prefix shifts every vector). The prefixes stay on because the model card says to use them, and 2e measures recall@k both ways.
- **Ollama may run the model with a shorter context than 8k and cut long input without warning.** Chunk sizes in 2b have to allow for this.
- **pgvector cannot index more than 2,000 dimensions** for the `vector` type, so the setting is capped there.

## Alternatives considered
- **One provider for chat and embeddings:** fewer settings, but it forces both models onto one server and one key.
- **A per-model prefix table in code:** less setup for known models, but it goes stale and hides a choice that changes results. Settings with nomic defaults keep it visible.
- **The `pgvector` or `sentence-transformers` Python packages:** sentence-transformers would run the model in-process but pulls in PyTorch, which is several GB. Whether to add `pgvector` for adapting vectors to Postgres is decided in 2b.

## Addendum: chunks and storage (step 2b)
- **Several chunks per node, by kind:** `summary` (name and description, always present, so undescribed models can still match by name), `columns`, and `sql`. Each starts with `name (type, folder)`. Text is split at about 2,000 characters (about 500 tokens): the longest Cal-ITP description, column list and SQL are each about 20–24k characters, over what Ollama may accept. A node ranks by its best chunk. One embed run is enough to compare summary only, plus columns, and plus SQL in the eval, by filtering on `kind`.
- **`node_chunks` has no foreign key to `nodes`.** `load-manifest` replaces every node, so a cascade would discard every embedding on each load. `asklineage embed` re-embeds a chunk only when its text hash or the model changed, and deletes chunks of nodes that are gone. Search joins to `nodes`, so a chunk left behind before the next `embed` never shows.
- **No vector index yet.** A few thousand rows is an exact scan in milliseconds. An approximate (HNSW) index would add its own misses to the recall numbers. Index choice is a Phase 5 experiment.
- **No `pgvector` Python package.** Vectors are sent in pgvector's text form (`'[...]'::vector`). The package needs numpy, and only writes and distances are needed here.
- **The embed command uses an autocommit connection**, so each batch of 64 commits on its own and an interrupted run keeps what it finished.
- **Vectors are tagged with the model and the document prefix** (`index_name`, for example `nomic-embed-text [search_document:]`). With the model name alone, changing the prefix would have kept the old vectors without any warning. A new tag re-embeds everything, and search only uses vectors whose tag matches the current settings.

## Addendum: search eval (step 2e)
- `asklineage eval search` runs the hand-written set through six configurations: semantic or keyword, each with chunk kinds summary, summary + columns, and summary + columns + SQL. It reports hit@1/3/5/10, MRR and recall@10, overall and by category. For a "nothing found" cutoff on the top score, it shows how many unanswerable questions each cutoff catches and how many answerable ones it blocks.
- A hit means an expected model, and expected models are marts only. A staging model ranked above its mart counts as a miss.
- Comparing prefixes on and off means running `embed` with each setting in turn, because only one set of vectors is stored at a time. Each switch takes about 60 s on Cal-ITP.

### First results (Cal-ITP, 30 answerable and 5 unanswerable questions, set 9749f350fffb)
| config | prefix | hit@1 | hit@5 | hit@10 | MRR | recall@10 |
|---|---|---|---|---|---|---|
| semantic: summary | on | 11 | 17 | 22 | 0.464 | 0.586 |
| semantic: + columns | on | 12 | 18 | 20 | 0.497 | 0.586 |
| semantic: + columns + sql | on | 13 | 19 | 21 | 0.525 | 0.619 |
| semantic: + columns + sql | off | 11 | 20 | 20 | 0.477 | 0.603 |
| keyword: + columns + sql | – | 2 | 9 | 13 | 0.161 | 0.392 |

- **Semantic beats keyword by a wide margin:** hit@5 19 vs 9, MRR 0.525 vs 0.161. OR'd full-text over these questions is weak. Keyword search is also slower (about 330 ms, because each query builds the text index from scratch) than semantic search (about 30 ms).
- **Each chunk kind adds a little:** columns help the column questions, and SQL helps one undescribed question. On 30 questions, a difference of one or two is within noise. All three kinds stay.
- **Prefixes stay on.** Their effect on ranking is mixed: hit@5 is one worse, but hit@1 and MRR are better. They clearly help the "nothing found" cutoff. With prefixes, a cutoff of 0.667 catches 3 of 5 unanswerable questions and blocks 1 of 30 answerable ones. Without prefixes, 3 of 5 are caught only by blocking 3 of 30. This reverses the single-pair reading from `embed-check`.
- **A score cutoff alone can't decide "nothing found":** catching all 5 blocks 6 of 30 answerable questions. The cutoff becomes a low-confidence flag, and a second check is needed for a real abstain.
- **Missed in every semantic configuration:** c04 (headway), c05 (occupancy), d03, d07, d08 and u01 (mechanical failures). The answers sit in snake_case column names inside long column lists, or in near-identical NTD templates.
- **Splitting snake_case in column and SQL chunks** (`occupancy_status` becomes "occupancy status"): all kinds now give hit@1 12, hit@5 20, hit@10 22, MRR 0.519, recall@10 0.636. Before: 13, 19, 21, 0.525, 0.619. That's within noise. Undescribed questions went from 4/6 to 5/6, and the cutoff improved (4 of 5 caught with 1 blocked, before 2 blocked). It stays. The three target questions (c04, c05, u01) still miss, so word splitting wasn't their problem.
- **Several misses find the right model family in the wrong layer.** For u01, the top hit is `stg_ntd__breakdowns_by_agency`, the staging model of an expected mart. For d08, it's the source `transit_technology_stacks__contracts` that feeds `dim_contracts`. For d03, the top hits are sibling marts of the expected `fct_scheduled_trips`.
- **Family hits** (a node up to 2 hops upstream of an expected mart counts): with all chunk kinds, fam@1 is 18 against a strict hit@1 of 12, and fam@5 is 23 against a strict hit@5 of 20. In 6 of 30 questions, the top result is the right data one layer too early. Lineage-aware ranking could gain up to 6 at rank 1, which is more than any chunking change did.

## Addendum: lineage-aware ranking
- Semantic search takes the top 30 candidates. A candidate outside `SEARCH_PREFERRED_PATHS` (default `models/mart/*`) lifts up to 3 of its preferred descendants within 2 hops to its own score. Non-preferred nodes keep 98% of their score, so the mart ranks just above the node it came via, and the hit records that node as `via`.
- **Trade-offs:**
  - dbt has no layer field, so "preferred" is a project path convention set in config.
  - The cap of 3 lifts per node stops a widely used source from filling the results.
  - The 0.98 factor also slightly demotes non-mart nodes that have no mart below them.
- **Result on Cal-ITP (all chunk kinds):** hit@1 rose from 12 to 16, hit@5 from 20 to 22, hit@10 from 22 to 24, MRR from 0.519 to 0.614 and recall@10 from 0.636 to 0.686. Undescribed questions are now 6/6. The cutoff table is unchanged, because a lifted score equals the matching node's own score. Latency rose by about 10 ms (p50 43 ms). This is the largest gain in Phase 2, and it comes from the lineage graph, not the embedding model. It is on by default.
- **Flooding is visible but cheap.** On u01, external NTD sources lift 7 of the top 10 slots, and the expected mart still ranks first. fam@1 is 18 against a strict hit@1 of 16, so 2 questions still stop one layer early.
- **Still outside the top 5 with every setting:** c02, c04, c05, d03, d08, d09, d12 and j01.
