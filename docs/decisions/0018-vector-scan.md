# 18. Vector scan: nearest-chunks query without an index now, HNSW past about 10k chunks

**Status:** accepted, 2026-11-22

## Context
Node search and incident search rank by each node's (or incident's) best chunk. Until now the SQL did this with `DISTINCT ON (node_id) ... ORDER BY node_id, distance`: it scores and sorts every chunk. No vector index can serve that shape, so "add HNSW" would have changed nothing. An index needs `ORDER BY embedding <=> q LIMIT n`.

The Phase 5 question was whether pgvector's HNSW index is worth having, and from what size. Two losses had to be kept apart:
- **Crowding:** fetching the n nearest chunks, then keeping each node's best, returns fewer distinct nodes when one node owns several of the n.
- **Approximation:** HNSW can miss some of the true nearest chunks.

## Decision
- **Three scans in `search/scan.py`, set by `SEARCH_VECTOR_SCAN`:**
  - `exact`: the old query.
  - `topn`: fetch the nearest `k x SEARCH_OVERFETCH` chunks with index scans turned off, then keep each node's best.
  - `hnsw`: the same SQL served by the HNSW index.
  `topn` and `hnsw` differ only in the plan, so a difference between them is the index, not the query.
- **The default is `topn` with overfetch 4.** On Cal-ITP it gives the same answers as `exact` and halves p95 latency, with no index involved. The saving is the sort: `exact` sorts all 2,799 rows with their text, `topn` keeps the 40 nearest.
- **Switch to `hnsw` when `topn` p95 passes the query-embedding call** (about 15 ms locally). Below that, the embedding call costs more than the scan. On padded Cal-ITP data that is around 10k chunks, about 3k nodes. Run `asklineage eval vector-scale` on your own chunks first (see Consequences).
- **With HNSW, `SEARCH_HNSW_EF_SEARCH=100`**, not the fetch size. At 100k well-spread chunks it raised recall from 0.885 to 0.978 for 0.6 ms more at p95.
- **The HNSW indexes stay in `schema.sql`** on `node_chunks` and `doc_chunks`. On Cal-ITP they take 12 MB and build in 0.5 s. Switching is then one setting, not a migration. `topn` turns index scans off, so until then the index only costs writes.
- **The planner is never left to choose.** From 30k chunks it picks HNSW by itself, which would make answers approximate without anyone deciding so. `hnsw` mode forces the index (`enable_seqscan` and `enable_sort` off); `topn` forbids it.
- **Settings are scoped to one query.** `vector_scan()` sets them with `set_config(..., true)` in a savepoint and resets them from `pg_settings.reset_val`: one statement each way. A failed query rolls them back with the savepoint.

## Options compared
| | **`topn` now, `hnsw` past ~10k chunks (chosen)** | `exact` (before) | `hnsw` always | Let the planner choose |
|---|---|---|---|---|
| Answers on Cal-ITP | same as exact | baseline | same as exact | same |
| p95 on Cal-ITP, no lineage ranking | 8.0 ms | 16.7 ms | 8.8 ms (forced) | 8.0 ms (no index at 3k) |
| Growth with chunks | linear (88 ms at 100k) | linear, and slower | flat (1–5 ms) | flips at ~30k |
| Results exact | yes until switched | yes | no (0.98 recall at 100k) | silently no past ~30k |
| Index cost | built, unused | none | ~80% of table size | same as hnsw |
| One setting to change | yes | — | — | none: that is the problem |

## Measurements
**Cal-ITP, real data (2,799 node chunks, 200 incident chunks).** `eval search --kind-sets full --scan exact,topn,hnsw --overfetch 2,4,8`. Latency is the SQL alone: questions are embedded once beforehand (`WarmEmbedder`), and each config gets an untimed warm-up query.

| Node search, summary+columns+sql | p50 ms | p95 ms | hit@1 | hit@5 | MRR |
|---|---|---|---|---|---|
| exact | 13.3 | 16.7 | 12 | 20 | 0.519 |
| topn x2 | 5.1 | 6.3 | 12 | 20 | 0.519 |
| topn x4 | 6.8 | 8.0 | 12 | 20 | 0.519 |
| hnsw x4 | 7.9 | 8.8 | 12 | 20 | 0.519 |
| exact, lineage ranking | 17.3 | 21.4 | 16 | 22 | 0.614 |
| topn x4, lineage ranking | 10.1 | 14.9 | 16 | 22 | 0.614 |
| hnsw x4, lineage ranking | 10.5 | 14.6 | 16 | 22 | 0.614 |

- Every scan at every overfetch matched `exact` on all metrics, including recall@10, family hits and the cutoff table (same top scores). Crowding cost nothing even at x2. Overfetch 4 is the default anyway: long models are split into several SQL chunks, and x4 leaves room for one node to fill several places for about 1.7 ms.
- Incidents: all 23 method and scan rows matched (hybrid hit@1 13, MRR 0.922). Everything ran under 6 ms. At 200 chunks `exact` was fastest, because the setting statements weigh as much as the query.
- Embedding a question with `nomic-embed-text` on Ollama: p50 13.8 ms, p95 16.0 ms.

**Scale: padded copies of the real chunks.** `eval vector-scale`. Copy 1 holds the real vectors. Each later copy adds noise of a given length to every vector, normalises it, and counts as new nodes, so content width is real. The query is the 40-chunk fetch only, for the 35 search questions. HNSW uses m=16 and ef_construction=64, with a serial build (Docker's 64 MB `/dev/shm` is too small for a parallel one) and 1 GB `maintenance_work_mem`.

| chunks | table MB | index MB | index build s (noise 0.3 / 1.0) | exact p95 ms | planner picks HNSW |
|---|---|---|---|---|---|
| 3,000 | 14.6 | 11.7 | 0.5 / 0.4 | 3.6–5.1 | no |
| 30,000 | 144.6 | 117.2 | 4.4 / 6.9 | 46–53 | yes |
| 100,000 | 482 | 391 | 11.9 / 31.7 | 88–91 | yes |
| 300,000 | 1,445 | 1,172 | 51.6 (noise 0.3) | 233 | yes |

HNSW recall against the exact 40 nearest chunks. Strict recall counts the same chunks; tie-aware recall counts any chunk as near as the exact 40th, within 0.001 (the ann-benchmarks convention).

| chunks | ef_search | noise 1.0: p95 ms | strict | tie-aware | noise 0.3: p95 ms | strict | tie-aware |
|---|---|---|---|---|---|---|---|
| 3,000 | 40 | 0.8 | 0.940 | 0.961 | 0.9 | 0.949 | 0.971 |
| 3,000 | 100 | 0.9 | 0.995 | 0.999 | 0.9 | 0.996 | 0.999 |
| 30,000 | 40 | 1.3 | 0.917 | 0.951 | 1.1 | 0.654 | 0.698 |
| 30,000 | 100 | 2.0 | 0.986 | 0.999 | 1.3 | 0.806 | 0.851 |
| 30,000 | 200 | 2.7 | 0.999 | 0.999 | 1.5 | 0.861 | 0.901 |
| 100,000 | 40 | 3.1 | 0.885 | 0.920 | 1.1 | 0.249 | 0.279 |
| 100,000 | 100 | 3.7 | 0.978 | 0.996 | 1.1 | 0.264 | 0.296 |
| 100,000 | 200 | 5.1 | 0.996 | 0.999 | 1.6 | 0.273 | 0.306 |

- **Noise 1.0** puts a copy about as close to its original as two different real chunks are. Recall is healthy and `ef_search` trades it cleanly for latency.
- **Noise 0.3** makes 36 near-copies of every chunk. Recall collapses and raising `ef_search` barely helps. Tie-aware recall barely helps either, so these are real misses, not ties: the search gets stuck inside clusters of near-duplicates.
- An earlier run used `ef_search` = fetch size (40) at noise 0.3, and read the collapse as tie-breaking. The tie-aware column ruled that out.

## Consequences
- **Near-duplicates are the real risk for HNSW here.** dbt projects have them: staging models share boilerplate SQL, and doc blocks repeat column descriptions. A project's true recall sits somewhere between the two noise rows. That is why the switch is a setting someone chooses after running `eval vector-scale` on their own chunks, not automatic.
- **Padded data is synthetic.** It shows where latency crosses and how HNSW behaves at two cluster densities. It does not show the recall a real 100k-chunk project would get.
- **The bench times one query, not the whole search.** Lineage-aware ranking adds a second exact lookup by node id. That lookup never uses the vector index and is not timed at scale.
- **Forcing plans is blunt.** `enable_seqscan` and `enable_sort` off penalise every alternative for that query. They are reset straight after, so they never reach the rest of the request.
- `exact` stays for comparison and as the reference in tests. Eval labels without a scan tag are `exact`. The CLI's default eval run now uses the configured scan, so a default `eval search` labels its rows `semantic[topn x4]`.
- Cut: IVFFlat (it needs training data and a rebuild as data grows, and HNSW already answers the question), `halfvec` storage (it halves index size, worth trying if 768-dimension indexes outgrow memory), and Pinecone (a second store to keep in step with the lineage tables and the `embed` refresh, for sizes this table shows Postgres handles in a few milliseconds).
