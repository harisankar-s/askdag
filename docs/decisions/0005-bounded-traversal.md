# 5. Bound lineage traversal by depth with per-level dedup, not path tracking

**Status:** accepted, 2026-11-21

## Context
Upstream/downstream answers come from a recursive CTE over `lineage_edges` (see 0001). A dbt DAG is full of diamonds: in the Cal-ITP project one node reaches about 600 distinct descendants along 4,405 distinct paths, the longest chain is 17 hops, and `dim_schedule_feeds` has 59 direct children. The usual recursive-CTE cycle guard (a path array, or Postgres's `CYCLE` clause) lists every path, so its work grows with the number of paths, not with the number of nodes.

## Decision
- The recursive term uses `UNION`, not `UNION ALL`, on `(node_id, hops)`, so each node appears at most once per level. Work is capped at nodes × depth rows.
- A depth cap (default 5, maximum 25) is the termination guarantee. dbt rejects cycles, but a manual edge or a future cross-project merge could add one; the cap ends the walk anyway, and the root is never listed as its own descendant.
- Each node is reported once at its shortest distance (`min(hops)`).
- The result carries a `truncated` flag: true only if a node at the cap has a neighbour that wasn't returned. A neighbour already returned at a shorter depth doesn't count.
- The result includes the edges among the returned nodes, so the grounding check can verify edge directions.

## Consequences
- No path is returned, only distances; "how does A reach B?" needs a separate query if we ever want it.
- Answers can say honestly when they stop short ("showing 5 levels; there's more").
- A cycle is not reported as an error, only cut off; a data-quality check for cycles belongs in ingestion if it is ever needed.

## Alternatives considered
- **Path array or `CYCLE` clause:** gives paths for free, but costs 4,405 rows instead of about 600 on real data, and worse on wider projects.
- **No depth limit:** dbt's `+model` selector is unlimited, but an unbounded answer on a hub model is a 300-row table nobody reads, and the cap doubles as cycle protection.
- **Walk the graph in Python:** simple BFS with a visited set is optimal, but it costs a round trip per level or loading the whole graph; that's the trade 0001 already rejected.
