# 1. Store lineage in Postgres, not a graph database

**Status:** accepted, 2026-11-21

## Context
Impact questions ("what breaks if I change X?") need graph traversal. Catalogs like Atlan and DataHub use dedicated graph stores. Our graph is a dbt DAG: the Cal-ITP sample has 865 nodes and 1,198 edges, and large dbt projects reach a few thousand nodes.

## Decision
Store the DAG as an adjacency list (`nodes` + `lineage_edges`) in Postgres and traverse it with recursive CTEs. Sources, seeds, snapshots and models share one `nodes` table so every edge is an ordinary foreign key.

## Consequences
- One database for lineage, vectors (pgvector) and app data: one backup, one connection string.
- Recursive CTEs on a few thousand nodes are fast with indexes on both edge directions.
- Multi-hop path queries are more verbose in SQL than in Cypher.

## Alternatives considered
- **Neo4j:** nicer query language, but another service to run for a graph that fits in memory.
- **Graph in application memory (networkx):** simple, but every process loads the whole graph, and SQL joins to columns and tests are lost.
