# 9. Conversation threads: opt-in, server-issued, checkpointed in Postgres

**Status:** accepted, 2026-11-21

## Context
6a's `/api/ask` is stateless: every request stands alone. Two things need memory between requests:
1. **Follow-up questions.** "And upstream?" only means something next to the previous question. A thin client (a chat box, later a Slack bot) can't be expected to send that context back. This is the main reason for server-side state.
2. **Clarifications.** After "`orders` matches 2 nodes", the user picks one. A stateless client can already ask again with `node=<unique_id>`; resuming only saves the second parse (about 0.6 s and 600 tokens on qwen2.5:7b) and keeps the direction and depth already read. A convenience, not a need.

LangGraph (ADR 0008) offers both through a checkpointer: state saved per `thread_id`, and `interrupt()` to pause a node until the client answers.

## Decision
- **Opt-in.** Requests are stateless unless they send `start_thread: true` (the server creates the thread) or `thread_id` (continue). One builder compiles twice: without a checkpointer for stateless requests, with `PostgresSaver` for threads.
- **The server issues thread IDs** (`gen_random_uuid()`, stored in our `ask_threads` table). Clients can't choose IDs, so they can't collide with or guess other threads. An unknown or pruned ID is a 404, not a silently empty new thread. There is no auth yet (Phase 6): a thread ID is effectively a bearer token.
- **Turn-scoped and thread-scoped state.** A checkpoint carries every key into the next question, so a stale `abstain_reason` or `root` would route the new question wrongly, and attempts would pile up. A `begin` node clears every turn-scoped key; the `attempts` reducer treats `None` as "start a new list". Only `previous` (the last answer's root, direction and depth) is kept across turns.
- **Clarify with `interrupt()`.** On a thread, when there are candidates to choose from, `clarify` interrupts with the clarification. The client answers with `{"thread_id", "choose": "<unique_id>"}` and the graph continues to `lineage` without reading the question again. Code checks the choice is one of the offered candidates (ADR 0004 holds: the user picks, code verifies). A choice that wasn't offered ends the turn with `reason: invalid_choice`; there is no re-interrupt loop. `choose` on a thread that isn't waiting is a 409. A new question on a waiting thread abandons the pending choice.
- **Follow-ups.** The parse prompt states the previous node. `ParsedQuestion.same_node` is true for a follow-up that names no node; the check rejects it when there is no previous node, or when it names a different node. A `same_node` follow-up reuses the remembered `Candidate` without resolving again. A missing direction or depth is kept from the previous question, whether or not the node changed.
- **One checkpoint per turn**: `durability="exit"` writes at the end of the turn or at the interrupt, not after each of the 5 steps. `answer` sets `result` (the whole walk) to `None`; the response holds the table already.
- **Registered state types.** The checkpoint serializer is built with an explicit allowlist (`STATE_TYPES` in `api/flow.py`). LangGraph 1.2 still deserialises unregistered types with a warning and says a future version will block them; ours fails now instead, and the thread tests run through the same serializer, so a missing type fails a test.
- **One turn at a time per thread**: `pg_try_advisory_xact_lock` on the thread ID, held for the request's transaction. A second concurrent request is a 409, not an interleaved checkpoint.
- **Retention**: the open-source saver has no TTL. `asklineage prune-threads --older-than-hours 24` deletes `ask_threads` rows unused for that long and each thread's checkpoints (`saver.delete_thread`). Running it on a schedule is the deployer's job.
- **Setup**: `asklineage init-db` runs `PostgresSaver.setup()`, which creates and migrates LangGraph's own tables (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations`). Those stay out of `schema.sql`; their schema belongs to the library.
- **A second, small pool** (max 4) for the saver: it needs `autocommit=True` and `prepare_threshold=0`, and the main pool runs requests in transactions.

## Options compared
| | Client sends context back | **Server threads (LangGraph checkpoints)** | Own sessions table |
|---|---|---|---|
| Follow-ups from a thin client | no | yes | yes |
| Server state to store, expire, lock | none | checkpoints + `ask_threads` | one table |
| Resume mid-flow | resend `node=<unique_id>` | `interrupt()` / `Command(resume=...)` | hand-written |
| Code we own | none | allowlist, `begin` reset, lock, prune | serialisation, resume logic, routing |
| Horizontal scaling | trivial | any instance can serve a thread (state in Postgres) | same |

## Consequences
- The stateless path is unchanged and stays the default; no client has to use threads.
- Threads add storage that grows until pruned, an extra pool, and four library-owned tables whose migrations run on `init-db`.
- A `same_node` follow-up is trusted to mean the previous node: code can check that it names no other node, not that the user meant the same one. The response echoes `node`, `direction` and `depth`, so a wrong reading is visible.
- The stateless graph and the thread graph share every node, so 6a's tests still cover both.

## First runs: qwen2.5:7b on Cal-ITP (6b)
- **Dependencies:** `langgraph-checkpoint-postgres` 3.2.1 is the only package added to the lock; its own dependencies (`psycopg`, `psycopg-pool`, `langgraph-checkpoint`, `orjson`) were already there.
- **A thread started with a typo** ("What does dim_schedule_feedz depend on?") returned a `not_found` clarification with `resumable: true` and the same five suggestions as 6a, `dim_schedule_feeds` first (0.89).
- **Choosing the suggestion, then asking "And downstream, two levels?"** on the same thread both returned lineage answers, as the author reported. Their latency and token figures weren't captured.
- **The parse prompt grew from about 600 to 881 tokens in**, for the follow-up rules and two more examples. Every free-text question pays this, follow-up or not. The parse took 3.5 s against 0.6 s in 6a; that call came right after a server restart, so part of it is probably Ollama reloading the model. Step 7's eval should measure parse latency on a warm model.
- **Operational lesson:** `uv add` rewrote `.venv` while `uvicorn --reload` was running. The reloader restarted midway through the install, the new worker failed to import, and the reloader kept port 8000 open, so requests hung instead of failing. Stop the server before changing dependencies.

## Known gaps
- A turn that fails after `start_thread` rolls back its `ask_threads` row, but its checkpoint (written on the autocommit pool) stays; `prune-threads` can't see it.
- The checkpoint pool opens on the first request, even if no request uses a thread.
- Threads have no owner until auth (Phase 6): anyone holding a thread ID can continue it.
