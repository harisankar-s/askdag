# 20. A second LLM: Claude Haiku 5.5 matches qwen2.5:7b; the local model stays the default

**Status:** accepted, 2026-11-22

## Context
Since ADR 0019, the LLM does only two jobs: it reads a free-text question (intent, node, direction, depth, follow-up) and it summarises retrieved incidents in checked, cited sentences (ADR 0016). Everything else is code. The question for Phase 5 was whether a hosted model does those jobs better than the local default, qwen2.5:7b on Ollama, and at what cost.

## Decision
- **qwen2.5:7b stays the default.** It needs no key and no account, sends nothing off the laptop, and on this golden set it is within one turn of the hosted model.
- **Claude works through the existing OpenAI-compatible adapter, with no new code path.** Four settings:
  `LLM_BASE_URL=https://api.anthropic.com/v1/`, `LLM_MODEL=claude-haiku-5-5`, `LLM_RESPONSE_FORMAT=off`, `LLM_TEMPERATURE=null`, plus the key in `LLM_API_KEY`.
  - Anthropic's endpoint accepts only `strict: true` schemas, and ours would need reworking to qualify. With `off`, the schema goes in the prompt and every reply is validated in code and retried once, as for any server.
  - Claude Haiku 5.5 rejects `temperature`, so `LLM_TEMPERATURE` (default 0) can now be left out with `null`.
- **No native Anthropic adapter.** The narrow interface (ADR 0006) made the hosted model a settings change. A native adapter would get schema-constrained output, but nothing in these results asks for it.

## Measurements
Golden set `ce5ccf78eedc` (53 items, 63 turns), Cal-ITP, the same code for both runs, one run each. Embeddings are local (`nomic-embed-text`) in both.

| | qwen2.5:7b (Ollama, schema-constrained) | claude-haiku-5-5 (API, validated after) |
|---|---|---|
| correct / acceptable / wrong | 61 / 1 / 1 (silent) | 62 / 1 / 0 |
| question reading: right node, direction and depth | 24/25 | 25/25 |
| follow-ups that name a new node | 1/2 | 2/2 |
| parse: first attempt passed | 41 of 45 (4 retried) | 44 of 45 (1 retried) |
| parse latency p50 / p95 | 683 / 795 ms | 712 / 1,101 ms |
| history: first attempt passed the checks | 3/3 | 3/3 |
| history latency p50 / p95 | 2,847 / 6,576 ms | 3,166 / 3,727 ms |
| tokens per parse call, in / out | 1,235 / 27 | 1,757 / 63 |
| tokens per history call, in / out | 1,099 / 110 | 1,608 / 713 |
| wall time per turn p50 / p95 | 0.67 / 1.53 s | 0.71 / 3.30 s |

- **The one difference in outcome** is `t-new-node` turn 2, "What about payments_entity_mapping?" after a downstream question. qwen read it as upstream; Haiku kept downstream. The two previous qwen runs (2026-11-21) got this turn right, so this is run-to-run variation of a 7B model at temperature 0 on Ollama, not a stable gap.
- **Haiku retried less without schema-constrained decoding** (1 retry against 4). Its first replies followed the prompt's schema more reliably than qwen's constrained ones followed the parse rules.
- **Token counts differ by tokenizer,** so they compare cost, not prompt size. A Haiku golden run used about 86k input and 5k output tokens. Multiply by the current price on Anthropic's pricing page for the cost of a run; qwen's cost is laptop time.
- **Haiku's history answers were about six times longer in tokens** and still passed every citation, name and number check. Three turns is an anecdote, not a rate.

## Options compared
| | **qwen2.5:7b local (default)** | Claude Haiku 5.5 via the compatible endpoint | Native Anthropic adapter |
|---|---|---|---|
| Golden outcome | 61 / 1 / 1 | 62 / 1 / 0 | same model, not run |
| Needs a key and network | no | yes | yes |
| Data leaves the laptop | no | question and incident excerpts | same |
| Output constrained to schema | yes | no; validated and retried | yes |
| Code to add | none | none (two settings handled) | an adapter, tests, a dependency |

## Consequences
- **One run per model cannot separate them.** The gap is one turn of 63, the same size as the variation seen between qwen runs. Deciding on quality alone would need repeated runs and a bigger golden set; no choice here depends on that, because the default is set by privacy and setup, not by this table.
- **The wrong answer pointed at a rule, not a model, so code now applies it** (`keep_previous_direction` in `answer/parse.py`). A lineage follow-up with no word that could give a direction ("What about x?") keeps the previous direction, whatever the model said. The list of cue words is broad (up, feed, depend, read, use, break, affect, from, source, where and others), and node names are left out of the search, so the rule fires only on bare follow-ups and never overrides a direction the question gives.
- **Hosted use sends text to a third party.** Here it is synthetic. A team pointing this at real incident write-ups should treat the provider like any other system that sees them.
- **Cut:** Sonnet as a quality ceiling. With the small model already at 62 of 63, a bigger model has nothing left to show on this set.
