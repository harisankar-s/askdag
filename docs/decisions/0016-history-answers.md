# 16. History answers: code lists the incidents, the LLM summarises with citations checked in code

**Status:** accepted, 2026-11-22

## Context
"Has this broken before, and how was it fixed?" is the first answer in this build where the LLM has to write substance and not just highlights. The cause and the fix are in prose that only language can condense. ADR 0007's rule still applies: compute what is computable, and generate only what needs language. Step 3c (ADR 0015) chose hybrid search and showed that no method abstains on its own.

## Decision
- **`history` is a fifth intent in the same parse call** (ADR 0013), with no node required. The question goes to incident search as asked. Name links are not used (ADR 0015 addendum), so there is nothing to resolve.
- **Code writes the facts.** It takes the top 3 incidents from hybrid search (keyword only when `EMBED_PROVIDER=none`), and their id, date, severity, title, the section that matched and the `url` back to the record come from SQL.
- **The LLM writes at most 4 sentences** from those incidents' Summary, Root cause and Fix sections, or says `answered: false` when none of them is about the question. Every sentence is checked in code, with one retry carrying the problems (`generate_checked`):
  - it cites at least one incident, and only incidents it was given;
  - any incident id in its text is one it cites;
  - any model or source it names is one that a cited incident names, using the same name detection as the index;
  - any number it uses (year, count, percentage) appears in a cited incident's text or date. Each incident's date is given to the model.
  - `answered: false` comes with no sentences.
- **If no reply passes, or there is no LLM, the list stands on its own.** The message says it is unsummarised and asks the reader to check that the incidents match.
- **Abstention is the model's call, made visible.** With `answered: false`, the response still shows the closest incidents but says none of them answers the question. A score cutoff isn't possible: 3c had 2 unanswerable questions, too few to set one.

## Options compared
| | **List by code + checked summary (chosen)** | Free-form LLM answer over retrieved chunks | List only, no LLM |
|---|---|---|---|
| Invented incidents or ids | impossible (ids are checked) | possible | impossible |
| Invented node names | rejected by the check | possible | n/a |
| Invented years, counts, percentages | rejected by the check | possible | n/a |
| "How was it fixed" in a sentence | yes | yes | the reader opens each incident |
| Says when nothing matches | yes (`answered: false`) | if the prompt works | no |
| Paraphrase can still be wrong | yes; only names, citations and numbers are checked | yes | n/a |

## Consequences
- The checks catch invented names, citations and numbers, not a wrong paraphrase of a cited section. Checking meaning would need a judge model, which this build doesn't use (CLAUDE.md, evaluation approach). The cited ids let a reader verify each sentence in one click.
- Top 3 is a fixed number. 3c scored hit@3 15/15 for hybrid on Cal-ITP, so the right incident is almost always among the three given.
- The golden set gains 3 history questions (version 4): two that name no node, and one that names a node and must not be read as lineage. They check the intent only; retrieval is `eval incidents`'s job.

## Addendum: first run (Cal-ITP, qwen2.5:7b)
- "Has dim_stops broken before, and how was it fixed?" found inc-008, inc-039 and inc-023, which are the right three, including the informal one. The summary cited the right incidents and passed every check, but it said **"7 May 2023"** and **"15 October 2023"**. The incidents are from 2025 and 2026, and their text says "On 7 May" with no year, so the model invented the years. That is exactly the paraphrase error the first version of the checks let through.
- Numbers can be checked in code, so they now are: a number in a sentence must appear in a cited incident's text or date, and the prompt gives each incident its date. That is ADR 0007's rule again: what can be computed or checked is not left to the model.
- "Has the cloud billing export ever arrived late?" returned the three closest incidents (all unrelated), and the model answered `answered: false`, so the response said that no recorded incident answers it. That is the right abstention, from one call of 15 output tokens.
- Rerun with the number check and dates in the prompt: the same three incidents, and the summary gave the right years (2025, 2026) and "about 8%" from inc-039, passing on the first attempt. Its two sentences explain what went wrong but barely say how each incident was fixed (revert and full refresh), and it left out the third incident (inc-023). Neither can be checked in code; it is the limit named above.
